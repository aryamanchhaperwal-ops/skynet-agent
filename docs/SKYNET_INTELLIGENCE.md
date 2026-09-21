# SKYNET Intelligence & Long-Term Memory (Phase P2)

Status: **implemented and tested** (62 dedicated tests + integration suite).
This document covers both halves of Phase P2: Skynet's own reasoning layer
(`agency.intelligence`) and persistent memory (`agency.memory`).

---

## 1. Intelligence architecture

```
Skynet Core (loop)
      │
      ▼
IntelligenceService          ← the single doorway to model reasoning
      │                        (one retry policy, one timeout policy, one usage log)
      ▼
LLMProvider (ABC)            ← provider seam
      │
      ├── MockLLMProvider            (deterministic, offline — default)
      └── OpenAICompatibleProvider   (OpenAI / Ollama / vLLM / OpenRouter)
```

Rules baked into the design:

- **No provider is hard-coded into the core.** The loop knows only the
  `Planner`/`Evaluator` ABCs; the service knows only `LLMProvider`.
- **Failure never destroys the core.** Every service-level call returns
  `None` on provider failure; the LLM planner and LLM evaluator fall back
  to their deterministic counterparts. The deterministic strategies are
  permanent residents, not scaffolding.
- **Model output is data, never truth.** Structured outputs are schema-
  validated; a malformed or schema-violating answer is discarded (falls
  back), never patched silently. LLM evaluations are recorded with
  `source="llm"` and never treated as objective verdicts.
- **Credentials live in the environment.** `OPENAI_API_KEY` is read at
  call time; a provider without its key refuses to activate. Nothing is
  ever committed or logged.

### Provider interface (`agency/intelligence/providers.py`)

| Member | Purpose |
|---|---|
| `info()` | Provider/model descriptor (id, model, capabilities, endpoint) |
| `capabilities()` | Declared `LLMCapability` set (generation, structured_output, streaming) |
| `generate(request)` | One completion; raises `LLMError` on failure |

`GenerateRequest` carries messages, model, temperature, max_tokens,
`json_mode`, timeout and metadata. `GenerateResponse` preserves provider,
model, latency, usage and finish reason.

The `mock` provider is fully scriptable via `scenario`
(`normal | failure | timeout | garbage | conflict`) so tests simulate
every failure path without a network.

### Planner integration (`agency/intelligence/planner.py`)

`LLMPlanner` wraps any `Planner` (in practice the `DeterministicPlanner`):

1. **Explicit goal steps always win** — `goal.params["steps"]` is honored
   verbatim without a model call (the deterministic contract, preserved).
2. Otherwise the model proposes steps through a strict schema
   (`{"steps": [{"type", "params"}], "rationale"}`).
3. Proposed steps are **filtered by the action allow-list** (hallucinated
   tool names are dropped) and capped by `max_steps_per_run`.
4. Any failure — provider error, timeout, unparseable JSON, schema
   violation, fully-filtered plan — falls back to the deterministic
   planner.

### Evaluator integration (`agency/intelligence/evaluator.py`)

`LLMEvaluator` judges whole runs (per-action judgment stays deterministic
by design — cheap, exact, and already sufficient). The model receives
goal, action records and bounded observation summaries, and must answer:

```json
{"score": 0.0-1.0, "passed": true, "reason": "...",
 "evidence": ["..."], "confidence": 0.0-1.0}
```

The returned `Evaluation` carries `source="llm"`, the model's reason and
evidence in `notes`, and its self-reported `confidence`. Any failure falls
back to `DeterministicEvaluator` (recorded as `source="deterministic"`).

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SKYNET_LLM_PROVIDER` | `mock` | `mock` or `openai` (OpenAI-compatible) |
| `SKYNET_LLM_MODEL` | `skynet-mock-1` | Model id handed to the provider |
| `SKYNET_LLM_BASE_URL` | empty | e.g. `http://localhost:11434/v1` for Ollama |
| `OPENAI_API_KEY` | empty | Read from the environment, never stored |
| `SKYNET_LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `SKYNET_LLM_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `SKYNET_LLM_MAX_TOKENS` | `1024` | Completion cap |
| `SKYNET_LLM_MAX_RETRIES` | `1` | Transient-failure retries |
| `SKYNET_PLANNER_STRATEGY` | `deterministic` | or `llm` (with fallback) |
| `SKYNET_EVALUATOR_STRATEGY` | `deterministic` | or `llm` (with fallback) |

Local Ollama example:

```bash
SKYNET_LLM_PROVIDER=openai
SKYNET_LLM_BASE_URL=http://localhost:11434/v1
SKYNET_LLM_MODEL=llama3.1
OPENAI_API_KEY=ollama   # Ollama accepts any placeholder key
```

---

## 2. Long-term memory architecture

```
        experiences / observations / AI conversations / human input
                              │
                              ▼  (deterministic extractors)
                     Memory Candidates
                              │ importance gate
                              ▼
                       MemoryManager            ← single doorway
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
         episodic         semantic        procedural
       (what happened)  (learned facts)  (what worked)
              └───────────────┼────────────────┘
                              ▼
                    MemoryStore (protocol)
                       ├── InMemoryStore        (tests)
                       └── SQLiteMemoryStore    (persists restarts; stdlib)
                       (vector/graph stores slot in behind the same protocol)
```

### Memory types

- **episodic** — what happened: research sessions, AI conversations,
  actions, failures, notable events.
- **semantic** — learned knowledge: facts, claims, findings, each with
  sources.
- **procedural** — strategies that worked (interface ready; populated by
  the consolidation loop in a later phase).

### Memory record

`MemoryRecord` (pydantic, JSON-serializable): `id`, `type`, bounded
`content` (≤ 20 000 chars, enforced at store time), `summary`, `source`,
`origin` (closed set: observation / experience / conversation / research /
human / consolidation / system), full **`provenance`** dict, `importance`
(0–1), optional `confidence`, `tags`, `goal_id`/`run_id` linkage,
`metadata`, `created_at`/`updated_at`.

### MemoryManager API

| Operation | Notes |
|---|---|
| `store(...)` | Importance-gated (`force=True` bypasses for operator inserts); returns `None` when rejected |
| `recall(query)` | **Budgeted** for prompt injection: `max_results` + `recall_max_chars`; never dumps the store |
| `search(...)` | Keyword + filters (type/tags/source/goal/time/importance), scored 0–1 |
| `get` / `forget` | Point lookup; explicit deletion is the only delete path |
| `stats()` | Census by type |

Relevance is deterministic token overlap weighted with importance — the
`MemoryStore` protocol is where vector similarity plugs in later, with no
changes above it.

### Provenance (mandatory)

Every derived memory traces back to its origin:

- observation memories carry `observation_id`, kind, timestamp and the
  observation's own provenance (e.g. web URL);
- conversation memories carry `conversation_id`, `participant_id`,
  `message_ids` — Skynet can always answer *"which AI said this?"*;
- AI-extracted claims are stored with `verified: false` — recorded, never
  auto-trusted;
- consolidated knowledge lists its `source_memory_ids`.

The raw streams (experiences table, trace events, conversation records)
are never replaced by extracted memories — extraction adds derived data,
originals stay untouched.

### Extraction (`agency/memory/extraction.py`)

- `memories_from_observations` — web research/tool outputs → semantic
  memories (confidence from the observation); errors → episodic with
  importance 0.7+ (failures are high-value learning material).
- `memories_from_conversation` — one episodic transcript-pointer memory
  per conversation + one semantic candidate per extracted claim.
- `memories_from_experiences` — errors 0.75, run summaries 0.65, failed
  actions 0.7; everything else stays below the default threshold (the raw
  stream already lives in the experiences table).

### Consolidation

`Consolidator.identify_candidates()` is the future-ready seam (many
experiences → recurring pattern → candidate knowledge). The deterministic
first version groups episodic memories sharing ≥ 2 tags and proposes one
semantic summary per recurring group (≥ 3 members), with source ids in
provenance. An LLM-driven consolidation implements the same interface
later; nothing else changes.

### Security boundaries

- Memory content is **data, never commands**: recalled memories enter
  `state.context["memory_recall"]` as structured records for the
  planner/evaluator to read; nothing in memory can execute, change
  configuration, or override system instructions.
- Stored content is capped (20 000 chars) — memory is not a bulk store.
- `origin` is a closed set; unknown origins are normalized to `system`,
  so external content cannot forge provenance categories.
- Deletion is explicit only (`forget`); nothing auto-expires knowledge.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SKYNET_MEMORY_BACKEND` | `memory` | `memory` (process-local) or `sqlite` (persists); `.env.example` ships `sqlite` |
| `SKYNET_MEMORY_SQLITE_PATH` | `data/skynet_memory.db` | SQLite file (created on first use) |
| `SKYNET_MEMORY_MAX_RESULTS` | `10` | Default recall/search cap |
| `SKYNET_MEMORY_IMPORTANCE_THRESHOLD` | `0.4` | Auto-store gate |
| `SKYNET_MEMORY_RECALL_MAX_CHARS` | `4000` | Recall budget injected into context |

---

## 3. Loop integration (how it all fits together)

```
GOAL
 └─► MEMORY RECALL  — relevant memories (budgeted) → state.context["memory_recall"]
                      (trace: MEMORY_RECALLED; a broken memory layer is never fatal)
 └─► OBSERVE → PLAN (deterministic or LLM) → ACTIONS → EVALUATE (deterministic or LLM)
 └─► memory_extract action — run observations → memory candidates → importance gate → store
      (trace: MEMORY_STORED per stored memory)
```

New actions: `memory_store`, `memory_search`, `memory_extract`
(category `memory`, always registered; availability of the backend is a
configuration matter, not a flag).

## 4. CLI

```bash
# Direct memory operations (independent of the agent loop):
.venv/Scripts/python.exe -m agency.cli remember "Fact worth keeping" --type semantic --importance 0.9
.venv/Scripts/python.exe -m agency.cli recall "agent memory" --limit 5 --json
.venv/Scripts/python.exe -m agency.cli memory-stats

# Full loop with persistent memory (survives restarts):
.venv/Scripts/python.exe -m agency.cli run --goal "Learn X" \
    --memory-backend sqlite \
    --steps '[{"type": "memory_store", "params": {"content": "...", "importance": 0.9}}]'

# Next process: what has Skynet previously learned?
.venv/Scripts/python.exe -m agency.cli recall "X"
```

## 5. Extension points

- **New provider**: subclass `LLMProvider`, add one factory entry in
  `build_llm_provider`. The service, planner, evaluator and memory layer
  are untouched.
- **Vector / graph store**: implement `MemoryStore` (search returns
  `(record, score)` pairs — semantic ranking slots straight in), extend
  `build_memory_store` with the backend name.
- **LLM importance scoring / distillation**: the extractors accept the
  same inputs an intelligence-backed implementation would; swap the
  deterministic scoring, keep the provenance plumbing.
- **Procedural memory population**: `Consolidator` proposals + run
  evaluations feed a future strategy-learning loop (Phase P7).
