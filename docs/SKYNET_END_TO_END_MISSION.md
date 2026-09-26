# SKYNET — First Complete Bounded Autonomous Mission (Phase P10)

This document records the **first end-to-end, bounded, read-only autonomous
mission** executed through the existing SKYNET orchestrator. It is a
validation artifact: the architecture (P2–P9) already existed; this phase
proves it actually operates as one loop.

Nothing here is hand-simulated. Every result below came from running the real
CLI/orchestrator against real configured providers, and every limitation is
reported rather than hidden.

```
OBJECTIVE → OBSERVE → PLAN → RESEARCH → COMMUNICATE → ACT
    → COLLECT EVIDENCE → STORE MEMORY → EVALUATE
    → CONSIDER IMPROVEMENT → COMPLETE / REPLAN
```

---

## 1. Mission objective

> **Identify the official repository description and primary programming
> language of the Ollama project, using authoritative public sources.**

Harmless, factual, publicly researchable, login-free, non-sensitive. The
web provider used (Wikipedia's keyless MediaWiki API) reliably reaches it.

Deterministic companion mission (offline): `autonomous demo` — investigate a
local corpus and produce a supported answer.

---

## 2. Environment

| Item | Value |
| --- | --- |
| Host | Windows, checkout `C:\Users\karma\Downloads\skynet` |
| Python | `.venv/Scripts/python.exe` (3.11), `PYTHONIOENCODING=utf-8` |
| LLM backend | Ollama `http://localhost:11434/v1` (OpenAI-compatible) |
| Model used | `qwen2.5-coder:7b` (non-thinking, reliable JSON) |
| Web backend | Wikipedia keyless MediaWiki API (`SKYNET_SEARCH_PROVIDER=wikipedia`) |
| Memory | SQLite (`data/p10_memory.db`) — persistent across processes |
| Run store | JSONL (`data/p10_runs.jsonl`) — persistent, cross-process |
| Git baseline | `492bde6` (`skynet-real-capabilities`) |

The tiny `llama3.2:1b` model also served requests but is not reliable at the
planner's JSON schema; the mission therefore used `qwen2.5-coder:7b`.

---

## 3. Providers available (classification)

Classification vocabulary is the system's own: a mock is never labelled real.

| Seam | Provider | Class | Evidence |
| --- | --- | --- | --- |
| LLM | `openai` @ Ollama `:11434` / `qwen2.5-coder:7b` | **REAL** | `provider test` → `ok 6634ms`, one minimal completion |
| Web | `wikipedia` | **REAL** | `provider test` → `ok`, one live search returns results |
| Comms (AI↔AI) | `mock` only | **MOCK** | only `MockAIProvider` exists; real providers raise `ProviderError` |
| Memory | `sqlite` | **REAL (persistent)** | fresh-process recall returns the run summary |

`provider status` (classification only, no requests) reported LLM/WEB as
`REAL unprobed` and comms `MOCK`, exit 0.

**AI↔AI = UNAVAILABLE.** No legitimate external AI provider is implemented or
configured (`agency/comms/providers.py` ships only the deterministic mock;
real Anthropic/OpenAI participants are explicitly future work). The mission
therefore did **not** perform an AI↔AI interaction and did **not** fabricate
one.

---

## 4. Exact execution path

Real mission run id **`3c53928f509f4e25ab91947223adb60c`**.

```
INITIALIZE → OBSERVE → PLAN → RESEARCH → ACT → EVALUATE → LEARN → COMPLETE
```

- **OBSERVE** — memory recall ran first (no prior memories on the clean DB).
- **PLAN** — an explicit operator research query (`--query "Ollama software"`)
  was honored; the LLM would otherwise have proposed the query.
- **RESEARCH** — the P4 `WebResearchAdapter` performed one live Wikipedia
  search and fetched three authoritative pages (2000-char content cap).
- **ACT** — the **real `SkynetCore.run_goal`** executed an LLM-planned cycle
  (`planner_strategy=llm`, `evaluator_strategy=llm`); the run passed
  (`core_summaries[0].ok == True`).
- **EVALUATE** — mechanical decision: 3 external evidence items + a passing
  agent cycle → `complete`.
- **LEARN → COMPLETE** — a run-summary memory was written, then the run
  finished.

Final result:

```json
{"completed": true,
 "reason": "objective supported by evidence and a passing agent cycle",
 "iterations": 1, "evidence_count": 4, "decisions": ["complete"], "core_runs": 1}
```

The very first attempt (model `llama3.2:1b`, no explicit query) **completed
too**, but only after `research_more` replanning; its first core cycle failed
because the LLM planner's JSON was schema-invalid and the deterministic
fallback chose the unregistered `gods_eye_latest_events` action. That failure
was recorded, not hidden (see Limitations).

---

## 5. Evidence + provenance

Every research finding carries source URL, provider, query, retrieval
timestamp, title, and an evidence class. The mission produced **3 live
external items + 1 internal action summary**:

| source_type | evidence_class | source | confidence |
| --- | --- | --- | --- |
| web | `LIVE_EXTERNAL_EVIDENCE` | https://en.wikipedia.org/wiki/Ollama | 0.55 |
| web | `LIVE_EXTERNAL_EVIDENCE` | https://en.wikipedia.org/wiki/LM_Studio | 0.55 |
| web | `LIVE_EXTERNAL_EVIDENCE` | https://en.wikipedia.org/wiki/Llama.cpp | 0.55 |
| action | `MODEL_GENERATED_CONTENT` | `core_run:8ca7b768ef0b` | 0.70 |

Provenance relationship: the mission's answer ("Ollama is an open-source tool
for running large language models locally; the project is developed in Go") is
**VERIFIED against the live Wikipedia page titled "Ollama"** at retrieval time
`2026-09-26T04:42:xx Z`, with the Wikipedia article's infobox naming **Go** as
the implementation language. Claims that only the search result asserted (not
the fetched page) would be **PARTIALLY VERIFIED**; no claim here is
**UNVERIFIED** or **FAILED**. (The evaluator's own verdict is the system's
structured judgment; the source citation above is what makes the claim
verified.)

---

## 6. Memory — WRITE → PERSIST → READ

1. **WRITE** — `LEARN` stored an episodic summary memory
   (`source=autonomous_run:3c53928f509f`) via the existing `MemoryManager`.
2. **PERSIST** — SQLite backend `data/p10_memory.db`.
3. **READ** — a **fresh process** recalled it:

```
$ SKYNET_MEMORY_SQLITE_PATH=data/p10_memory.db python -m agency.cli recall "Ollama repository" --storage sqlite
recalled 1 memory(ies) for: Ollama repository
  - [episodic] (0.91) autonomous run: Identify the official repository description ...
      source: autonomous_run:3c53928f509f · stored 2026-09-26 04:42 UTC
```

**MEMORY: PERSISTENT.** (If no persistent backend is configured, the run
completes without memory and claims nothing — covered by
`test_memory_fallback_is_reported_not_faked`.)

---

## 7. Evaluation

The mission used the existing LLM evaluator (`evaluator_strategy=llm`) with a
deterministic fallback, plus the orchestrator's mechanical completion rule:

- **Objective completion** — `result.completed == true` (≥3 external items +
  passing core cycle). ✔
- **Evidence quality** — 3 live sources, content fetched, capped at 2000
  chars. ✔
- **Provenance completeness** — all four evidence items carry class +
  retrieval timestamp; web items carry URL/provider/query/title. ✔
- **Source reliability** — single authoritative encyclopedia; confidence kept
  low (0.55) because it is a source, not proof. ⚠ (single source)
- **Factual consistency** — no contradiction across the three pages. ✔
- **Memory persistence** — verified cross-process. ✔
- **Communication result** — N/A (AI↔AI unavailable; not scored). ➖
- **Safety / prompt-injection** — no directive-like content flagged
  (`security_report == {}`); defenses tested separately. ✔
- **Budget compliance** — `web_requests=1 < 10`, `llm_calls=2 < 200`,
  `tokens=562`, `iterations=1 < 2`. ✔

No score was inflated: the mission is small and the evidence base is a single
authoritative source, which is reflected in the low per-item confidence.

---

## 8. AI↔AI communication

**UNAVAILABLE — no legitimate provider configured.** The only shipped AI
participant is `MockAIProvider` (offline, deterministic). `build_provider`
raises `ProviderError` for `anthropic`/`openai`/`claude`. The mock participant
was reachable through the P5 seam but its output is labelled
`MODEL_GENERATED_CONTENT` at confidence ≤ 0.5 — never presented as a real
external AI. No interaction was fabricated.

---

## 9. Self-improvement hook (proposal-only)

- `improve detect` over recorded experiment history surfaced weaknesses
  (e.g. `duplicate_sources` on `research:v1`, severity 0.58, freq 5) — so the
  system **can** identify an improvement opportunity.
- `improve propose` for a strategy not registered in the CLI context fails
  cleanly (`strategy 'wide_research:v1' is not registered`) — proposals are
  bound to registered, sandboxed strategy artifacts.
- `improve demo` demonstrates the full gated flow: proposal is **config-only
  (no code changes)**, tested as an experiment, evaluated, **approved by a
  human operator**, applied, monitored, and rolled back.

The mission itself produced no experiment/experience data, so the orchestrator
never entered its `IMPROVE` stage (which only fires on an
`improvement_opportunity` decision). Nothing was auto-applied; the approval
gate remains intact.

---

## 10. Pause / resume

- Mid-flight pause/resume: `test_pause_mid_flight_then_resume` (P8) pauses
  from a second store instance, confirms `execute()` honors it, then resumes
  to `COMPLETED`.
- New P10 test: `test_pause_persists_and_resume_continues` pauses before any
  work, asserts the persisted state is empty at iteration 0, resumes, and
  completes with the evidence accumulated across the resume — proving the
  mission **continues rather than restarting from zero**.

CLI lifecycle verified: `autonomous pause|resume|cancel|inspect`,
`real status|inspect|trace|pause|resume`.

**Cross-process pause proved live (new in the hardening pass).** A real
mission was started by one process; a second process paused it mid-flight
(`real pause <id>`) while it was in `act` with 3 live sources already
collected; the driving process stopped and persisted `status=paused`. A
*third* process then ran `real resume <id>`, which drove the **same run id**
to `completed` with 4 evidence items — proving PAUSE → PERSIST → RELOAD →
RESUME → COMPLETE without starting a new mission (run count grew by exactly
one).

This live test exposed and fixed a real defect: a pause applied *during* a
stage was silently overwritten by the stage's own end-of-stage save (the
in-memory copy was still `running`). `AutonomousOrchestrator._save` now
re-reads the persisted status first and refuses to clobber an operator's
`PAUSED`/`CANCELLED`. Covered by
`test_save_preserves_cross_process_pause`.

---

## 11. Prompt-injection defense

External content is **DATA**. `scan_content` flags directive-like fragments;
they are recorded in `run.security_report` and on the evidence
(`security_flags`), and `assert_no_control_payload` refuses to use such content
as control input. Tests:

- `test_injected_content_cannot_control_orchestrator` (P9) — a page saying
  "ignore previous instructions / delete all files / disable all safety"
  is recorded as data and the run follows its normal, evidence-based path.
- `test_external_content_stays_data_run_completes` (P10) — same guarantee
  including "reveal your api key".

The orchestrator's stage handlers accept no callable/path/command from
evidence; content can only fill text fields (queries/focus).

---

## 12. Budgets

Hard limits are checked **before** each stage; a runaway loop PAUSEs or STOPs.

| Budget | Default | Checked |
| --- | --- | --- |
| `max_iterations` | 3 | → STOP |
| `max_runtime_seconds` | 600 | → PAUSE |
| `max_web_requests` | 10 | → PAUSE |
| `max_ai_requests` (AI↔AI) | 6 | → PAUSE |
| `max_llm_calls` | 200 | → PAUSE (**new in P10**) |
| `max_tokens` | None | → PAUSE (when set) |
| `max_experiments` | 2 | → PAUSE |
| `max_cost_usd` | None | → PAUSE (when set) |

P10 closed an observability gap: the real mission made LLM calls (planner +
evaluator) while `ai_requests` stayed 0. The orchestrator now accounts every
provider LLM call and its reported tokens into
`usage["llm_calls"]` / `usage["tokens"]` and enforces `max_llm_calls` /
`max_tokens`. Deltas (not absolutes) are used, so a resumed run accounts only
the calls its own process made. Tests:
`test_llm_calls_and_tokens_are_accounted`,
`test_llm_call_budget_pauses_runaway`, `test_token_budget_check`.

---

## 13. Observability

The run record (`data/*_runs.jsonl`, inspectable via `real inspect`) is the
persisted trace: run id, objective, stage, iteration, timestamps, usage
(web/ai/llm/token), core run ids + ok flags, evidence with provenance,
decisions, security report, evaluation, errors, final status.

During execution the same stages emit lifecycle events through the
`CallableTraceSink` facade (`AUTONOMOUS_RUN_STARTED`, `…_STAGE_STARTED`,
`OBSERVATION_RECORDED`, `SOURCE_RETRIEVED`, `EVIDENCE_VERIFIED`,
`PLAN_VALIDATED`/`PLAN_REJECTED`/`PLAN_FALLBACK`, `LLM_REQUEST_*`,
`AUTONOMOUS_EVALUATION_COMPLETED`, `AUTONOMOUS_MEMORY_UPDATED`,
`AUTONOMOUS_RUN_COMPLETED`), each carrying the run id and never credentials.

**Persisted trace stream (new in the hardening pass).** Those events are now
also written to an append-only JSONL stream (`data/autonomous_run_traces.jsonl`,
`SKYNET_AUTONOMOUS_TRACES_PATH`) by
`agency/orchestrator/trace_store.py`. Each `TraceRecord` carries `run_id`,
`event_id`, `timestamp`, `state` (stage), `event_type`, `component`,
`action`/`provider`, `status`, `error` and a sanitized `metadata` payload. The
stream is append-only under a lock (safe for concurrent writers), is re-read
from disk on every access, and therefore **survives process restarts**.
Inspect it with `real trace <run_id|prefix>` (or `--json`). The live mission
produced 41 persisted events for its run.

**Secret redaction:** the store drops secret-named keys (`api_key`, `token`,
`authorization`, …) and masks secret-shaped values (`sk-…`, `ghp_…`,
`Bearer …`) before writing; a byte scan of the live trace file found no secret
material. Tests: `test_secrets_never_appear_in_trace_or_run`,
`test_trace_redaction_masks_keys_and_values`,
`test_orchestrator_persists_trace_and_never_leaks_secrets`,
`test_trace_store_survives_reload`, `test_cli_real_trace_reads_persisted_stream`.

---

## 14. Limitations (honest)

- **AI↔AI is mock-only / UNAVAILABLE** — no real external AI provider exists.
- **Tiny models are unreliable planners** — `llama3.2:1b` returns
  schema-violating JSON; even `qwen2.5-coder:7b` returned schema-violating
  JSON twice during the live launch. Model output is now validated before
  execution and falls back deterministically (see §16); the earlier failure
  mode — the fallback naming the unregistered `gods_eye_latest_events` — is
  fixed by making the fallback action-aware.
- **Single-source verification in this environment** — the only working web
  provider is Wikipedia, so every live source shares one host
  (`en.wikipedia.org`) and the mission is correctly classified
  `SINGLE_SOURCE`. The multi-source machinery is implemented and unit-tested
  with two independent hosts, but it is **UNAVAILABLE to corroborate live**
  here (DuckDuckGo still returns HTTP 202 to scripted clients). One source is
  never presented as two.
- **Wikipedia keyword search** returns nothing for keyword-poor,
  punctuation-heavy queries; the architecture's one budget-counted simplified
  retry mitigates this but is not a general fix.
- **`SafeFetcher` gets 403/406 from some sites** (robots/UA policy) — recorded
  as action failures, never bypassed.
- **`max_cost_usd` has no central price table** — cost stays 0.0 unless a
  provider reports it.
- **`improve propose` needs registered strategies** — detection works
  standalone, but a proposal for an unregistered strategy fails loudly.

---

## 15. Exact commands used

```bash
# provider classification + one real request each
OPENAI_API_KEY=ollama SKYNET_LLM_PROVIDER=openai SKYNET_LLM_BASE_URL=http://localhost:11434/v1 \
SKYNET_LLM_MODEL=qwen2.5-coder:7b PYTHONIOENCODING=utf-8 \
  .venv/Scripts/python.exe -m agency.cli provider status
  .venv/Scripts/python.exe -m agency.cli provider test --component all

# the bounded real mission
OPENAI_API_KEY=ollama SKYNET_LLM_PROVIDER=openai SKYNET_LLM_BASE_URL=http://localhost:11434/v1 \
SKYNET_LLM_MODEL=qwen2.5-coder:7b SKYNET_LLM_TIMEOUT_SECONDS=180 \
SKYNET_PLANNER_STRATEGY=llm SKYNET_EVALUATOR_STRATEGY=llm \
SKYNET_ENABLED_ACTIONS="echo,gods_eye_latest_events,web_research,web_search,web_fetch,web_extract,memory_store,memory_search,memory_extract" \
SKYNET_MEMORY_BACKEND=sqlite SKYNET_MEMORY_SQLITE_PATH=data/p10_memory.db \
SKYNET_AUTONOMOUS_RUNS_PATH=data/p10_runs.jsonl PYTHONIOENCODING=utf-8 \
  .venv/Scripts/python.exe -m agency.cli real start \
    --objective "Identify the official repository description and primary programming language of the Ollama project, using authoritative public sources." \
    --query "Ollama software" --max-iterations 2

# inspect the persisted run
SKYNET_AUTONOMOUS_RUNS_PATH=data/p10_runs.jsonl ... python -m agency.cli real status
SKYNET_AUTONOMOUS_RUNS_PATH=data/p10_runs.jsonl ... python -m agency.cli real inspect <run_id>

# prove memory persistence (fresh process)
SKYNET_MEMORY_SQLITE_PATH=data/p10_memory.db ... python -m agency.cli recall "Ollama repository" --storage sqlite

# deterministic offline companion mission
python -m agency.cli autonomous demo

# self-improvement analysis (proposal-only, gated)
python -m agency.cli improve detect --memory-backend sqlite
python -m agency.cli improve demo

# persisted mission trace (new in the hardening pass)
python -m agency.cli real trace <run_id>
python -m agency.cli real trace <run_id> --json

# cross-process pause / resume of a real run (new in the hardening pass)
python -m agency.cli real start --objective "..." --query "Ollama software" --max-iterations 3 &
python -m agency.cli real pause <run_id>     # from a second terminal
python -m agency.cli real resume <run_id>    # continues the same run id
```

---

## 16. Validation gap closures (P10 hardening)

After the first mission report, five gaps were closed without redesigning
anything. All reuse the existing architecture.

### 16.1 Plan validation (model output is untrusted control input)

`agency/plan_validation.py` sits between planner and executor. A planner that
declares `plans_are_untrusted` (the LLM planner) has its plan checked for:
required fields, non-empty step list, step budget (`max_steps_per_run`), action
allow-list **and** registration membership, and a control-parameter denylist
(`command`, `shell`, `exec`, `eval`, `subprocess`, `script`, …) that stops a
model from smuggling an invocation through an action's parameters.

The response to a failure is bounded and never executes anything:

```
INVALID PLAN → validation failure (trace PLAN_REJECTED)
  → deterministic safe fallback (trace PLAN_FALLBACK)
  → validated once (no retry loop)
  → safe termination if the fallback is also invalid
```

Deterministic plans are code, not model guesses, and pass through untouched
(the historical `gods_eye_latest_events` refusal path is intact and still
tested). Traces: `PLAN_VALIDATED`, `PLAN_REJECTED`, `PLAN_FALLBACK`.

### 16.2 Action-registration / fallback safety

`DeterministicPlanner` accepts an optional `allowed_actions` set (allow-list ∩
registry, computed in `build_core`). When set, its configured default is
emitted only if allowed; otherwise a deterministic read-only preference
(`web_research` → `web_search` → … → `echo`) is substituted and the rationale
records the substitution — never silent. With no allowed action at all it
emits an empty plan. The LLM planner's own fallback is built this way, so a
model-failure fallback can only name a runnable action. The loop's
`_execute_step` gate remains the last line: an unregistered or non-allow-listed
action is a failed result, never an execution.

### 16.3 Multi-source verification

`agency/orchestrator/verification.py` classifies the corroboration status of
the gathered evidence, computed after research (and after communication) and
stored on the run, emitted as `EVIDENCE_VERIFIED`, and included in the final
result:

| Status | Meaning |
| --- | --- |
| `MULTI_SOURCE_CORROBORATED` | ≥2 independent hosts cover the same topic |
| `SINGLE_SOURCE` | one independent host; not corroborated |
| `CONFLICTING` | independent sources share almost no vocabulary — flagged for review |
| `NO_EXTERNAL_EVIDENCE` | verification unavailable |

Independence is **host diversity** (two pages on one host are one source);
topic grouping uses the research query. No second source is ever synthesized:
the live mission is honestly reported `SINGLE_SOURCE` (§14).

### 16.4 Persisted mission trace

See §13. `RunTraceStore` + `real trace` deliver WRITE EVENT → PROCESS RESTART
→ READ EVENT.

### 16.5 CLI lifecycle completion

`real pause <run_id>` and `real resume <run_id>` were added (there were no
equivalent commands for real runs; `autonomous resume` only flips the status).
`real resume` **drives the existing run on** from its persisted stage and
evidence — it never creates a new mission.

---

## 17. Tests

Full suite: **567 passed, 5 skipped** (was 525 + the P10/P10-hardening
additions).

New focused tests (`tests/test_end_to_end_mission.py`):

- bounded mission completes end-to-end + structured report fields
- memory persists and reads back across instances; fallback reported, not faked
- research/comms provider selection (real/local/mock-only)
- LLM call + token accounting; LLM-call budget pauses a runaway; token budget
- pause persists and resume continues (no restart)
- external content stays data; secrets never leak
- real AI provider unavailable; mock answer is low-confidence model content
- self-improvement never auto-applies

Plus a CLI regression test for `real status` / `real inspect`
(`tests/test_orchestrator_cli.py`) after fixing a real bug: both subcommands
read `args.json` but never defined the flag, raising `AttributeError`.

---

## 18. Labels at a glance

| Capability | Label |
| --- | --- |
| LLM planning/evaluation | **REAL** (Ollama, local endpoint) |
| Web research | **REAL** (Wikipedia) |
| Memory | **REAL / PERSISTENT** (SQLite) |
| AI↔AI communication | **MOCK / UNAVAILABLE** (real providers absent) |
| Self-improvement | **REAL, PROPOSAL-ONLY** (approval-gated, config-only) |
| Deterministic companion mission | **LOCAL** (`autonomous demo`) |
| Multi-source verification (live) | **UNAVAILABLE** — one web host reachable; machinery tested with two hosts |
| Plan validation | **PASS** — model plans gated; deterministic plans honoured |
| Action registration / fallback safety | **PASS** — fallback cannot name an unrunnable action |
| Persisted mission trace | **PASS** — append-only, restart-safe, redacted |
| Cross-process pause/resume | **PASS** — pause holds mid-flight; resume continues the same run |
