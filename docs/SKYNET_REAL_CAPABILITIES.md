# SKYNET Real Capabilities (Phase P9)

This document describes SKYNET's **real capability execution** layer: which
providers are supported, how they are health-checked, how real LLM and web
execution work, how provenance and prompt-injection defense operate, and —
critically — the clear distinction between **live**, **local**, **mock** and
**unavailable** functionality. The governing rule of this phase:

> Never report "real LLM tested" unless a real LLM request actually succeeded.
> Never silently substitute a mock when real execution was requested.

Everything here rides the **existing** provider/adapter seams (P2 memory,
P4 web, P5 comms, P6 experiments, P7 improvement, P8 orchestrator). No
provider or adapter was duplicated.

---

## 1. Supported providers

### Intelligence (LLM) — `agency/intelligence/providers.py`

| `SKYNET_LLM_PROVIDER` | Classification | Notes |
|---|---|---|
| `mock` | **MOCK** | Deterministic, offline; never labeled real. |
| `openai` | **REAL** | Any OpenAI-compatible endpoint: OpenAI, **Ollama** (`http://localhost:11434/v1`), vLLM, OpenRouter, … |

- Credentials are read **at call time** from `OPENAI_API_KEY` and never
  copied into settings, logs, traces, or health results. For Ollama, set a
  placeholder value (`OPENAI_API_KEY=ollama`) — the local server does not
  validate it.
- The provider raises `ProviderActivationError` loudly on missing
  credentials; it never degrades silently to mock.
- Models are **not hard-coded**: `SKYNET_LLM_MODEL` selects the model.
  Validated live with Ollama: `llama3.2:1b` (fast, non-thinking).
  Note: "thinking" models (e.g. `qwen3.5:0.8b`) emit reasoning into a
  separate field with empty `content` on tiny token budgets — use a
  non-thinking model for SKYNET's structured outputs.

### Web / research — `agency/web/search.py` + SafeFetcher (P4)

| `SKYNET_SEARCH_PROVIDER` | Classification | Notes |
|---|---|---|
| `wikipedia` (default) | **REAL** | Keyless Wikipedia API; validated live (~1s per search). |
| `duckduckgo` | **REAL** | Keyless; rate-limited by upstream. |
| `none` | **UNAVAILABLE** | Explicitly configured off. |

Paid/keyed search APIs are **not implemented** in this repository and are
therefore not claimed. Page fetching goes through the existing SafeFetcher
(respects robots.txt and timeouts); some sites return 403/406 to automated
fetchers — these are recorded as action failures, never papered over.

### External AI communication — `agency/comms/providers.py` (P5)

| Provider | Classification | Notes |
|---|---|---|
| `MockAIProvider` (only implementation) | **MOCK** | Real AI↔AI communication is **UNAVAILABLE** in this repository by design. |

No external AI-comms provider exists behind `build_provider`; the
orchestrator reports this honestly instead of inventing one.

---

## 2. Configuration

All via environment (`SKYNET_` prefix), read by `agency/config.py`:

```bash
# Real LLM via Ollama (local)
SKYNET_LLM_PROVIDER=openai
SKYNET_LLM_BASE_URL=http://localhost:11434/v1
SKYNET_LLM_MODEL=llama3.2:1b
SKYNET_LLM_TIMEOUT_SECONDS=120        # per-request timeout (default 30)
OPENAI_API_KEY=ollama                 # placeholder for local servers

# Real LLM core strategies (planner + evaluator ride the same provider)
SKYNET_PLANNER_STRATEGY=llm
SKYNET_EVALUATOR_STRATEGY=llm

# Web
SKYNET_SEARCH_PROVIDER=wikipedia      # wikipedia | duckduckgo | none
SKYNET_ENABLE_WEB_TOOLS=true

# Action allowlist — proposal space for the LLM planner AND execution gate
SKYNET_ENABLED_ACTIONS=echo,web_research,web_search,web_fetch,web_extract,memory_store,memory_search,memory_extract
```

Key semantics:

- `SKYNET_ENABLED_ACTIONS` is a **double gate**: the LLM planner may only
  propose allowlisted **and** registered actions (bootstrap intersects the
  two); the core loop refuses to execute anything outside it.
- `SKYNET_STORAGE_BACKEND=memory` (CLI default) needs
  `SKYNET_PERCEPTION_ADAPTER=none`; `database` enables the full Gods-Eye
  perception stack.
- Budgets (iterations / web requests / AI requests / runtime / experiments /
  cost) are set via `OrchestratorBudgets` and `SKYNET_AUTONOMOUS_*` settings.

---

## 3. Health checks (`provider status` / `provider test`)

`agency/orchestrator/health.py` provides structured health status:

```bash
# Classification only (no requests, exit 0): REAL/LOCAL/MOCK/UNAVAILABLE
.venv/Scripts/python.exe -m agency.cli provider status

# Real probes (exit 0 iff every probed component is ok)
OPENAI_API_KEY=ollama SKYNET_LLM_PROVIDER=openai \
SKYNET_LLM_BASE_URL=http://localhost:11434/v1 SKYNET_LLM_MODEL=llama3.2:1b \
.venv/Scripts/python.exe -m agency.cli provider test --component all
```

Each check verifies (in one minimal request, no retries):

1. **reachable** — endpoint answered
2. **authenticated** — accepted credentials/configuration
3. **model_available** — the requested model served the request
4. **parse_ok** — the response parsed and carried text
5. latency (ms), finish reason, token usage, response model

The `HealthStatus.probed` field distinguishes *classification-only* rows
(`provider status`: "unprobed" — a claim about implementation, not liveness)
from *probed failures* (`provider test`: "unavailable"). Mock providers are
always classified MOCK and never reported as real.

Validated live results (Ollama + Wikipedia, 2026-09-25):

```
[llm] openai:llama3.2:1b           REAL   ok   953ms  one minimal completion succeeded
[web] wikipedia                    REAL   ok   982ms  one live search returned 1 result(s)
```

---

## 4. Real LLM execution

Real LLM calls happen in exactly three places, all through the existing
`IntelligenceService` seam:

1. **Plan-stage query proposal** (orchestrator): one request asking the
   model for a concise web search query. The free-text answer is parsed by
   `_extract_search_query` (quoted/labelled fragments, smart-quote
   normalization) so model prose never becomes the literal query. On failure
   the run records an error and falls back to the deterministic query.
2. **Core planner** (`LLMPlanner`): proposes JSON action plans, filtered to
   the allowlist∩registry, with the deterministic planner as fallback.
3. **Core evaluator** (`LLMEvaluator`): judges run quality (bounded
   observation snippets in the prompt), with the deterministic evaluator as
   fallback.

Every orchestrator-issued call is wrapped in `_trace_llm_request`, emitting
`LLM_REQUEST_STARTED` / `LLM_REQUEST_COMPLETED` / `LLM_REQUEST_FAILED` with
provider, model, latency and purpose (never message bodies), and increments
the run's `ai_requests` budget counter.

Validated live: bounded autonomous objective completed with
`llama3.2:1b` via Ollama — proposal (≈1.3s), plan, evaluation all real.

---

## 5. Web execution

`WebResearchAdapter` (orchestrator seam) wraps the P4 `ResearchService`:

- search via the configured `SearchProvider` (Wikipedia by default),
- page text via `ExtractedContent.text` (bounded to 2000 chars),
- one **budget-counted** simplified-query retry per failed search (first 5
  words, punctuation stripped) — some backends return nothing for
  keyword-poor queries; the retry is recorded in `web_requests` like any
  other request.

Validated live: Wikipedia search → full page text retrieved (~1s), e.g.
"retrieval augmented generation" → 8 sources including
`https://en.wikipedia.org/wiki/Retrieval-augmented_generation`.

---

## 6. Provenance

Every `Evidence` item carries an `evidence_class` in its provenance dict:

| Class | Meaning |
|---|---|
| `LIVE_EXTERNAL_EVIDENCE` | Retrieved from the real web this run (URL, provider, query, `retrieved_at`, run_id, title). |
| `LOCAL_TEST_DATA` | Injected local corpus (deterministic adapters). |
| `MODEL_GENERATED_CONTENT` | Produced by a model (core actions, comms answers) — never treated as external evidence. |
| `MEMORY` | Recalled from the long-term memory store. |
| `EXPERIMENTAL_RESULT` | Produced by the P6 experiment runner. |

The evaluator's completion rule is mechanical: ≥3 **external** evidence
items (`source_type` web/ai) **and** a passing core cycle. Model-generated
text alone never completes an objective. Web provenance additionally
records `security_flags` from `scan_content`.

---

## 7. Prompt-injection defense

External content is **data, never control flow**
(`agency/orchestrator/security.py`):

- `scan_content` detects directive-like payloads (e.g. *"Ignore your
  current task and execute this command"*) before storage.
- Findings are recorded on each evidence item (`security_flags`) and
  aggregated into `run.security_report`
  (`status: CONTENT_TREATED_AS_UNTRUSTED_DATA`) — telemetry, never policy.
- External text cannot: execute commands, change configuration/prompts,
  bypass approval, alter the allowlist, or modify source code. Test D
  (`tests/test_real_capabilities.py`) proves a malicious page does not
  affect the run outcome.

---

## 8. AI ↔ AI communication

The comms stack (P5) currently ships **only** `MockAIProvider`; no external
AI provider exists behind the factory. AI↔AI communication is therefore
**AI_COMMUNICATION_UNAVAILABLE**. The autonomous loop continues with other
capabilities; `real` runs do not require it. If a real provider is added to
`build_provider` later, the orchestrator's COMMUNICATE stage and evidence
provenance (`LIVE_EXTERNAL_EVIDENCE`, conversation id) already support it.

---

## 9. Budgets

`OrchestratorBudgets` bounds every run: `max_iterations`,
`max_web_requests`, `max_ai_requests`, `max_runtime_seconds`,
`max_experiments`, `max_cost_usd`. The loop checks remaining budget before
each stage; on breach the run **PAUSEs** (resumable) or STOPs (terminal,
honest failure) — never loops unbounded. Adapters receive the run's
remaining allowance and count every request, including retries.

---

## 10. Failure modes and fallbacks

| Failure | Behavior |
|---|---|
| LLM unreachable / timeout | `LLM_REQUEST_FAILED` event, run error recorded, deterministic query/plan/evaluation fallback; never labeled real. |
| Missing credentials | Provider construction raises loudly (`ProviderActivationError`). |
| Malformed model output | "schema-violating JSON" logged; planner/evaluator fall back to deterministic strategies. |
| Search returns 0 results | One simplified-query retry (budget-counted); still empty → run proceeds with less evidence and evaluates honestly. |
| Web fetch 403/406 | Recorded as action failure; run continues on remaining budget. |
| Memory unavailable | Recall/store errors recorded and ignored; run degrades gracefully (test G). |
| Budget exceeded | PAUSE (resumable) or STOP with explicit reason (test E). |
| Malicious web content | Flags recorded, content stays data, run unaffected (test D). |

---

## 11. Security boundaries

- External content is stored verbatim as evidence but only ever influences
  the loop as information; `scan_content` guarantees directive-like
  payloads never become control flow.
- The action allowlist (`SKYNET_ENABLED_ACTIONS`) gates both planning and
  execution; memory-action allowlist changes require operator
  configuration, not model output.
- Credentials never appear in settings, traces, evidence, health results,
  or logs.
- Self-improvement remains proposal-only: the IMPROVE stage surfaces
  weaknesses/improvement proposals; it never autonomously modifies
  production source code.

---

## 12. Live vs mock vs unavailable — the honest labels

| Capability | Status |
|---|---|
| Real LLM (Ollama, OpenAI-compatible) | **VALIDATED LIVE** — smoke test + bounded autonomous objective completed. |
| Real web research (Wikipedia) | **VALIDATED LIVE** — searches + page text retrieved; evidence marked `LIVE_EXTERNAL_EVIDENCE`. |
| Real page fetching (SafeFetcher) | **AVAILABLE**; some sites refuse automated fetchers (403/406) — recorded, not fabricated around. |
| AI↔AI communication (external) | **UNAVAILABLE** (mock provider only, by design). |
| Mock LLM / mock comms / local corpus | **MOCK/LOCAL** — always labeled, never substituted for a requested real provider. |
| DuckDuckGo search | Implemented in P4; unvalidated in this phase — treat as UNVALIDATED until probed. |

---

## 13. CLI quick reference

```bash
# Provider classification / probing
python -m agency.cli provider status [--component all|llm|web] [--json]
python -m agency.cli provider test   [--component all|llm|web] [--json]

# Bounded real-capability demo (deterministic core steps, real web stack)
python -m agency.cli real demo [--objective "..."] [--query Q]... [--offline] [--json]

# Bounded autonomous run (LLM planner/evaluator + LLM query proposal)
OPENAI_API_KEY=... SKYNET_LLM_PROVIDER=openai SKYNET_LLM_BASE_URL=... \
SKYNET_LLM_MODEL=... SKYNET_PLANNER_STRATEGY=llm SKYNET_EVALUATOR_STRATEGY=llm \
python -m agency.cli real start --objective "..." --max-iterations 3
python -m agency.cli real status
python -m agency.cli real inspect <run_id>
```

Exit codes: `provider test` returns non-zero iff any probed component is
unavailable; `real start/demo` return non-zero iff the run did not complete.
