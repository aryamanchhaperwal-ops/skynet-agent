# SKYNET Experimentation & Self-Evaluation Engine

Phase P6. This document describes the scientific machinery the future
self-improvement engine will use: controlled experiments, frozen acceptance
criteria, baselines, repeated trials, deterministic evaluation, persistence,
and the security boundary that keeps experiments from becoming a code-
modification channel.

The self-improvement engine itself is **not implemented yet** — this phase
provides the measurement instrument only.

---

## 1. What an experiment is

An experiment compares a **candidate strategy** against a **baseline
strategy** over the same procedure, for N trials each, and records:

- the **hypothesis** (a testable claim with rationale and expected effect),
- **frozen acceptance criteria** (written before the first trial),
- per-trial **metrics** (declared via `MetricDefinition` with a direction),
- a **comparison report** (descriptive deltas, improvements, regressions),
- an **evaluation verdict** (`success` / `failure` / `inconclusive` /
  `error` / `cancelled`) with evidence.

```
Experiment
├── Hypothesis (statement, rationale, expected effect, provenance)
├── baseline: TargetRef   e.g. research_strategy:v1  ── registered artifact
├── candidate: TargetRef  e.g. research_strategy:v2  ── registered artifact
├── metrics: [MetricDefinition]      (name + direction, custom ones welcome)
├── success_criteria: [SuccessCriterion]   FROZEN before the first trial
├── trials_results: [TrialResult]    per-trial metrics + provenance
├── comparison: ComparisonReport     descriptive deltas, no fake significance
└── evaluation: ExperimentEvaluation verdict + evidence + criteria outcomes
```

## 2. Strategies are registered artifacts — never code references

A **strategy** is a named, versioned artifact in a
:class:`agency.experiments.registry.StrategyRegistry`:

```python
registry.register(
    name="multi_query", version="v1",
    capability="research_strategy",        # closed vocabulary
    procedure=my_async_callable,           # receives a sandboxed TrialContext
    config={"max_queries": 4},             # frozen, data-only
)
```

Experiments reference `name:version` pairs (`TargetRef`). The registry is
the structural safety property:

- an experiment **cannot name code** — only registered artifacts;
- capabilities are a closed set (`planner_policy`, `research_strategy`,
  `tool_config`, `retrieval_params`, `benchmark_def`,
  `evaluation_policy`);
- unregistered targets are **refused before any trial runs**;
- duplicate registrations are rejected (explicit `unregister` required).

Registration is an explicit, auditable act by an operator (or, later, by
the improvement engine through the same public API).

## 3. Sandbox and trial context

Every trial procedure receives a `TrialContext` — the only object it can
act through:

| It CAN | It CANNOT |
|---|---|
| read its `params` (procedure config) | access environment variables or secrets |
| read `strategy.config` (frozen) | touch the database, settings, or memory |
| use explicitly **injected** `services` | import/reach the agent's own modules |
| write to its per-experiment `scratch_dir` | write anywhere else on disk |
| know its `arm`, `trial_index`, `seed` | spawn processes, open raw sockets |

The runner enforces a per-trial timeout (`asyncio.wait_for`) and a total
wall-clock budget per experiment; exhaustion produces `CANCELLED`, never a
fabricated result. Heavier isolation (subprocess/container) can replace
`Sandbox` later without touching the runner.

## 4. Frozen-before-run acceptance criteria

Criteria are copied from the hypothesis into the experiment record **before
the first trial** (`experiment.criteria_frozen()`); the verdict is computed
exclusively against that frozen list. Available kinds:

| kind | meaning |
|---|---|
| `improves` | relative improvement (candidate−baseline)/|baseline| ≥ threshold |
| `not_regresses` | relative regression not worse than −threshold |
| `max` | candidate mean ≤ threshold (absolute bounds; zero baselines) |
| `min` | candidate mean ≥ threshold |

An experiment **without** criteria can only ever be `inconclusive`.

## 5. Verdict semantics

The deterministic evaluator (metrics decide; uncertainty stays uncertainty):

- `success` — every criterion passed (and ≥ 3 trials per arm; fewer
  downgrades to `inconclusive`: a single lucky run is not evidence).
- `failure` — at least one criterion failed, none passed.
- `inconclusive` — mixed evidence, unmeasurable data (zero baseline,
  missing metrics), or no criteria.
- `error` — every trial failed, or the experiment never ran its trials.
- `cancelled` — budget exhausted or externally cancelled.

Additionally, a detected regression on a metric **not covered** by any
criterion downgrades `success` to `inconclusive` (the criteria didn't
price it in). Regressions on a covered metric (e.g. an absolute `max`
criterion) do not.

**No statistical significance is ever claimed.** The comparison report
carries `significance_tested: False` and descriptive deltas only; an
actual test (t-test, bootstrap) is a future phase.

## 6. Trials and reproducibility

- `trials` per arm (default 3, demo uses 5).
- `seed` — deterministic seeds per trial: baseline gets
  `seed + index`, candidate gets `seed + 10_000 + index`, so both arms
  share conditions while staying reproducible.
- `per_trial_timeout_seconds` and `max_total_seconds` bound every run.
- The finished record keeps an **environment snapshot** (python version,
  platform, timestamp) plus strategy `config`, procedure params, criteria,
  and both target refs — enough to re-run the comparison.

## 7. Runner lifecycle and tracing

```
EXPERIMENT_CREATED → EXPERIMENT_STARTED
    → (TRIAL_STARTED → TRIAL_COMPLETED|TRIAL_FAILED → METRIC_RECORDED)*
    → BASELINE_MEASURED / CANDIDATE_MEASURED
    → EXPERIMENT_COMPARED → EXPERIMENT_EVALUATED
    → EXPERIMENT_COMPLETED | EXPERIMENT_FAILED | EXPERIMENT_CANCELLED
```

All events flow through the run's existing `Trace` facade (same stream,
same `run_id`, same seq ordering — no second logging system).

## 8. Persistence and "have I tried this before?"

Finished experiments are persisted to `SKYNET_EXPERIMENTS_REGISTRY_PATH`
(JSONL, atomic temp-file writes) by `ExperimentRegistry`:

- `list()` — summaries for `experiment list`
- `get(id)` — full record for `experiment inspect`
- `search(query)` — keyword search over name/description/objective/hypothesis
- `prior_for(baseline, candidate)` — exact prior pair matches

## 9. Memory integration

`record_experiment()` (memory bridge) writes, with mandatory provenance
(`experiment_id`, hypothesis, both refs, goal/run linkage):

- an **episodic memory** for every finished experiment (importance 0.45–0.8
  by verdict — failures are valuable),
- a **procedural memory** on `success` ("prefer candidate over baseline
  under these conditions"), phrased as measured evidence, never truth.

Recall it: `.venv/Scripts/python.exe -m agency.cli recall "multi_query"`.

## 10. Actions and CLI

Actions (category `lab`, gated by the dark `SKYNET_ENABLE_SELF_IMPROVEMENT`):

- `experiment_run` — create + execute one experiment to a verdict
- `experiment_list` — prior history (query before proposing)
- `experiment_inspect` — full record

CLI:

```bash
# Built-in offline demonstration (deterministic, no network, no APIs):
.venv/Scripts/python.exe -m agency.cli experiment demo
#   → single_query:v1 vs multi_query:v1 over an injected corpus,
#     5 trials each, seed 42, real measured verdict.

# Custom experiment:
.venv/Scripts/python.exe -m agency.cli experiment run \
    --name my-experiment \
    --hypothesis "Strategy B covers more sources than A" \
    --baseline single_query:v1 --candidate multi_query:v1 \
    --criterion '{"kind": "improves", "metric": "sources_found", "threshold": 0.1}' \
    --metric sources_found:maximize --trials 3 --seed 7 \
    --param "goal=agent memory architectures" --memory-backend sqlite

.venv/Scripts/python.exe -m agency.cli experiment list
.venv/Scripts/python.exe -m agency.cli experiment inspect <id>
```

## 11. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SKYNET_EXPERIMENTS_REGISTRY_PATH` | `data/experiments.jsonl` | experiment history file |
| `SKYNET_EXPERIMENT_DEFAULT_TRIALS` | `3` | trials per arm |
| `SKYNET_EXPERIMENT_PER_TRIAL_TIMEOUT_SECONDS` | `60` | per-trial wall clock |
| `SKYNET_EXPERIMENT_MAX_TOTAL_SECONDS` | `600` | total experiment budget |
| `SKYNET_ENABLE_SELF_IMPROVEMENT` | `false` | registers the `lab` actions (dark) |

## 12. Security boundary (summary)

- Experiments operate on **registered strategies + data** — never source
  code. No action in this phase can read, modify, or execute code files.
- The sandbox exposes no secrets, no DB sessions, no settings, no network
  credentials; services must be explicitly injected per experiment.
- Budgets: per-trial timeout + total experiment budget; exhaustion is
  recorded, never hidden.
- The **LLM advisor can only comment** (`LLMExperimentAdvisor`); it cannot
  alter metrics, criteria, or any verdict.
- Actions ride the existing category gate (`lab` → `enable_self_improvement`,
  default off) plus the loop's action allow-list — two independent gates.

## 13. Extension points

- **Statistical significance** — replace the descriptive comparison with a
  real test when trial counts justify it; `significance_tested` is the flag.
- **Heavier sandboxes** — subprocess/container implementations of `Sandbox`.
- **Improvement engine (P7)** — proposes hypotheses from weaknesses,
  registers candidate strategies, calls `experiment_run`, reads verdicts,
  and (on success) promotes strategy versions — always through this public
  API, never by editing the runner.
- **Postgres persistence** — swap `ExperimentRegistry`'s backend for the
  blueprint's `experiments` table; the API is shaped to survive it.
