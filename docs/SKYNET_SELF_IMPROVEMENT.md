# SKYNET Self-Improvement Engine (Phase P7)

The improvement loop: **detect weakness → hypothesize → propose → test →
evaluate → approve → apply/version → monitor → rollback**, with every step
measured, traced, and reversible. This is the machinery the future
autonomous learning system (P8+) will drive; today every state-changing
step is operator-gated.

Status: **implemented and tested** (47 dedicated tests). Not implemented:
autonomous proposal generation, code-level improvements, automatic
application without approval — see
[Current capabilities vs future](#current-capabilities-vs-future).

---

## Architecture

```
recorded experiences/experiments
        │
        ▼
 WeaknessDetector (agency/improve/detector.py)     WEAKNESS_DETECTED
        │  evidence-backed weaknesses only
        ▼
 build_proposal_artifacts (pipeline.py)             HYPOTHESIS_CREATED
        │  hypothesis + proposal + candidate        IMPROVEMENT_PROPOSED
        │  (config-only data artifacts)             CANDIDATE_CREATED
        ▼
 ImprovementPipeline.test → P6 ExperimentRunner     CANDIDATE_TEST_STARTED
        │  (the ONE executor; sandbox;              BENCHMARK_STARTED/COMPLETED
        │   frozen criteria; P6 trace events)
        ▼
 ImprovementPipeline.evaluate (AcceptanceGate)      ACCEPTANCE_EVALUATED
        │  ACCEPT / REJECT / INCONCLUSIVE           IMPROVEMENT_ACCEPTED/REJECTED
        │  = P6 Verdict.SUCCESS only
        ▼
 operator approval (CLI)                            HUMAN_APPROVAL_REQUIRED
        ▼
 ImprovementPipeline.apply → StrategyVersion        IMPROVEMENT_APPLIED
        │  (ledger append; registry keeps
        │   running parent procedure + new config)
        ▼
 monitor (expected vs actual)                       IMPROVEMENT_MONITORED
        │  regression events; optional rollback     IMPROVEMENT_ROLLED_BACK
        ▼
 memory + history stores (provenance-complete)
```

**No second execution engine.** Benchmarks are ordinary P6 experiments:
`ImprovementPipeline.test` builds an `Experiment` and hands it to the
existing `ExperimentRunner`, which enforces the sandbox, per-trial
timeout, budget, frozen-before-run criteria, and emits the full
`EXPERIMENT_*` trace stream.

## Weakness detection

`WeaknessDetector` runs deterministic rules over recorded data —
`ExperienceRecord` streams and finished `Experiment` records. Every rule
is a configurable threshold (`DetectorThresholds`, wired from
`SKYNET_IMPROVEMENT_*` settings); every weakness carries verbatim
evidence items (record ids, counts, measured values). Rules:

| Rule | Category | Threshold |
|---|---|---|
| repeated failures grouped by signature | `task_failure` | `min_task_failures` |
| evaluations below a score bar | `low_evaluation` | `low_score`, `min_low_scores` |
| runs exceeding a step budget | `excessive_tool_calls` | `excessive_steps` |
| experiments with many failed trials | `planning_failure` | `experiment_failure_trials` |
| duplicate-rate above bar in any arm | `duplicate_sources` | `duplicate_rate` |

Severity is a deterministic function of frequency (log2 growth between a
floor and a cap) — same data, same severity, always. A weakness needs
`min_frequency` supporting records; **the detector cannot fabricate one**
(an empty or healthy dataset yields zero weaknesses, verified by test).

## Hypotheses and proposals

- `ImprovementHypothesis` — the testable claim, built deterministically
  from the weakness; carries the P6 `SuccessCriterion` set, evidence,
  provenance. A `confidence` field exists but is recorded, never treated
  as evidence.
- `ImprovementProposal` — the full audit record: weakness → hypothesis →
  target → proposed change (data) → experiment plan (metrics, trials,
  seed, budgets, `procedure_params`) → acceptance criteria → rollback
  plan. Statuses: `proposed → approved_for_test → testing → accepted |
  rejected | rolled_back`. `origin` is `internal` or `external`; external
  suggestions (web pages, other AIs) get **no elevated trust** — they
  enter through `improve_propose` / `record_weakness` and face the same
  benchmark + approval gates as internal detections.
- `CandidateStrategy` — **config, not code**: the parent strategy's
  registered procedure plus a new frozen config dict. The candidate
  version is derived (`cand-<weakness[:8]>[-N]`) so repeat attempts never
  collide with already-registered versions.

## Testing: the P6 experiment engine

`pipeline.test()` registers the candidate (parent procedure + candidate
config) in the `StrategyRegistry` and runs one experiment:

- both arms resolve from the registry (unregistered → error before any
  trial);
- the sandbox `TrialContext` sees only injected services + params +
  scratch dir (no env, no settings, no DB, no memory);
- criteria freeze before the first trial; trials, seed, timeouts come
  from the plan;
- the P6 runner emits `TRIAL_*`, `METRIC_RECORDED`,
  `BASELINE_MEASURED`, `CANDIDATE_MEASURED`, `EXPERIMENT_COMPARED`,
  `EXPERIMENT_EVALUATED` — the improvement layer adds no new metric or
  trace plumbing.

## Acceptance gate

`pipeline.evaluate()` is mechanical: the decision is `accepted` **iff**
the P6 evaluator returned `Verdict.SUCCESS` — all frozen criteria passed
on real measured data with enough trials. Everything else is `rejected`
with the evaluator's explanation as the reason. There is no path where
an LLM, a proposal author, or an external suggestion flips a verdict.
`HUMAN_APPROVAL_REQUIRED` is emitted on acceptance; the operator reads
the evidence (proposal → experiment record) before approving.

## Approval and application

`pipeline.approve(proposal, approved_by=…, note=…)` records explicit
human approval on an ACCEPTED proposal. `pipeline.apply(...)` refuses to
run without that record (`PermissionError`) and refuses anything not
ACCEPTED. Application:

1. appends a `StrategyVersion` to the ledger (id, parent version,
   proposal + experiment ids, measured benchmark results, acceptance
   record incl. who approved, full config, provenance);
2. moves the `ActiveStrategy` pointer (previous ref recorded);
3. never deletes or overwrites anything — the ledger is append-only.

`seed_baseline` lets an operator register the currently-active version
once, so rollback always has a known-good target.

## Rollback

`pipeline.rollback(proposal, reason=…)` restores the previous known-good
version: the active pointer moves back, the proposal transitions to
`rolled_back`, `IMPROVEMENT_ROLLED_BACK` is traced, and the failed
candidate version **stays in the ledger** with all its evidence. Without
a previous version, rollback refuses rather than guess.

## Post-deployment monitoring

`pipeline.monitor(proposal, actual=…, rollback_below=…)` compares
measured production values against configured floors. Each violation
creates a persisted `RegressionEvent` and an `IMPROVEMENT_MONITORED`
trace. With `monitor_policy='auto_rollback'` in the proposal metadata
(explicit opt-in; default off — see `SKYNET_IMPROVEMENT_AUTO_ROLLBACK`)
the pipeline rolls back automatically; otherwise the event is recorded
for the operator.

## Memory integration

`record_proposal` (improve/memory_bridge.py) writes decisions into the
P2 memory system with mandatory provenance
(weakness → hypothesis → proposal → candidate → experiment → verdict):

- any decision → **episodic** memory (importance: accepted 0.9, rejected
  0.65, proposed 0.4);
- rejected proposals additionally → **procedural** memory ("this
  configuration failed to beat the baseline; do not re-propose without
  new evidence").

Origin `improvement` was added to the memory system's closed origin set
(alongside `experiment` — P6 records had been normalizing to `system`;
that latent gap is fixed in this phase). Memory failures never break the
pipeline: recording is best-effort and logged.

## History and "have I tried this before?"

`ImprovementStore` (JSONL, `data/improvements.jsonl` by default, atomic
appends) persists weaknesses, proposals, candidates, versions, active
pointers, and regressions. It answers across restarts:

- *have I attempted this improvement before?* — `proposals()` /
  `prior_proposals_for`; new proposals record `prior_attempts` in
  metadata;
- *what happened the last time?* — proposal status + decision reason +
  experiment id (full metrics in the experiment registry);
- *which version is active / previous known-good?* — `active(name)` /
  `versions(name)`.

## Security boundaries

Enforced structurally and tested:

- **No source-code changes, ever.** Improvements are registry
  config-artifacts; the candidate reuses the parent's registered
  procedure object (asserted by test).
- **Approval gate.** `apply` without `approve` raises `PermissionError`
  (tested, including for `origin="external"` proposals).
- **No privileged actions.** Improvement actions ship under category
  `lab` behind the dark `SKYNET_ENABLE_SELF_IMPROVEMENT` flag plus the
  action allow-list — two independent gates. There is no
  `improve_apply` action at all.
- **Sandbox.** Candidates run inside the P6 `TrialContext` boundary.
- **External content is data.** Suggestions from web/AI sources become
  ordinary weaknesses; directive-like text in them has no execution
  path (same posture as the P5 comms security layer).
- **No secrets.** Configuration keys are thresholds and paths only.

## CLI

```
skynet improve detect                # detect weaknesses from recorded history
skynet improve propose <id>          # hypothesis + proposal + candidate
                                     #   --origin external for external suggestions
                                     #   --param key=value candidate config
skynet improve history               # prior attempts + decisions
skynet improve demo                  # full offline end-to-end demonstration
```

Testing/applying/rolling back are exposed via the pipeline API and
`improve_propose` / `improve_inspect` / `improve_history` actions; the
CLI demo walks the whole lifecycle including approval, rollback, and the
rejection path.

## The offline demonstration

`python -m agency.cli improve demo [--memory-backend sqlite]` runs, with
real measurements on the injected corpus (no network, no APIs):

1. a real evidence experiment measures a wasteful 4-variant strategy
   (duplicate_rate ≈ 0.62);
2. the detector raises `duplicate_sources` from the persisted record;
3. a config-only candidate (cap per-query results, trim fan-out) is
   proposed through the standard pipeline;
4. the P6 runner benchmarks it (5 trials/arm, seed 42);
5. the gate accepts it **only if** the measured criteria pass —
   duplicates drop ~65% with coverage held;
6. approval is recorded, the version is applied;
7. monitoring measures a coverage collapse → `auto_rollback` restores
   the known-good `v1` (candidate preserved in the ledger);
8. a deliberately bad candidate is honestly REJECTED and stored to
   memory;
9. every decision lands in memory with full provenance.

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `SKYNET_IMPROVEMENTS_REGISTRY_PATH` | `data/improvements.jsonl` | history file |
| `SKYNET_IMPROVEMENT_MIN_FREQUENCY` | `2` | recurrence bar for weaknesses |
| `SKYNET_IMPROVEMENT_LOW_SCORE_THRESHOLD` | `0.4` | low-evaluation rule |
| `SKYNET_IMPROVEMENT_DUPLICATE_RATE_THRESHOLD` | `0.5` | duplicate-sources rule |
| `SKYNET_IMPROVEMENT_MAX_ACCEPTANCE_TRIALS` | `3` | trials per arm for improvement experiments |
| `SKYNET_IMPROVEMENT_AUTO_ROLLBACK` | `false` | monitor-triggered rollback policy |
| `SKYNET_ENABLE_SELF_IMPROVEMENT` | `false` | dark flag gating all `lab` actions |

## Current capabilities vs future

**Now (P7):** detection from recorded data; deterministic hypothesis +
proposal + candidate generation; benchmarking through the P6 engine;
mechanical acceptance; human-gated application; append-only versioning;
restorable rollback; monitoring with optional auto-rollback; complete
memory + history provenance.

**Not yet (future phases):**

- **Autonomous proposal generation** — P8's loop proposes on its own;
  today proposals are explicit (operator, actions, or demo).
- **Code-level improvements** — the `ImprovementProposal` schema and
  pipeline stages are the seam for future *code* artifacts (proposal →
  isolated sandbox → build → test → benchmark → gate → versioned
  change); none of that executes today.
- **Persistent version replay** — "active" is a pointer + config record;
  wiring the active config into the production loop's runtime strategy
  selection is future work (strategies already carry frozen configs).
- **Statistical significance** — the comparator remains descriptive;
  verdicts rest on frozen absolute/relative criteria, not p-values.
