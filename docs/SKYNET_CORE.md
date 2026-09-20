# SKYNET Core — Implementation Guide

**Phase:** Core Foundation (P0.5) · **Status:** implemented and tested
Companion to `docs/SKYNET_BLUEPRINT.md` (architecture) and
`docs/BASELINE_VALIDATION_REPORT.md` (protected God's Eye baseline).

---

## 1. What this phase delivers

A modular, tested **Skynet core** (`agency/`) that receives a goal and executes
a controlled agent cycle — running on top of the untouched God's Eye backend
(`main.py`, `connector_usgs.py`, the `events` table). It deliberately contains
**no LLM calls, no web access, no self-modification**: those arrive in later
phases behind the interfaces defined here.

```
GOAL → OBSERVE → PLAN → SELECT ACTION → EXECUTE → RECORD EXPERIENCE
     → EVALUATE → UPDATE STATE → COMPLETE / CONTINUE
```

## 2. Module responsibilities

| Module | Responsibility |
|---|---|
| `agency/config.py` | `SkynetSettings` (pydantic-settings, `SKYNET_` prefix): storage backend, provider, perception source, action allow-list, run budgets, **dark capability flags** (`SKYNET_ENABLE_WEB_TOOLS/EXTERNAL_COMMS/SELF_IMPROVEMENT` — all default false) |
| `agency/observation.py` | `Observation` — the perception boundary currency (source, kind, content, confidence, timestamps, goal/run linkage). Content is `Any`: text, JSON, GeoJSON, tool output, error |
| `agency/state.py` | `AgentState` — serializable runtime state: run/goal ids, step cursor, observations, action records, errors, context, timestamps |
| `agency/goals.py` | `Goal`/`GoalSpec`/`GoalManager` — creation, priority, metadata, and the **status state machine** (`pending → running → paused/completed/failed/cancelled`) with illegal-transition rejection |
| `agency/planner.py` | `Planner` ABC (`plan(goal, state) -> Plan`) + `DeterministicPlanner` placeholder (honors goal-specified `params["steps"]`, else a default action) |
| `agency/actions/base.py` | `ActionSpec`/`ActionResult`/`ActionRecord`/`ActionContext`, the `Action` ABC, `ActionRegistry`, and `run_action()` — the single choke point that captures timing + every exception (a raising action can never crash a run) |
| `agency/actions/builtins.py` | `EchoAction` (deterministic demo) and `GodsEyeLatestEventsAction` (read-only fetch from the existing `events` table) |
| `agency/perception.py` | `PerceptionAdapter` ABC + `GodsEyePerceptionAdapter` (read-only SELECT on `events`; degrades to an `error` observation if the DB is down) + `NullPerceptionAdapter` |
| `agency/evaluator.py` | `Evaluator` ABC (`evaluate_action`, `evaluate_run` → `Evaluation`) + `DeterministicEvaluator` |
| `agency/experience.py` | `ExperienceRecord`, `ExperienceSink` ABC, in-memory + PostgreSQL sinks, buffering `ExperienceRecorder` (flush after each stage) |
| `agency/trace.py` | `TraceEventType` (RUN_STARTED … RUN_FAILED), `TraceEvent`, `TraceSink` ABC, logging/memory/database sinks, `Trace` facade (per-run `seq` ordering, broken-sink isolation) |
| `agency/storage.py` | `Storage` ABC (goals + runs) with `MemoryStorage` and `DatabaseStorage`; `ensure_core_schema()` creates **only** the four new tables (goals, runs, run_events, experiences) — never the God's Eye schema |
| `agency/loop.py` | `SkynetCore.run_goal()` — the cycle itself, with step/time budgets, graceful failure handling, and `RunSummary` |
| `agency/bootstrap.py` | `build_core()` — composition root; wires defaults and gates action categories behind feature flags |
| `agency/cli.py` | `python -m agency.cli run --goal "..."` (and the `skynet` console script after `pip install -e .`) |

Dependency direction (unchanged from the blueprint):
`database ← agency` — one-way. God's Eye code never imports `agency`.

## 3. Execution lifecycle of one run

1. **GOAL** — `GoalManager.create()` persists a `pending` goal, marks it
   `running`; `RUN_STARTED` / `GOAL_CREATED` traced; a `runs` row is created.
2. **OBSERVE** — the configured `PerceptionAdapter` returns observations
   (or an error observation when a sensor is down — the run survives);
   each is traced (`OBSERVATION_RECEIVED`) and recorded as an experience.
3. **PLAN** — `Planner.plan(goal, state)`; traced (`PLAN_CREATED`) and the
   serialized steps are persisted on the run row.
4. **SELECT ACTION → EXECUTE** — per step, within budgets
   (`SKYNET_MAX_STEPS_PER_RUN`, `SKYNET_MAX_RUN_SECONDS`): the registry is
   consulted, the configuration allow-list is enforced, `run_action()`
   executes with exception capture. `ACTION_SELECTED/STARTED/COMPLETED/FAILED`
   traced; action + evaluation experiences recorded and flushed per step.
5. **EVALUATE** — `Evaluator.evaluate_action()` per step,
   `evaluate_run()` at the end (`EVALUATION_COMPLETED`).
6. **UPDATE STATE** — goal transitions to `completed` iff the cycle held
   **and** the evaluation passed, else `failed`; the run row is finalized;
   `RUN_COMPLETED`/`RUN_FAILED` traced.
7. **SUMMARY** — a structured `RunSummary` (pydantic, JSON-serializable,
   includes the full final `AgentState`) is returned. `run_goal()` **never
   raises**: planner crashes, evaluator crashes, storage outages and broken
   sinks all degrade into a `failed` summary.

**Status semantics (stable contract):** run `status` = *cycle integrity*;
`evaluation.passed` = *quality*; `summary.ok` = both. A refused action fails
the evaluation, not the cycle; a crashed planner fails the cycle.

## 4. How to run Skynet

```bash
# With the real database (skynet-postgis container running):
.venv/Scripts/python.exe -m agency.cli run --goal "Report the strongest recent earthquake"

# Deterministic, database-free:
.venv/Scripts/python.exe -m agency.cli run --goal "Demo" --storage memory \
    --steps '[{"type": "echo", "params": {"text": "hello"}}]'

# JSON output, custom budget:
.venv/Scripts/python.exe -m agency.cli run --goal "Quick" --json --max-steps 3
```

Exit codes: 0 = run completed (and evaluation passed), 1 = run failed,
2 = CLI usage error. The first database-backed run calls
`ensure_core_schema()` to create the four Skynet tables (idempotent).

## 5. Extension points

### Add a new action/tool

```python
from agency.actions.base import Action, ActionContext, ActionSpec

class WebSearchAction(Action):
    name = "web_search"          # registry key
    category = "web"             # core | web | comms | lab
    description = "Search the web."

    async def execute(self, spec: ActionSpec, ctx: ActionContext) -> object:
        return {"results": [...]}
```

Register it either via `bootstrap.register_action(registry, WebSearchAction(), settings)`
(gated by the category's feature flag — `web` requires `SKYNET_ENABLE_WEB_TOOLS=true`)
or pass `extra_actions=(WebSearchAction(),)` to `build_core()`. Then allow it:
`SKYNET_ENABLED_ACTIONS=echo,gods_eye_latest_events,web_search`. **The core loop
does not change.** Out-of-the-box categories not yet implemented are gated dark.

### Add a new planner

```python
from agency.planner import Planner, Plan
from agency.state import AgentState
from agency.goals import Goal

class LLMPlanner(Planner):
    name = "llm"
    async def plan(self, goal: Goal, state: AgentState) -> Plan:
        ...  # call the provider layer (intelligence phase)
```

Inject with `build_core(settings, planner=LLMPlanner())` — or wire it in
`bootstrap.build_core` behind `SKYNET_DEFAULT_PROVIDER`. The run row records
`planner` per run, enabling A/B comparison later.

### Add a new evaluator

Same pattern: subclass `Evaluator`, implement `evaluate_action` /
`evaluate_run`, inject via `build_core(..., evaluator=...)`. Planned variants
(interface already supports them): LLM-judge, benchmark-based, statistical,
human-feedback, multi-agent.

### New observation sources

Implement `PerceptionAdapter.observe(state) -> list[Observation]` and select it
via configuration. Sensors may fail freely: return an `error`-kind observation
and the run continues.

## 6. How experiences are recorded

`ExperienceRecorder` buffers `ExperienceRecord`s — one per observation,
action, evaluation and error, plus a final `summary` record — and flushes them
after every stage to the configured `ExperienceSink`:

- `InMemoryExperienceSink` — tests/dry runs.
- `DatabaseExperienceSink` — the `experiences` table
  (`kind`, `summary`, JSONB `payload`, `outcome`, `confidence`, run/goal FKs).

Flushes are best-effort inside the loop (`_safe_flush`): a storage outage is
logged and the run still completes with its summary. This table is the
substrate future learning (memory, knowledge extraction, self-improvement
weakness detection) will mine.

## 7. Persistence & the God's Eye boundary

- Additive tables only: `goals`, `runs`, `run_events`, `experiences`
  (ORM in `database/models/skynet_core.py`, naming conventions inherited from
  `database/base.py`). The `events` table and all God's Eye code are untouched.
- God's Eye → Skynet flows through exactly two read-only seams:
  `GodsEyePerceptionAdapter` (OBSERVE stage) and `GodsEyeLatestEventsAction`
  (tool use). Neither writes to God's Eye state.
- Alembic migrations will supersede `ensure_core_schema()` in the next
  infrastructure phase; the models are migration-ready.

## 8. Safety boundaries enforced in this phase

- **Action allow-list** (`SKYNET_ENABLED_ACTIONS`) — unknown or unpermitted
  actions are refused at selection time and recorded as failures.
- **Dark capability flags** — web/comms/lab action categories cannot execute
  until explicitly enabled by configuration; no such actions exist yet.
- **Budgets** — step and wall-clock limits end runs deterministically.
- **No self-modification, no shell, no package installs, no credential
  access, no deployments** — no such actions exist in the registry, and the
  category gating gives future actions the same structure of
  proposal → review → explicit enablement.

## 9. Test map

| File | Covers |
|---|---|
| `tests/test_config.py` | defaults, dark flags, allow-list parsing, env overrides, validation errors |
| `tests/test_observation.py` | kinds, confidence bounds, JSON round-trip |
| `tests/test_state.py` | serialization round-trip, mutators, status constraint |
| `tests/test_goals.py` | lifecycle transitions, illegal/terminal/idempotent cases, manager paths |
| `tests/test_planner.py` | interface, deterministic strategy, malformed-step rejection |
| `tests/test_actions.py` | registry, duplicate guard, success/exception capture, built-ins, DB degradation |
| `tests/test_evaluator.py` | action/run scoring incl. zero-action and error paths |
| `tests/test_experience.py` | buffering, flush, deep-copy isolation, validation |
| `tests/test_trace.py` | required event set, ordering, fanout, JSON logging, broken-sink isolation |
| `tests/test_perception.py` | adapters, normalization, dead-DB degradation, builder errors |
| `tests/test_storage.py` | memory backend, protocol, table-set restriction, laziness |
| `tests/test_bootstrap.py` | composition, category gating, DB wiring |
| `tests/test_loop.py` | full lifecycle, budgets, crash-proofing, trace ordering, status semantics |
| `tests/test_cli.py` | parser, exit codes, JSON output, error paths |
| `tests/test_integration_gods_eye.py` | **db-marked**: real PostgreSQL round-trip + app contract |

Run: `.venv/Scripts/python.exe -m pytest -v` (db-marked tests auto-skip
without PostgreSQL).
