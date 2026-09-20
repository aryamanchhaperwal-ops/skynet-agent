# SKYNET — Technical Audit & Implementation Blueprint

**Status:** Phase 0 deliverable — audit complete, no code changes made.
**Scope:** Audit of the repository as it actually exists in this workspace, followed by a concrete, phased blueprint for transforming it into Skynet.

---

## Part 1 — Technical Audit of the Existing Repository

### 1.1 What this repository actually is

The workspace contains a compact, self-described **SKYNET prototype**: a global-intelligence / early-warning platform with a Python FastAPI backend that ingests public geophysical and news data sources into PostgreSQL+PostGIS, and a React + CesiumJS 3D globe frontend that visualizes ingested events. It is already partially "Skynet-branded" (package name `skynet`, `SKYNET_` env prefix, "SKYNET | Earth Intelligence" UI title) but contains **no agent loop, no LLM calls, no memory, no experimentation, and no self-improvement code**. It is a data-pipeline + visualization prototype — a solid perception/storage substrate, not an agent.

**Verified state:** every Python module imports cleanly under the project's own `.venv` (Python 3.11.9); the frontend has a production build in `frontend/dist`; a `frontend/node_modules` install is present.

### 1.2 Complete file inventory (all source files, verified)

```
main.py                          FastAPI app + scheduled USGS ingestion loop (~150 lines)
connector_usgs.py                USGS earthquake GeoJSON fetch + upsert (~130 lines)
pyproject.toml                   Package metadata, deps, pytest/ruff config
docker-compose.yml               PostGIS 16-3.4 container only (host port 5433)
.env.example                     ~120 lines of documented configuration
.gitignore                       Secrets, Python, Node, local runtime artifacts
package-lock.json                Root stub (name "skynet", empty packages) — vestigial
database/                        Persistence layer
  __init__.py                    Docstring-only; exports Base
  base.py                        DeclarativeBase + naming convention + timestamps mixins
  session.py                     Async engine/session factory, session_scope, health probe
  settings.py                    DatabaseSettings (pydantic-settings, SKYNET_ prefix)
  init/01-extensions.sql         postgis, postgis_topology, pg_trgm, `skynet` schema
  init/02-events.sql             `events` table + indexes
frontend/                        React 19 + Vite 7 + Cesium 1.136
  package.json                   dev/build/preview scripts (no lint/test scripts)
  vite.config.ts                 react + vite-plugin-cesium, port 5173
  index.html                     SPA shell
  src/main.tsx, src/App.tsx      Single-file app: globe + event pins + detail card
  src/styles.css                 Sci-fi dark HUD styling
  dist/                          Prebuilt production bundle
Personalized Learning Roadmap Skill.docx   Unrelated document (3 MB)
.freebuff/, .ruff_cache/, __pycache__/, .venv/   Tooling artifacts (not source)
```

**Total first-party source: 12 Python files (4 modules), 4 frontend files, 2 SQL files, 3 config files. This is a small codebase — a full read was performed.**

### 1.3 Languages, frameworks, versions

| Layer | Technology | Version evidence |
|---|---|---|
| Backend language | Python | `requires-python = ">=3.11"`; venv is 3.11.9 (Windows/MSYS layout) |
| Web framework | FastAPI | 0.141.1 installed |
| ASGI server | uvicorn[standard] | installed (watchfiles, websockets, httptools present) |
| Data validation | pydantic 2.x, pydantic-settings | 0.0.5 / 2.x installed |
| ORM / DB access | SQLAlchemy 2 async + asyncpg | installed |
| Geospatial | PostGIS 3.4 (PostgreSQL 16), GeoAlchemy2, `geometry(PointZ, 4326)` | docker image `postgis/postgis:16-3.4` |
| Migrations | Alembic | 1.20.0 installed, **no `alembic/` directory or `alembic.ini` exists** |
| HTTP client | httpx | installed |
| Retry/robustness | tenacity | installed (referenced by .env.example connector tuning) |
| Feed parsing | feedparser (RSS/Atom), python-dateutil, orjson | installed |
| Testing | pytest, pytest-asyncio, pytest-cov, respx | installed |
| Linting | ruff (E,F,I,UP,B,ASYNC,RUF; line 100; py311) | installed |
| Frontend | React 19, Vite 7, TypeScript 5.8 | package.json |
| 3D globe | CesiumJS 1.136 + vite-plugin-cesium | package.json |
| Infra | Docker Compose (single PostGIS service) | docker-compose.yml |

### 1.4 Entry points

- **Backend:** `main.py` → `app = FastAPI(title="SKYNET API", lifespan=lifespan)`. Run via `uvicorn main:app`. The `lifespan` hook starts `asyncio.create_task(_ingestion_loop())` which calls `ingest_usgs()` every 300 s, cancels the task and disposes the DB engine on shutdown.
- **Frontend:** `frontend/src/main.tsx` → `App.tsx`, Vite dev server on port 5173, build via `npm run build`.
- **Infrastructure:** `docker compose up -d` (PostGIS only; app processes run on host by design).

### 1.5 Existing AI/ML components

**None.** No LLM client, no model code, no embeddings, no agent logic exists anywhere in the tree. However, the **configuration substrate for an AI layer already exists** and is well designed:
- `.env.example` defines `SKYNET_AI_PROVIDER` (`auto | anthropic | openai | deterministic`), `SKYNET_AI_TIMEOUT_SECONDS`, `SKYNET_AI_MAX_OUTPUT_TOKENS`, `ANTHROPIC_API_KEY`, `SKYNET_ANTHROPIC_MODEL` (claude-sonnet-4-5), `OPENAI_API_KEY`, `SKYNET_OPENAI_MODEL` (gpt-4o-mini), `SKYNET_OPENAI_BASE_URL`.
- Its comments describe an intended contract: *"When no provider key is present SKYNET falls back to the deterministic evidence-only provider, which restates retrieved evidence without inventing any. Investigations therefore always work."*

This tells us the upstream project planned an "investigations" feature with provider abstraction and a deterministic fallback — the exact seam Skynet needs. It is **planned but not implemented**; Skynet should honor this contract.

### 1.6 Vision / perception components

- **No camera, CV, or imagery pipeline exists.**
- The de-facto "perception" today = scheduled HTTP ingestion of public feeds:
  - `connector_usgs.py`: USGS `all_day.geojson` → validated → upsert into `events` (GeoJSON-compatible JSONB `properties`, `ST_MakePoint` 3D geometry, `ON CONFLICT` upsert).
  - `.env.example` documents four more planned connectors: `nws_weather_alerts`, `gdacs_disasters`, `gdelt_news` (via `SKYNET_ENABLED_CONNECTORS`), plus a credential-gated **ACLED** conflict-data connector (`SKYNET_ACLED_API_KEY`/`SKYNET_ACLED_EMAIL`, "present in the codebase upstream but refuses to run without a key" — **not present in this workspace**).
  - Connector etiquette knobs exist: `SKYNET_CONNECTOR_USER_AGENT` (api.weather.gov requires a descriptive UA), `SKYNET_CONNECTOR_TIMEOUT_SECONDS`, `SKYNET_CONNECTOR_MAX_RETRIES`.

### 1.7 APIs

- `GET /api/v1/health` — runs DDL defensively (`CREATE EXTENSION IF NOT EXISTS postgis`, `CREATE TABLE IF NOT EXISTS events`), returns `{status, database, event_count}`; degrades to `database: disconnected` instead of failing.
- `GET /api/v1/events` — returns the whole `events` table as a GeoJSON `FeatureCollection` (unordered pagination, no filters, no auth).

### 1.8 Database / storage

- PostgreSQL 16 + PostGIS 3.4 via asyncpg; SQLAlchemy 2 async ORM scaffolding exists (`database/base.py` with a disciplined constraint naming convention, `CreatedAtMixin`/`TimestampMixin`, timezone-aware-UTC policy).
- **The only table is `events`.** There is no `database/models.py` and no `database/repositories.py`, although `database/__init__.py` documents that they *should* contain "sources, observations, events, relationships, anomalies, investigations, assessments, predictions, outcomes, evaluations, improvement candidates, backtest runs and model versions." That docstring is effectively the upstream project's intended domain model — and it maps almost one-to-one onto Skynet's needs (evaluation, improvement candidates, backtest runs, model versions).
- Schema management is currently ad hoc: DDL is executed inline at request time in `main.py` and in `connector_usgs.py` (`CREATE TABLE IF NOT EXISTS ...` on every health check / ingestion run). Alembic is a declared dependency with zero migrations.
- `database/init/*.sql` bootstraps extensions and the `events` table on first container start; a `skynet` schema is created for future analytical helpers.
- Operational tuning already in settings: pool size/overflow, recycle, server-side `statement_timeout`, `pool_pre_ping`, `SKYNET_EVENT_RETENTION_DAYS` for pruning.

### 1.9 Agent logic

None. The only autonomous behavior is the 5-minute ingestion loop in `main.py`.

### 1.10 Existing tools

In the agent sense: none. In the developer sense: pytest (with `db`/`network` markers defined in pyproject — auto-skipping when DB is unavailable/offline), ruff, Alembic, Docker Compose, Vite/tsc.

### 1.11 Existing UI

Single-view Cesium globe: status HUD (backend ONLINE/UNAVAILABLE, event count, terrain badge), magnitude-scaled colored points (≥5.0 orange, else cyan), click-to-select event card (magnitude, place, depth, time, event id, source), optional Cesium world terrain when an Ion token is provided. No routing, no state library, no component library, no tests.

### 1.12 Configuration & environment variables

Single root `.env` consumed by both halves (per `.env.example` header). Full `SKYNET_`-prefixed namespace covering: runtime (`SKYNET_ENV`, log level, CORS origins), database, Cesium Ion token (`NEXT_PUBLIC_*` inlined client-side), AI provider block, connector block, anomaly-engine tuning (`SKYNET_ANOMALY_*` — lookback, baseline windows, min events, alpha with multiple-comparison correction, z-threshold), correlation-engine tuning (`SKYNET_CORRELATION_*` — radius km, window hours, min lift, max results), and forecasting (`SKYNET_FORECAST_*` — horizon hours, baseline weeks). **The anomaly/correlation/forecast settings describe intelligence features that do not exist in this workspace** — more evidence of a larger upstream codebase from which this fork was trimmed.

### 1.13 Dependencies

Backend runtime: fastapi, uvicorn[standard], pydantic, pydantic-settings, sqlalchemy[asyncio], asyncpg, geoalchemy2, alembic, httpx, tenacity, feedparser, python-dateutil, orjson. Dev: pytest, pytest-asyncio, pytest-cov, respx, ruff. All installed in `.venv`. Frontend: react, react-dom, cesium; dev: vite, plugin-react, plugin-cesium, typescript, @types/*. Note: **pydantic-settings is in `[project.dependencies]` but absent from `.env.example`-referenced runtime? — no, it is installed and used by `database/settings.py`; no gap.** No LLM SDKs (anthropic/openai clients) are installed yet.

### 1.14 Docker / container configuration

Single service: `postgis/postgis:16-3.4` on host port **5433** (deliberately off 5432), named volume `skynet-pgdata`, init scripts mounted read-only into `docker-entrypoint-initdb.d`, healthcheck via `pg_isready`, deterministic `--locale=C` collation. No backend or frontend containers — hot-reload-first local workflow.

### 1.15 Tests

**Zero test files exist**, despite pytest being fully configured (testpaths `tests`, `pythonpath = .`, asyncio auto mode, `db`/`network` markers). The `tests/` directory itself is absent. This is the largest immediate quality gap.

### 1.16 Build system

- Backend: setuptools (`[build-system]`), four intended top-level packages (`database`, `connectors`, `intelligence`, `backend`) with a documented one-way dependency rule (`database ← connectors ← backend`, `database ← intelligence ← backend`). **Only `database` exists as a package.**
- Frontend: `tsc -b && vite build`; `frontend/dist` contains a prior successful build (including copied Cesium assets).

### 1.17 Existing automation

Only the in-process ingestion loop. No cron, no task queue, no CI (no `.github/`), no pre-commit.

### 1.18 Existing network functionality

Outbound: httpx feed fetches (USGS today; NWS/GDACS/GDELT/ACLED configured but unimplemented). Inbound: the two FastAPI endpoints, CORS wide open (`allow_origins=["*"]`). The frontend calls the API from the browser using `VITE_SKYNET_API_BASE_URL` (default `http://localhost:8080` in code, `http://localhost:8000` in `.env.example` — a minor inconsistency to reconcile).

### 1.19 Existing security mechanisms

**Essentially none beyond hygiene practices.** Findings:
- CORS `allow_origins=["*"]` with `allow_credentials=False` — fine for local dev, must be tightened via the already-defined `SKYNET_CORS_ORIGINS` before any deployment.
- No authentication/authorization on any endpoint; `/api/v1/events` exposes the full table unbounded.
- `.env` is git-ignored; secrets are env-only (good); Cesium Ion token is client-scoped by policy (good).
- No rate limiting, no request validation beyond FastAPI defaults, no security headers.
- Endpoints execute DDL (`CREATE EXTENSION/TABLE`) at request time — convenient for a prototype, wrong for production; must move to migrations.
- No secrets in source tree (verified).

### 1.20 Directory/module responsibility summary

| Path | Responsibility | State |
|---|---|---|
| `main.py` | App factory, lifespan, ingestion scheduler, health + events endpoints | Working |
| `connector_usgs.py` | USGS feed → `events` upsert | Working |
| `database/` | Engine/session/settings/base; SQL bootstrap | Working but minimal vs. its documented intent |
| `frontend/src/App.tsx` | Entire UI | Working |
| `docker-compose.yml` | PostGIS only | Working |
| `pyproject.toml` | Deps, packaging, tool config | Working; describes aspirational packages |
| `.env.example` | Full config contract incl. AI layer | Working; documents unbuilt features |
| `tests/` | — | **Missing** |
| `alembic/` | — | **Missing** (dependency present) |
| `connectors/`, `intelligence/`, `backend/` | — | **Missing** (described in pyproject) |

### 1.21 Evidence of a trimmed upstream

The docstrings/config reference a much larger original: multi-connector ingestion, anomaly detection with statistical correction, spatio-temporal correlation, forecasting, investigations with LLM providers and a deterministic fallback, and a full domain model (sources, observations, relationships, anomalies, investigations, assessments, predictions, outcomes, evaluations, improvement candidates, backtest runs, model versions). This fork is a **reduced snapshot**. Practical consequence: the blueprint below rebuilds missing pieces only where Skynet needs them, and treats `.env.example` + `database/__init__.py` docstrings as the upstream design contract to honor.

---

## Part 2 — Component Classification

Legend: **A** KEEP AS-IS · **B** EXTEND · **C** REFACTOR · **D** REPLACE · **E** REMOVE · **F** UNKNOWN

| # | Component | Class | Rationale |
|---|---|---|---|
| 1 | `database/base.py` (Base, naming convention, mixins) | **A** | High-quality, dependency-free, exactly what Skynet's ORM needs |
| 2 | `database/session.py` (engine, session_scope, health) | **A** | Correct lazy-init, pooling, timeouts; reusable for all future tables |
| 3 | `database/settings.py` (pydantic-settings pattern) | **A** | Template for every future layer's settings |
| 4 | `database/init/*.sql` | **A** | Correct bootstrap; extend with Skynet tables later via Alembic |
| 5 | `connector_usgs.py` ingestion logic | **B** | Keep behavior; move into `connectors/` package, generalize |
| 6 | `main.py` lifespan + `_ingestion_loop` | **B** | Keep pattern; generalize into a scheduler for many loops |
| 7 | `GET /api/v1/events` + GeoJSON shape | **B** | Keep contract (frontend depends on it); add pagination/filters |
| 8 | `.env.example` config contract | **B** | Keep and extend with Skynet sections; honor existing AI keys |
| 9 | `GET /api/v1/health` | **C** | Keep response shape; remove request-time DDL once Alembic lands |
| 10 | Frontend `App.tsx` globe + event card | **C** | Keep as Skynet's "situation room" shell; split into components, add panels for agent/memory/experiments |
| 11 | Frontend build tooling (Vite/Cesium/TS) | **A** | Modern, working, prebuilt |
| 12 | `docker-compose.yml` | **B** | Keep PostGIS; add optional services later (Redis, sandbox) |
| 13 | `pyproject.toml` packaging + ruff/pytest config | **B** | Keep; create the missing `connectors/`, `intelligence/`, `backend/` packages it declares |
| 14 | Inline DDL in `main.py` / `connector_usgs.py` | **D** | Replace with Alembic migrations (dependency already present) |
| 15 | Root `package-lock.json` (empty stub) | **E** | Vestigial; no root package.json exists. Remove after confirmation |
| 16 | `Personalized Learning Roadmap Skill.docx` | **E** | Unrelated to this codebase (confirm with owner before deleting) |
| 17 | CORS `allow_origins=["*"]` | **C** | Wire to `SKYNET_CORS_ORIGINS` (already defined) |
| 18 | `connector_usgs.py` at repo root | **C** | Relocate into `connectors/` per declared architecture |
| 19 | Tests | **B** | Create `tests/` (config already expects it) starting with DB-free unit tests |
| 20 | Alembic setup | **B** | Initialize env using `DatabaseSettings.sync_database_url` |
| 21 | AI provider layer | **B** | Build to the contract `.env.example` already defines (provider auto/deterministic fallback) |
| 22 | ACLED connector | **F** | Documented as existing upstream; absent here. Reimplement when credential is available |
| 23 | Anomaly/correlation/forecast engines | **F** | Tuning config exists, code absent. Decide: rebuild per config, or descope |
| 24 | Frontend API base URL default | **C** | Reconcile 8080 (code) vs 8000 (`.env.example`) |

**Nothing functioning is removed.** The only removal candidates are two inert files (#15, #16) pending owner confirmation.

---

## Part 3 — Skynet Architecture (based on the actual repo)

### 3.1 Design principles

1. **Preserve what runs.** The USGS pipeline, globe, and DB layer keep working at every phase; each phase ends with a system that still boots.
2. **Honor the declared architecture.** `pyproject.toml` already prescribes the package layout and dependency direction — Skynet fills the empty slots instead of inventing a new topology.
3. **Honor the declared config contract.** New behavior is added as new `SKYNET_*` settings following the existing pattern.
4. **Safety over capability.** The self-improvement layer proposes and benchmarks; it never silently rewrites the running system. Human approval gates promotions.
5. **Everything is recorded.** Every perception, conversation, claim, experiment, and version change is a row in Postgres with provenance.

### 3.2 Target package layout

```
skynet/
├── main.py                     # (existing, extended) FastAPI app + loop registry
├── connector_usgs.py           # (existing) → thin shim re-exporting connectors.usgs
├── pyproject.toml              # (existing, extended)
├── alembic/                    # NEW: versioned schema migrations
├── database/                   # (existing layer — bottom of the graph)
│   ├── base.py / session.py / settings.py      # unchanged
│   ├── models/                 # NEW: ORM models (Part 6 data model)
│   └── repositories/           # NEW: query objects per aggregate
├── connectors/                 # NEW package (declared in pyproject, now real)
│   ├── base.py                 # Connector protocol: name, schedule, fetch() -> observations
│   ├── usgs.py                 # migrated logic from connector_usgs.py
│   ├── nws_alerts.py / gdacs.py / gdelt_news.py / acled.py   # Phase P3
│   └── registry.py             # slug → connector, driven by SKYNET_ENABLED_CONNECTORS
├── intelligence/               # NEW package (declared in pyproject, now real)
│   ├── providers/              # LLM abstraction honoring .env.example contract
│   │   ├── base.py             # Provider protocol: complete(messages, tools) -> LLMResult
│   │   ├── anthropic_provider.py / openai_provider.py / deterministic.py
│   │   └── selector.py         # SKYNET_AI_PROVIDER=auto resolution + fallback
│   ├── memory/                 # episodic + semantic memory stores (Postgres + pgvector)
│   ├── knowledge/              # claim extraction, source credibility, knowledge graph
│   ├── reasoning/              # planner, critic, decomposition
│   ├── comms/                  # AI↔AI communication layer (Part 5)
│   └── analytics/              # anomaly / correlation / forecast (rebuild where needed)
├── agency/                     # NEW top-level package for the autonomous loop
│   ├── loop.py                 # the Skynet cycle (Part 4)
│   ├── goals.py / planner.py / executor.py / experience.py
│   ├── web/                    # web exploration: fetch, search, browse (Phase P4+)
│   └── tools/                  # typed tool registry with allow-lists
├── lab/                        # NEW: the controlled self-improvement laboratory
│   ├── experiments.py          # Experiment lifecycle state machine
│   ├── hypotheses.py           # ImprovementProposal generation & parsing
│   ├── benchmarks/             # versioned benchmark suites (Part 5 of loop)
│   ├── runner.py               # isolated execution of candidate versions
│   ├── promotion.py            # accept/reject decision logic + human gate
│   └── rollback.py             # version registry + instant restore
├── backend/                    # NEW package (declared in pyproject): API routers
│   ├── routers/events.py       # moved from main.py, contract unchanged
│   ├── routers/health.py
│   ├── routers/loop.py         # observe/pause/resume the autonomous loop
│   ├── routers/memory.py / knowledge.py / conversations.py
│   ├── routers/lab.py          # experiments, benchmarks, proposals, versions
│   └── schemas.py              # pydantic request/response models
├── frontend/                   # (existing) extended with operator panels
├── tests/                      # NEW: unit (db-free) + marked db/network suites
└── docs/                       # ADRs, roadmap, runbooks
```

Dependency direction remains strictly one-way: `database ← {connectors, intelligence, agency, lab} ← backend`.

### 3.3 Subsystem mapping (requirement → existing/new)

| Skynet requirement | Provided by |
|---|---|
| Perception | Existing connector pipeline (EXTEND to more sources) + `agency/web` exploration |
| Autonomous task execution | `agency/loop.py` — asyncio task started in FastAPI lifespan (pattern already proven by `_ingestion_loop`) |
| Web research / browsing | `agency/web` — httpx (already a dep) for fetch; Playwright added only when JS rendering is justified |
| AI↔AI communication | `intelligence/comms` — adapter-per-protocol over outbound httpx (Part 5) |
| Long-term / episodic memory | Postgres tables + pgvector embeddings; `intelligence/memory` |
| Knowledge & graph | `intelligence/knowledge` — claims, entities, edges in Postgres |
| Reasoning / planning | `intelligence/reasoning` + provider layer |
| Tool use | `agency/tools` — typed registry, capability-scoped |
| Experiment management | `lab/experiments.py` + DB state machine |
| Self-evaluation / benchmarking | `lab/benchmarks` — deterministic suites + LLM-judge with fixed rubric |
| Improvement proposals | `lab/hypotheses.py` — structured JSON proposals from the improver |
| Controlled experimentation | `lab/runner.py` — subprocess isolation, workspace copies, no network unless granted |
| Version management / rollback | `lab/rollback.py` — immutable version registry; promoted config/strategy artifacts |
| Logging / observability | structlog-style JSON logs, per-run run records, `/api/v1/loop/status` |
| Security | Tool allow-lists, egress policy, sandboxing, scoped keys (Part 7) |
| Human oversight | API + UI approval gates on every promotion; loop is pause/resumable |

---

## Part 4 — The Core Autonomous Loop

The loop is a **long-running asyncio task** cohabiting the FastAPI process (same proven pattern as the ingestion loop), driven by rows in Postgres so it survives restarts and is fully inspectable.

```
GOAL → OBSERVE → RESEARCH → INTERACT → REASON → ACT → RECORD EXPERIENCE
      → EVALUATE → IDENTIFY WEAKNESS → PROPOSE IMPROVEMENT → EXPERIMENT
      → BENCHMARK → ACCEPT/REJECT → UPDATE KNOWLEDGE → CONTINUE
```

### Stage contracts (who writes what)

| Stage | Module | Reads | Writes |
|---|---|---|---|
| GOAL | `agency/goals.py` | `goals` (human or self-subgoals) | goal status |
| OBSERVE | connectors + `agency/web` | enabled connectors, queue | `observations`, `experiences` (raw) |
| RESEARCH | `agency/web` + `intelligence/knowledge` | open questions, known claims | `sources`, `claims`, `experiences` |
| INTERACT | `intelligence/comms` | conversation targets | `agents`, `conversations`, `messages` |
| REASON | `intelligence/reasoning` | memory, knowledge, observations | `plans`, `tasks` |
| ACT | `agency/executor` + `agency/tools` | approved plan | `task_runs`, tool audit rows |
| RECORD | `agency/experience.py` | task_runs, messages, sources | `experiences` (structured) |
| EVALUATE | `lab/benchmarks` + task outcomes | experiences vs. success criteria | `evaluations` |
| IDENTIFY WEAKNESS | `lab/` analyzer | evaluations, failures | `failures` (categorized) |
| PROPOSE | `lab/hypotheses.py` (LLM) | failures + relevant knowledge | `improvement_proposals` |
| EXPERIMENT | `lab/experiments.py` + `runner.py` | proposals | `experiments`, candidate artifacts |
| BENCHMARK | `lab/benchmarks` | experiment runs | benchmark `evaluations` |
| ACCEPT/REJECT | `lab/promotion.py` | benchmark deltas + gates | `system_versions`, decision record |
| UPDATE KNOWLEDGE | `intelligence/knowledge` | accepted changes, new claims | knowledge graph updates |
| CONTINUE | scheduler | — | next cycle |

### Communication between stages

- **Not function calls — rows.** Each stage consumes and produces database entities. The loop scheduler picks up ready work (e.g., "goal has no active plan", "experiment AWAITS_REVIEW"), which makes the loop **resumable, auditable, and pausable** — any stage can be inspected or re-run from its row.
- **The scheduler.** One asyncio supervisor with independently-timed sub-loops (connector cadence from `SKYNET_ENABLED_CONNECTORS`; cognition cadence `SKYNET_LOOP_INTERVAL_SECONDS`; experiment cadence separate). Sub-loops never share a DB session; each unit of work opens its own `session_scope()` (existing helper).
- **Event spine.** A lightweight `events_log` table (run_id, stage, level, payload JSONB) is the bus: the API streams it to the UI, and stages append structured facts. No in-process message bus is introduced until scale demands it.
- **Budgets.** Every cycle carries a budget (max steps, max LLM calls, max wall-clock, max spend estimate) enforced in `agency/loop.py`; exhaustion ends the cycle with a recorded `Failure` rather than an infinite tangent.

---

## Part 5 — Controlled Self-Improvement Mechanism (`lab/`)

**Prime directive: Skynet never modifies its own executing code.** Improvements operate on declared, versioned **strategy artifacts** — prompt templates, planner policies, tool configurations, retrieval parameters, benchmark definitions — materialized as data, loaded at loop start. Code changes happen only through the normal human development workflow.

### Lifecycle

```
DETECTED  → weakness recorded with evidence (failures, evaluation deltas)
PROPOSED  → ImprovementProposal(s): hypothesis, change artifact (JSON), predicted effect
SPECIFIED → benchmark suite + acceptance criteria frozen BEFORE running
BUILDING  → candidate SystemVersion assembled (artifact bundle + parent pointer)
SANDBOXED → runner executes benchmarks in isolation:
              • subprocess with its own env, resource limits, no credentials
              • network egress denied unless the suite explicitly requires it
              • workspace = temp copy; artifacts are data files, not source
BENCHMARKED → candidate vs. baseline on identical suites, same seeds, N repeats
GATED     → decision by objective criteria:
              accept iff  primary_metric ≥ baseline + δ  AND  no regression > ε
                          on any guard metric  AND  N ≥ min_repeats  AND  variance
                          within bounds  AND  (default) human approval granted
ACCEPTED  → new SystemVersion marked `current`; parent retained
REJECTED  → recorded with full results; knowledge updated ("what didn't work")
ROLLED_BACK → any regression detected post-promotion reactivates the parent version
```

### Safeguards

1. **Artifact-typed improvements only.** The proposal schema has a closed set of `change_type` values (`prompt_template`, `planner_policy`, `tool_config`, `retrieval_params`, `benchmark_def`). Anything else is rejected at parse time — this is the structural guarantee against unrestricted self-modification.
2. **Frozen-before-run acceptance criteria.** `acceptance_criteria` are written into the experiment row before the runner starts and are compared verbatim afterwards.
3. **Statistical honesty.** Paired comparisons, minimum repeats, effect-size threshold δ, guard metrics (latency, cost, safety violations) with regression tolerance ε.
4. **Immutability.** `system_versions` rows are append-only; artifacts are content-addressed; rollback = flipping the `current` pointer, which is always one transaction.
5. **Human gate.** `SKYNET_LAB_REQUIRE_APPROVAL=true` (default): promotions sit in `AWAITS_REVIEW` until approved via the API/UI. Full-auto mode is a deliberate, logged configuration change.
6. **Rate limits.** Max concurrent experiments, max experiments per weakness per day, cooldown after a rollback.
7. **Complete audit.** Every experiment keeps: weakness evidence, hypothesis, artifact diff, benchmark raw results, decision, and who/what decided.

---

## Part 6 — Data Model (initial schemas)

All tables use the existing `Base` (named constraints, tz-aware UTC), live in Alembic migrations, and follow the `database/models/` package split. `*` = primary key (UUID), all FKs named per convention.

```
agent            id*, name, kind(self|external), provider, model, endpoint,
                 capabilities JSONB, trust_score, created_at
conversation     id*, agent_id→agent, channel(api|mcp|webhook|internal),
                 topic, started_at, ended_at, status
message          id*, conversation_id→conversation, role(system|user|assistant|
                 tool|external), content TEXT, token_count, created_at, meta JSONB
source           id*, kind(url|api|feed|agent|file), uri, title, publisher,
                 retrieved_at, credibility NUMERIC, fingerprint SHA256, meta JSONB
observation      id*, source_id→source, connector_slug, observed_at, content JSONB,
                 geom geometry(NULL,4326), event_time TIMESTAMPTZ
knowledge_node   id*, kind(entity|concept|event|strategy), label, summary,
                 embedding VECTOR(n), confidence, source_refs JSONB, unique(kind,label)
knowledge_edge   id*, from_node→knowledge_node, to_node→knowledge_node,
                 relation, weight, evidence_refs JSONB
claim            id*, statement TEXT, extracted_from→source, confidence,
                 stance, verified BOOL NULL, verification_method, created_at
experience       id*, task_id→task NULL, goal_id→goal NULL, kind(observation|
                 conversation|action|experiment|reflection), summary,
                 payload JSONB, embedding VECTOR(n), outcome, created_at
task             id*, goal_id→goal, plan_id→plan NULL, kind, status
                 (pending|running|succeeded|failed|cancelled), params JSONB,
                 result JSONB, started_at, finished_at
plan             id*, goal_id→goal, steps JSONB, created_by(planner|human),
                 status, created_at
goal             id*, parent_id→goal NULL, title, description, origin(human|self),
                 priority, status(active|paused|done|failed), success_criteria JSONB,
                 budget JSONB, created_at
experiment       id*, hypothesis_id→hypothesis, status lifecycle (Part 5),
                 suite_id→benchmark_suite, baseline_version→system_version,
                 candidate_version→system_version, results JSONB, decision,
                 decided_by, created_at, decided_at
hypothesis       id*, weakness_id→failure, statement, change_type
                 (prompt_template|planner_policy|tool_config|retrieval_params|
                 benchmark_def), artifact JSONB (content-addressed),
                 predicted_effect, risk_notes, status
benchmark_suite  id*, name, version, definition JSONB (cases, metrics, seeds,
                 guard_metrics), frozen BOOL, created_at
evaluation       id*, subject_type(task|experiment|conversation|benchmark_run),
                 subject_id, metric, value NUMERIC, details JSONB, created_at
improvement_proposal id*, hypothesis_id→hypothesis, author(skynet|human),
                 change_type, artifact_diff JSONB, predicted_effect,
                 acceptance_criteria JSONB, status, reviewed_by, reviewed_at
system_version   id*, version STRING UNIQUE, parent_id→system_version NULL,
                 artifacts JSONB (map change_type→content hash), status
                 (candidate|current|retired|rolled_back), promoted_at,
                 promoted_by, notes
capability       id*, name, description, current_version→system_version,
                 baseline_metric JSONB, enabled BOOL
failure          id*, task_id→task NULL, experiment_id→experiment NULL,
                 category, description, evidence JSONB, severity, created_at,
                 resolved BOOL
learned_strategy id*, name, statement, origin_experiment→experiment NULL,
                 evidence_refs JSONB, confidence, active BOOL
events_log       id*, run_id, stage, level, message, payload JSONB, created_at
```

### Relationships (cardinality)

```
goal 1─* goal (subgoals)        goal 1─* plan 1─* task 1─* task_run/experience
source 1─* observation          source 1─* claim
conversation 1─* message        conversation *─1 agent
knowledge_node 1─* knowledge_edge (both directions, typed relations)
failure 1─* hypothesis 1─1 improvement_proposal 1─* experiment
benchmark_suite 1─* experiment  experiment *─1 system_version (baseline)
                                experiment *─1 system_version (candidate)
system_version self-reference (parent) — forms the version tree
experience links optionally to task, goal, or stands alone (reflection)
evaluation polymorphic on (subject_type, subject_id)
```

**Note on pgvector:** requires the `pgvector` extension. It is a one-line addition to `docker-compose.yml` (use the `pgvector/pgvector:pg16` image or install the extension in the current image) plus an Alembic migration. Phase P2 dependency.

---

## Part 7 — AI-to-AI Communication Layer (design only)

### Principles

- **Legitimate interfaces only.** The layer speaks only to endpoints the operator has explicitly configured: public APIs, MCP-compatible services, documented agent protocols, or keys the human supplies. There is no discovery-by-scanning of third-party systems; "agent discovery" is limited to operator-curated directories/registries.
- **Everything is recorded.** Every inbound and outbound message is a `messages` row; conversations are auditable and replayable.
- **Treat external agents as sources, not oracles.** Their claims enter the knowledge base through the same claim-verification path as web sources, with credibility tracking per agent.

### Interface adapters (`intelligence/comms/adapters/`)

| Adapter | Transport | Use |
|---|---|---|
| `RestAdapter` | HTTP/JSON APIs (OpenAI-compatible chat endpoints, custom REST) | First target; many agent products expose this shape |
| `MCPAdapter` | Model Context Protocol (JSON-RPC over stdio/HTTP) | Talk to MCP servers as external reasoning participants; also lets Skynet *expose* tools later |
| `WebhookAdapter` | inbound HTTP endpoint Skynet exposes | Receive callbacks/messages from other systems (auth via shared secret) |
| `A2AAdapter` | Agent-to-Agent style HTTP protocol (agent cards, tasks) | When interoperating with A2A-speaking agents |

All adapters implement one protocol: `connect()`, `send(Message) → Reply`, `stream()`, `capabilities()`. Registry + retry/backoff reuse `tenacity` (already a dependency).

### Conversation policy engine (`intelligence/comms/policy.py`)

Before any outbound message: allow-list check (target configured?), budget check (per-conversation and global turn/spend caps), topic scope check (is this within an active goal?), PII/secret redaction of outbound payloads, rate limiting per target. Default posture: **no unsolicited initiation** — Skynet responds and asks follow-ups within approved conversations; initiating contact with new agents requires operator-configured targets.

### Dialogue manager (`intelligence/comms/dialogue.py`)

Multi-turn state machine per conversation: question → answer → **challenge** (the default follow-up is adversarial: ask for evidence, edge cases, or a counterargument) → compare with other agents' answers and internal knowledge → record agreement/disagreement as `claims` with provenance → decide incorporation (only after verification or corroboration).

### Evaluation before incorporation

External answers are never written directly into knowledge. They become `claims` with `verified=NULL` until: corroborated by ≥2 independent sources/agents, consistent with internal knowledge, or explicitly tested by a small experiment. Disagreements between agents are themselves valuable records (`claim` rows with conflicting stances) and can spawn research goals.

---

## Part 8 — Phased Implementation Roadmap

Each phase ends with the system still booting and the existing globe still working.

### P0 — Foundation & Safety (no new features)
**Objective:** make the repo safe to evolve: version control, migrations, tests, packaging hygiene.
- Components: `git init` + initial commit (workspace currently has **no `.git`**); Alembic init + baseline migration reproducing `events` exactly; create `tests/` with DB-free unit tests (event parsing/validation, settings validation, API contract via httpx TestClient) + `db`-marked tests; create empty `connectors/`, `intelligence/`, `backend/`, `agency/`, `lab/` packages so imports resolve; move DDL out of request paths (health/events read-only); fix frontend/backend port mismatch; decide fate of stub `package-lock.json` and the `.docx`.
- Files: new `alembic/`, `tests/`, package `__init__`s; edits to `main.py`, `connector_usgs.py`, `docker-compose.yml` (pgvector image optional here), `.env.example`.
- Acceptance: `pytest` green (db tests skip cleanly without Postgres); `uvicorn main:app` serves health + events identically; `alembic upgrade head` on a fresh DB produces a working system; ruff clean.
- NOT yet: no AI calls, no loop, no new connectors, no schema beyond `events`.

### P1 — Connector Framework & Perception Layer
**Objective:** turn one-off ingestion into a pluggable perception subsystem.
- Components: `connectors/base.py` protocol + `registry.py` (slug-driven, honors `SKYNET_ENABLED_CONNECTORS`); migrate USGS into `connectors/usgs.py` (root file becomes a shim); add `nws_alerts`, `gdacs`, `gdelt_news` (feedparser already a dep); `sources` + `observations` tables; retention pruning job (`SKYNET_EVENT_RETENTION_DAYS`); scheduler generalization in `main.py` (per-connector cadence).
- Tests: connector unit tests with respx (mocked HTTP); registry tests; one `network`-marked live test per connector.
- Acceptance: enabling/disabling connectors via env works with zero code changes; observations carry provenance; globe unchanged.
- NOT yet: no LLM, no browsing.

### P2 — Memory, Providers & Knowledge Core
**Objective:** give Skynet a brain that persists.
- Components: provider layer in `intelligence/providers` implementing the `.env.example` contract (auto/deterministic fallback; add `anthropic`/`openai` SDK deps); pgvector + `knowledge_node`/`knowledge_edge`/`claim`/`experience` tables; embedding pipeline (provider-backed, deterministic fallback = hash-based for tests); memory API (`/api/v1/memory/query`, `/api/v1/knowledge/*`); basic claim extraction from observations.
- Tests: provider selection/fallback tests (respx-mocked); embedding store round-trip (`db` marker); claim extraction unit tests.
- Acceptance: with no API key, everything still works via deterministic provider; with a key, real LLM available; knowledge queryable.
- NOT yet: no autonomous loop, no external conversations.

### P3 — Reasoning, Tools & Supervised Task Execution
**Objective:** single-goal, supervised autonomy.
- Components: `agency/goals|planner|executor|tools`; typed tool registry (http-fetch, DB-query, calculator, code-eval in subprocess); `plans`/`tasks`/`failures` tables; loop v1 **with human trigger** ("run one cycle on goal X" from UI/API); budgets and step limits; experience recording.
- Tests: planner unit tests with scripted provider responses; tool registry allow/deny tests; executor integration test on a trivial goal.
- Acceptance: a goal can be decomposed, executed with tools, and produce a recorded experience + evaluation; the loop never runs unattended yet.
- NOT yet: unattended continuous loop; self-improvement.

### P4 — Web Exploration
**Objective:** autonomous research within policy.
- Components: `agency/web` (search-provider adapter, fetcher with robots/UA etiquette reusing connector settings, readability extraction, dedup via source fingerprint); research sub-loop feeding claims/knowledge; domain allow/deny lists via new `SKYNET_WEB_*` settings.
- Tests: mocked search/fetch flows; politeness/rate-limit tests; extraction unit tests.
- Acceptance: given a goal, Skynet can research a topic and record sources+claims with provenance; egress restricted to policy.
- NOT yet: AI↔AI comms; experiments.

### P5 — AI↔AI Communication
**Objective:** external agents as research participants.
- Components: `intelligence/comms` per Part 7 (RestAdapter first, MCP next); policy engine; dialogue manager with challenge/comparison flows; conversation UI panel; agent registry with credibility tracking.
- Tests: adapter contract tests against local mock servers; policy enforcement tests; dialogue state machine tests.
- Acceptance: Skynet can hold a recorded multi-turn conversation with a configured external model/agent, challenge its answers, and file claims — nothing enters knowledge unverified.
- NOT yet: autonomous experiment loop.

### P6 — Experimentation & Self-Evaluation (`lab/` v1)
**Objective:** measure before improving.
- Components: `benchmark_suite` definitions (first suites: retrieval accuracy on a fixed QA set; planning success on scripted tasks; claim-verification precision); `experiments`/`evaluations`/`failure` plumbing; weakness analyzer (categorize failures from experiences); experiment runner in subprocess isolation; version registry (`system_versions`).
- Tests: benchmark determinism tests; runner isolation tests (no network, no credentials, temp workspace); state-machine transition tests.
- Acceptance: an experiment runs candidate-vs-baseline on a frozen suite and produces a defensible accept/reject record; nothing auto-promotes yet.
- NOT yet: improvement generation.

### P7 — Improvement Proposals & Controlled Promotion
**Objective:** close the loop, safely.
- Components: `hypotheses.py` (LLM-generated structured proposals over the closed `change_type` set); artifact store (content-addressed JSON); acceptance-gate evaluator; promotion workflow with human approval UI; rollback command + cooldown; rate limits.
- Tests: proposal schema rejection tests (illegal change types); gate math tests; rollback integration test; end-to-end: weakness → proposal → sandbox run → reject on regression.
- Acceptance: full cycle DEMONSTRATED end-to-end with a prompt-policy improvement: detected weakness → proposal → benchmarked → (accept or reject) → version tree intact → rollback proven.
- NOT yet: continuous unattended operation.

### P8 — The Continuous Autonomous Loop
**Objective:** run the full Part 4 cycle unattended, bounded by budgets and policy.
- Components: loop supervisor (resumable, pausable); event spine + live UI stream; anomaly/correlation/forecast engines rebuilt to match existing `SKYNET_ANOMALY_*`/`SKYNET_CORRELATION_*`/`SKYNET_FORECAST_*` settings (F-class items resolved here or descoped explicitly); operator dashboard (goals, runs, experiments, versions, spend).
- Acceptance: system runs for 24 h unattended in policy, produces daily evaluation reports, proposes ≤ N improvements, all promotions human-approved; kill-switch stops everything cleanly; restart resumes from DB state.
- NOT ever: unreviewed self-modification of source code; unbounded egress; undeletable memory.

### P9 — Hardening & Leverage
- Multi-agent internal debate (Skynet challenging *itself* before external agents), knowledge-graph dedup/decay, benchmark suite growth, cost telemetry per capability, optional queue/worker split if loops outgrow one process, deployment story (real containers, auth, CORS lockdown).

**Dependency chain:** P0 → P1 → P2 → P3 → P4/P5 (parallelizable) → P6 → P7 → P8 → P9. Phases P4 and P5 are independent of each other; everything depends on P2's provider layer.

---

## Part 9 — Repository Safety Plan

1. **Immediate: restore version control.** The workspace has no `.git` (the fork was copied as plain files). First action of P0 is `git init` + initial commit so every subsequent change is diffable and revertable. (If the original fork exists elsewhere, re-attaching its remote is even better — operator decision.)
2. **No deletions without owner sign-off.** The two removal candidates (`package-lock.json` stub, `.docx`) are flagged, not removed.
3. **Behavior-preserving refactors first.** USGS logic moves into `connectors/usgs.py` with the root module kept as a shim; endpoint contracts (paths, response shapes) are frozen and covered by contract tests before any router moves.
4. **Migrations, never request-time DDL.** All schema evolution goes through Alembic; `CREATE ... IF NOT EXISTS` calls leave the hot paths in P0.
5. **Upstream contract honored.** `.env.example` variable names and semantics are treated as a public contract; new settings extend, never rename.
6. **Skynet layer clearly distinguishable.** New packages (`connectors`, `intelligence`, `agency`, `lab`, `backend`) are exactly the packages `pyproject.toml` already declares — the boundary between upstream code and Skynet additions is the package boundary itself, documented in ADRs under `docs/adr/`.
7. **Architectural decisions recorded.** One short ADR per significant choice (e.g., ADR-0001: improvements are data artifacts, not code rewrites; ADR-0002: Postgres-only persistence incl. vectors; ADR-0003: loop lives in-process until scale demands otherwise).
8. **Milestone checkpoints.** Each phase ends with a commit tagged `skynet-pN`; every promotion in `lab/` is itself a recorded, revertable transaction.
9. **Secrets discipline.** Existing practice maintained: env-only secrets, git-ignored `.env`, client-scoped Cesium token; sandboxed experiment runner receives no credentials by default.

---

## Part 10 — Immediate Answers to Open Questions (for the operator)

1. **Missing upstream modules** (`connectors`, `intelligence`, `backend`, tests, Alembic env): the blueprint rebuilds them only where Skynet needs them. Flag if you have the original upstream and prefer porting instead.
2. **F-class items** (ACLED connector, anomaly/correlation/forecast engines): rebuild-to-config in P8, or descope. Default plan rebuilds them because the tuning config already exists.
3. **Root `package-lock.json` + `.docx`**: propose removal in P0; awaiting confirmation.
4. **Port mismatch** (frontend default 8080 vs `.env.example` 8000): propose standardizing on 8000 in P0.
5. **LLM keys**: none configured; everything through P2 works with the deterministic fallback, so no key is required to begin.
