# Baseline Validation Report — God's Eye / GPDS Eye (SKYNET foundation)

**Date:** 2026-09-20 · **Branch:** `main` · **Verdict: BASELINE VALID — protected baseline confirmed**

---

## 1. Repository status

- Branch `main`, 2 commits, working tree clean before this report (this report is the only untracked file).
- Commit `04fc052` (tag `baseline-audit`): pristine snapshot of the repository as received (25 files; no functional changes).
- Commit `89992b3`: adds `docs/SKYNET_BLUEPRINT.md` (+576 lines, documentation only).
- Diff between `baseline-audit` and `HEAD` contains **exactly one file**: the blueprint document. All audit-phase changes are therefore documentation-only.
- One harmless repository-configuration change predates the blueprint commit and is inside the baseline commit itself: `.gitignore` gained a single `.freebuff/` line (local tooling state), explicitly recorded in that commit's message.
- No runtime source file (`main.py`, `connector_usgs.py`, `database/*`, `frontend/*`) has been modified in any commit or in the working tree. Zero code changes so far.

## 2. Existing functionality verified (all PASS)

| Check | Result |
|---|---|
| Dependency integrity | `pip check` → "No broken requirements found" |
| Backend importability | `main`, `connector_usgs`, `database`, `database.session`, `database.settings` import OK |
| Infrastructure | `skynet-postgis` container (postgis/postgis:16-3.4) **already running, healthy**, published on 5433 |
| Entry point | Real uvicorn boot of `main:app` on 127.0.0.1:8010 → startup complete, **graceful shutdown** (ingestion task cancelled, engine disposed) |
| Perception (live network) | Ingestion loop first tick fetched USGS `all_day.geojson` → HTTP 200 → **"Upserted 222 USGS earthquake events"** |
| Persistence | `/api/v1/health` → 200 `{"status":"ONLINE","database":"connected","event_count":495}` |
| API contract | `/api/v1/events` → 200, GeoJSON `FeatureCollection` with 495 features (sample: `ci41335535`, "12 km W of Anza, CA", mag 0.09) — the exact contract `frontend/src/App.tsx` consumes |
| Frontend build | `tsc -b && vite build` → exit 0 (29 modules, 225.92 kB JS, built in 1.31 s) |

## 3. Tests

- **Tests passed:** none exist — the repository ships **zero tests** (pre-existing gap; pytest is fully configured but `tests/` does not exist).
- **Tests failed:** none. `pytest -q` exits with code **5** = "no tests collected" (not a failure of existing tests).
- Functionality was instead verified by direct execution (Section 2), which is the strongest automated check available without writing new code (out of scope for this validation).

## 4. Build / type-check / lint results

| Command | Result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q` | exit 5 — 0 tests collected (pre-existing) |
| `.venv/Scripts/python.exe -m ruff check .` | exit 1 — **2 pre-existing findings** (below) |
| `cd frontend && npm run build` (includes `tsc -b` type-check) | exit 0 — clean |
| `.venv/Scripts/python.exe -m pip check` | exit 0 — clean |

## 5. Pre-existing problems (all inherited from the pristine baseline; NOT introduced by us)

1. **No test suite** — pytest exits 5; `tests/` absent despite full pytest configuration.
2. **Ruff findings (2):**
   - `database/base.py:37` — `RUF012` mutable default for class attribute (`type_annotation_map`). False-positive-ish for a SQLAlchemy declarative class attribute; code is correct at runtime.
   - `database/session.py:147` — `RUF100` unused `noqa: BLE001` (BLE rule family not enabled in ruff config).
   - Left unfixed deliberately: fixing them would modify baseline code, which this validation was forbidden to do. Trivial to resolve in Phase 0.
3. **Request-time DDL** — `main.py` and `connector_usgs.py` run `CREATE EXTENSION/TABLE IF NOT EXISTS` on every health check / ingestion; schema also duplicated in `database/init/02-events.sql` (3 sources of truth; no Alembic env despite alembic being installed).
4. **No authentication, CORS `allow_origins=["*"]`, unbounded `/api/v1/events`** — acceptable for local dev, listed for the record.
5. **Port defaults are inconsistent** — frontend code defaults to API `:8080`, `.env.example` says `:8000`. Environment observation: **on this machine, 127.0.0.1:8000 is occupied by an unrelated container** (`open-notebook-local-podcast-setup-surrealdb-1`), so the SKYNET API must run elsewhere (validation used 8010). Reinforces making the port fully env-driven in Phase 0.
6. **Missing pieces documented in the blueprint**: no `connectors/`/`intelligence/`/`backend/` packages, no Alembic setup, no README, no CI; stray files (root `package-lock.json` stub, 3 MB `.docx`) retained per owner decision.

## 6. Problems introduced by our changes

**None.** Zero code modifications; the only deltas vs. the pristine snapshot are (a) one `.gitignore` line and (b) two documentation files (`docs/SKYNET_BLUEPRINT.md`, this report). Every failure above reproduces identically on the pristine `baseline-audit` tag.

## 7. Exact commands used

```bash
git status && git log --oneline --decorate && git diff && git diff --cached && git diff baseline-audit..HEAD --stat
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -c "import main, connector_usgs, database, database.session, database.settings; print('backend imports OK')"
netstat -ano | grep LISTENING | grep -E ":(8000|8080|5173|5433)\s"   # found 5433 healthy; 8000 occupied by unrelated container
tasklist //FI "PID eq 21280"                                          # 8000+5433 listener = com.docker.backend.exe
docker ps                                                             # skynet-postgis Up (healthy); unrelated containers untouched
.venv/Scripts/python.exe -m pytest -ra                                # 0 collected
.venv/Scripts/python.exe -m pytest -q >/dev/null 2>&1; echo $?        # exit 5
.venv/Scripts/python.exe -m ruff check .                              # 2 pre-existing findings
cd frontend && npm run build                                          # tsc -b && vite build → exit 0
# Live entry-point smoke test (uvicorn programmatically on 127.0.0.1:8010,
# real TCP requests to /api/v1/health and /api/v1/events, then clean shutdown):
#   uvicorn.Config(main.app, port=8010) → started=True
#   GET /api/v1/health  → 200 {"status":"ONLINE","database":"connected","event_count":495}
#   GET /api/v1/events  → 200 FeatureCollection, 495 features
#   ingest loop: GET earthquake.usgs.gov ... 200 → "Upserted 222 USGS earthquake events"
#   server.should_exit=True → shutdown complete, no errors
```

## 8. Recommended next step

Proceed to **Phase 0 of `docs/SKYNET_BLUEPRINT.md`** (Foundation & Safety), in this order:

1. Alembic init + baseline migration reproducing the `events` schema exactly; remove request-time DDL from `main.py` / `connector_usgs.py` (behavior-preserving; contract tests first).
2. Create `tests/` (DB-free unit tests + `db`-marked integration tests against the running container) and fix the two trivial ruff findings.
3. Resolve the port/URL mismatch via env configuration only.
4. Create the declared empty packages (`connectors`, `intelligence`, `backend`, `agency`, `lab`) so the import graph exists — no logic yet.

No Skynet feature modules were implemented, per instructions. The working state is preserved: containers untouched (only reads; the pre-existing unrelated containers were left alone), no code changed, baseline tag intact.
