---
name: api-server
description: Run, extend or debug the FMI server - FastAPI app, Celery workers (gpu/cpu queues), Postgres/SQLite models, R2/S3 storage, docker compose - and add new endpoints with tests.
---

# API server

Code: `apps/api/fmi_api/` — `main.py` (routes), `service.py` (job work), `tasks.py` (Celery), `db.py`,
`schemas.py`, `settings.py` (env vars `FMI_*`, see `infra/.env.example`).

## Run
```bash
FMI_CELERY_EAGER=1 uvicorn fmi_api.main:app --reload          # inline jobs, SQLite, no Redis
celery -A fmi_api.tasks worker -Q gpu -c 1                    # GPU box: video pipeline
celery -A fmi_api.tasks worker -Q cpu -c 4                    # open data + AI reports
docker compose -f infra/docker-compose.yml --env-file infra/.env up -d api worker-cpu
```
Docs at `/docs`. Auth: set `FMI_API_TOKEN` → clients send `Authorization: Bearer <token>`.

## Endpoint map
matches (`POST/GET /matches`, `/video`, `/video/import`, `/squad`, `/process`, `/jobs`), data (`/summary`,
`/roster`, `/events`, `/metrics/{physical|sprints|shape_timeseries|phases|pressure|xt_moves|auto_vs_provider_f1}`,
`/key-moments`, `/tracking`, `/frame`, `/scenes/{moment_id}`, `/heatmap/{player_id}`, `/export/{events|roster|physical}.csv`),
review (`/review`, `/corrections`, `/rerun`), AI (`/report`, `/ask`), open data (`/opendata/catalog`, `/opendata/import`).

## Adding an endpoint
1. Read data through `service.load_package(match_id)` (cached by package mtime); return `to_jsonable(...)` or
   `_df(df)` so NaN becomes null.
2. Long work → a function in `service.py` that updates the `Job` row + a Celery task in `tasks.py` + `tasks.submit`.
3. Test with `TestClient` and `FMI_CELERY_EAGER=1` like `tests/test_api.py` (synthetic video, no GPU).

## Gotchas
- Jobs record failures on the Job row; eager mode swallows task exceptions after recording them.
- Corrections only apply to video matches (`source == "pipeline"`); the response says which stages will re-run.
- Keep the DB small: match data lives in storage (Match Package), the DB only tracks state.
