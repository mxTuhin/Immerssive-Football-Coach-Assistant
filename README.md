# Football Match Intelligence (Immersive Football Coach Assistant)

A coach uploads a match video and gets tracking in pitch metres, named players, the ball, which way every player
faces, events, physical and tactical analytics, an AI match report with key moments, and (phase 2) 2D/3D/XR
playback, with no command line involved.

**Phase 1 (this branch): the backend pipeline and the server.** The web viewer and the Quest 3S app are pending
(contract and plan ready in [`docs/XR_PLAN.md`](docs/XR_PLAN.md)).

| Doc | |
|---|---|
| [`docs/plan/03-build-plan.md`](docs/plan/03-build-plan.md) | The build plan (v2.0) |
| [`docs/STATUS.md`](docs/STATUS.md) | What is built, what was measured, what is next |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Packages, conventions, stage contract, decisions |
| [`docs/SETUP.md`](docs/SETUP.md) | Local setup after `git pull`: data, GPU worker, server |
| [`docs/XR_PLAN.md`](docs/XR_PLAN.md) | Web 3D viewer + Quest 3S plan (pending) |
| [`CLAUDE.md`](CLAUDE.md), [`.claude/skills/`](.claude/skills) | Guidance for Claude Code sessions in this repo |

## What's inside

```text
apps/api/            FastAPI: matches, uploads, jobs, data, review/corrections, AI analyst, open data
apps/web/, apps/quest/   phase 2 (pending)
workers/pipeline/    fmi_pipeline: stage contract + cached runner, pretrained model wrappers + fakes,
                     tracker (pitch metres), offline linking, ball DP, 13 stages, CLI `fmi`
packages/schema/     fmi_schema: Match Package (Parquet + JSON), pitch model, SkillCorner/Metrica/kloppy adapters
packages/analytics/  fmi_analytics: RTS smoothing, facing direction, possession, events, physical, shape, phases,
                     pressing, xT, QA, scenes, key moments
packages/agent/      fmi_agent: Claude + deterministic tools, grounding check, 30-question eval set
eval/                fmi_eval: SoccerTrack v2 loader, tracking/identity/ball/event/facing metrics
infra/               docker compose (Postgres, Redis, MinIO/R2, API, CPU + GPU workers)
data/opendata/       SkillCorner (20 matches) + Metrica samples, bundled (gzipped)
data/videos, models, data/soccertrack   large assets flagged with *.download files
```

Pipeline: `ingest → detect → calib → track → link → team → identity → ball → project → facing → events → analytics → package`.
Every stage is idempotent, cached by input hash, writes a QA record, and can be re-run alone after a correction.

## Quick start (CPU, no GPU or downloads needed)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt && pip install --no-deps -e .
pytest -q                     # tests with fake models
make synth                    # full video pipeline on a synthetic match
make opendata                 # SkillCorner match → analytics, facing, events, key moments
FMI_CELERY_EAGER=1 uvicorn fmi_api.main:app --reload     # http://localhost:8000/docs
```

Real video needs the pretrained weights and clips: `python scripts/fetch_data.py --list`, then `--models --videos`
(see [`docs/SETUP.md`](docs/SETUP.md)).

## Measured so far

* Synthetic match, whole pipeline with fake models: median position error 0.07 m, all 22 players identified.
* Our automatic pass detection vs the providers' events: **F1 0.77** (SkillCorner match 1886347 and Metrica game 1,
  ±1 s). Shot detection is still weak (F1 about 0.35–0.44); this is the next thing to improve.
* Real-video numbers (V1/V2/V3) need a GPU and the downloads: see `docs/STATUS.md` → Next steps.

## Data credits

SkillCorner open data (MIT), Metrica Sports sample data, roboflow/sports models and clips, SoccerTrack v2.
See [`data/README.md`](data/README.md).
