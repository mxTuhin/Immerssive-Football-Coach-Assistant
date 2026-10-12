# CLAUDE.md — Football Match Intelligence (FMI)

Video of a football match → tracking in pitch metres → identities, ball, facing → events → analytics →
AI analyst → 3D/XR. Plan: `docs/plan/03-build-plan.md`; progress: `docs/STATUS.md`; design: `docs/ARCHITECTURE.md`.

## Layout
- `packages/schema` (`fmi_schema`): Match Package, pitch, IO, adapters (SkillCorner, Metrica, kloppy)
- `packages/analytics` (`fmi_analytics`): smoothing, facing, possession, events, metrics, QA, scenes, moments
- `packages/agent` (`fmi_agent`): AI analyst tools + Claude loop + eval set
- `workers/pipeline` (`fmi_pipeline`): stage contract/runner, models (pretrained + fakes), tracker, stages, CLI `fmi`
- `apps/api` (`fmi_api`): FastAPI + Celery + DB; `apps/web`, `apps/quest`: phase 2 (pending)
- `eval` (`fmi_eval`): SoccerTrack v2 loader, metrics, eval CLI; `infra/`: docker; `scripts/`: data tools

## Commands
```bash
pip install -r requirements/dev.txt && pip install --no-deps -e .   # CPU dev
pytest -q                                   # all tests, fake models, no GPU
make synth                                  # full pipeline on synthetic video
make opendata                               # SkillCorner → Match Package
python scripts/fetch_data.py --list         # large assets flagged with *.download
FMI_CELERY_EAGER=1 uvicorn fmi_api.main:app --reload
```
Without installing, tests can be run with `PYTHONPATH=packages/schema:packages/analytics:packages/agent:workers/pipeline:apps/api:eval`.

## Conventions (non-negotiable)
- Metres, origin at the centre spot, x along the length, y across, y up; angles in degrees CCW from +x.
- Attacking direction only via `meta.periods[].home_attack_sign` / `fmi_analytics.common.attack_sign`.
- `period` + `t` (period clock, s); `frame` = global 10 Hz index (proxy-video frames before the `project` stage).
- Every output carries confidence (`confidence`, `estimated`, `yaw_conf`, event `source`/`confidence`); every stage
  returns a `StageQA`. Wrong data is flagged or sent to review, never shown as fact.
- Stage changes: follow `.claude/skills/pipeline-stage/SKILL.md` (hash inputs, bump `version`, add fakes + tests).
- Algorithms on tracking go in `fmi_analytics` so they run on open data and pipeline output alike.
- Heavy deps (torch, ultralytics, cv2, transformers, anthropic) are imported lazily; core stays CPU-importable.
- Data rule: files < 100 MB are committed (gzip text first); larger/gated ones get a `*.download` flag file.
- AI analyst: every number from a tool result; model `claude-opus-5-5`; no player ratings.
- 3D/XR clients are pending; keep the `scene.v1` contract stable (`fmi_analytics/scenes.py`).

## Skills
`.claude/skills/`: pipeline-stage, run-pipeline, fetch-data, evaluate, open-data, api-server, ai-analyst,
review-corrections, web-viewer (pending), xr-quest (pending).
