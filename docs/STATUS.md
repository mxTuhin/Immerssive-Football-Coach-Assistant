# Build status vs. the build plan

Source of truth for the plan: [`docs/plan/03-build-plan.md`](plan/03-build-plan.md). This file tracks what exists
in code, what was measured, and what is next. Update it in every PR that moves an item.

Legend: ✅ built and tested · 🟡 built, needs real-data validation (GPU / downloaded data) · ⏸️ pending (later phase) · ❌ not started

## Phase 1 (this branch): backend pipeline + server

### Pipeline stages (plan §4)

| # | Stage | Status | Code | Notes |
|---|---|---|---|---|
| 1 | Ingest | ✅ / 🟡 | `workers/pipeline/fmi_pipeline/stages/ingest.py` | ffmpeg 25 fps proxy, segment cut, scene-cut detection, squad CSV. Tested on synthetic video; real ffmpeg path needs V1. Resumable uploads + Veo/Pixellot import: later |
| 2 | Detection | 🟡 | `stages/detect.py`, `models/pretrained.py`, `models/base.py` | roboflow/sports YOLO weights via ultralytics; own tiling + NMS for 4K (SAHI optional). Needs weights (`scripts/fetch_data.py --models`) |
| 3 | Calibration | 🟡 | `stages/calib.py`, `geometry.py` | 32-keypoint model → RANSAC homography; static (solve once + verify) and moving (keyframes + median smoothing + interpolation); line-alignment check on painted lines; held-out reprojection; one review item per failed shot; manual 4-point override |
| 4 | Online tracking | ✅ / 🟡 | `tracking/tracker.py`, `stages/track.py` | BoT-SORT style, gates in pitch metres, IoU + distance + OSNet appearance, weak boxes in 2nd pass, new IDs ≥ 0.6, ball box inflation |
| 5 | Offline linking | ✅ | `tracking/linking.py`, `stages/link.py` | Global min-cost assignment over tracklet ends/starts; feasible speed ≤ 10 m/s; class/team conflicts forbidden; human merge/split |
| 6 | Team & roles | ✅ / 🟡 | `stages/team.py` | SigLIP → UMAP → KMeans(2) (colour-histogram fallback); GK by side; referees by class + outliers; kit colours name clusters; ≤ 11 per team check |
| 7 | Identity | ✅ / 🟡 | `stages/identity.py` | PARSeq jersey OCR with confidence voting; squad lookup; temporal-conflict resolution; OSNet ReID inheritance; squad slots; review queue |
| 8 | Ball | ✅ | `tracking/ball_path.py`, `stages/ball.py` | One path per match by DP over top-K candidates, long-jump re-acquisition, gap interpolation (≤ 2 s, `estimated`), in-air flag |
| 9 | Projection & smoothing | ✅ | `stages/project.py`, `fmi_analytics/kinematics.py` | Vectorised CV Kalman + RTS smoother, speed cap 10 m/s, accel QA, `estimated` flag, periods + attack direction |
| 10 | Facing direction | ✅ | `fmi_analytics/facing.py`, `stages/facing.py` | §4.1 rules, circular Kalman forward/backward fusion, 720°/s cap; runs on open data too |
| 11 | Events | ✅ | `fmi_analytics/events.py`, `stages/events.py` | Possession with hysteresis → passes/interceptions/recoveries/shots/crosses/set pieces; SPADL view; human event edits |
| 12 | Analytics | ✅ | `fmi_analytics/{physical,shape,phases,pressing,xt,possession}.py` | Physical, shape, line height, compactness, phases, pressing + PPDA, xT (grid fitted on open data) |
| 13 | AI analyst | ✅ / 🟡 | `packages/agent/fmi_agent/` | Claude (`claude-opus-5-5`) + 11 deterministic tools, report + Q&A, grounding check; needs an API key to run for real |
| 14 | 3D / XR | ⏸️ | `fmi_analytics/scenes.py` (backend part ✅) | `scene.v1` packages are served by the API; web viewer + Quest app are phase 2 (`docs/XR_PLAN.md`) |
| 15 | Review UI | ✅ backend / ⏸️ UI | `fmi_pipeline/corrections.py`, API `/review`, `/corrections`, `/rerun` | Corrections re-run only affected stages (verified by tests); the web UI is phase 2 |

### Platform (plan §5, §9)

| Item | Status | Code |
|---|---|---|
| Stage contract (idempotent, hash-cached, QA per stage, re-run alone, fakes in CI) | ✅ | `fmi_pipeline/contract.py`, `runner.py` |
| Match Package schema (Parquet + JSON, kloppy-compatible adapters) | ✅ | `packages/schema/fmi_schema/` |
| SkillCorner + Metrica adapters | ✅ | `fmi_schema/adapters/` |
| FastAPI (matches, uploads, jobs, data, review, AI, open data) | ✅ (tests need CI deps) | `apps/api/fmi_api/main.py` |
| Celery + Redis queue, CPU/GPU queues | ✅ | `apps/api/fmi_api/tasks.py` |
| Postgres (SQLite in dev) | ✅ | `apps/api/fmi_api/db.py` |
| R2 / S3 storage mirror | 🟡 | `fmi_pipeline/storage.py` |
| Docker compose + GPU worker image | 🟡 | `infra/` |
| CI (tests + open-data event regression) | ✅ | `.github/workflows/ci.yml` |

### Measured so far

| What | Result | Target | How |
|---|---|---|---|
| Synthetic match, full pipeline with fake models | median position error 0.07 m, ball within 2 m 100%, 22/22 identified, teams 11/11 + referee | (sanity) | `tests/test_pipeline_e2e.py` |
| Auto passes vs SkillCorner provider events | **F1 0.77** (match 1886347), **0.745** (1899585) at ±1 s | ≥ 0.7 | `python -m fmi_eval.run opendata` |
| Auto passes vs Metrica events (sample game 1) | **F1 0.77** at ±1 s | ≥ 0.7 | `fmi_analytics.run.run_all` |
| Auto shots (same data) | F1 0.26–0.44 | ≥ 0.7 | known gap: ball-speed rule on broadcast-extrapolated balls; upgrade = learned action spotting |
| Real-video metrics on V1/V2/V3 | not run yet | §2 table | needs GPU + downloads (see below) |

## Next steps (in order)

1. On a GPU machine: `python scripts/fetch_data.py --models --videos`, then `fmi run --match-id v1 --video data/videos/v1_121364_0.mp4` (plan week 1).
2. Download one SoccerTrack v2 half (`--soccertrack`), run the pipeline with `--tier panoramic`, then `python -m fmi_eval.run v2` → first position-error number (plan weeks 2–3).
3. Hand-label 300 facing samples (`eval/labels/facing_template.csv`) on V2/V3 and run `fmi_eval.run facing` (weeks 6–7).
4. Improve shot detection (ball speed + height when available, goal-line crossings, learned spotting) until shot F1 ≥ 0.7.
5. Phase 2: web 2D minimap + three.js viewer + review UI (`apps/web`), then the Quest 3S app (`apps/quest`) — see `docs/XR_PLAN.md`.

## Pending by design (plan §10, parked)

Body keypoints, head direction, posture, scanning metric, engine animation, pose-driven bodies. The Match Package
reserves a `body` table for them.
