# Architecture

```text
Web app (React + three.js) ── API (FastAPI) ── Postgres · Redis queue · R2 storage
        ▲                          │
        │                   GPU worker (Celery, queue "gpu")
 Quest 3S (Unity)   ingest → calib/detect → track → link → team → identity → ball → project → facing → events
        ▲                          │
        └──── Match Package (Parquet + JSON) ──► analytics ──► AI analyst (Claude + tools)
```

## Packages

| Package | Import | Role |
|---|---|---|
| `packages/schema` | `fmi_schema` | Match Package data model, pitch geometry, IO, provider adapters (SkillCorner, Metrica, kloppy) |
| `packages/analytics` | `fmi_analytics` | Tracking-level algorithms shared by pipeline and open data: smoothing, facing, possession, events, physical, shape, phases, pressing, xT, QA, scenes, key moments |
| `packages/agent` | `fmi_agent` | AI analyst: deterministic tools + Claude tool loop + grounding check + eval set |
| `workers/pipeline` | `fmi_pipeline` | Video pipeline: stage contract, runner/cache, storage, models (pretrained + fakes), tracker, linking, ball DP, stages, CLI |
| `apps/api` | `fmi_api` | FastAPI app, Celery tasks, DB models, services |
| `eval` | `fmi_eval` | Ground-truth loaders (SoccerTrack v2), metrics, evaluation CLI |

Dependency direction: `schema` ← `analytics` ← `agent`; `schema`, `analytics` ← `pipeline` ← `api`. Nothing
imports `api`.

## Conventions (read before writing code)

* **Coordinates**: metres, origin at the centre spot, x along the length (±L/2), y along the width (±W/2), y up
  in a top-down view. Angles in degrees, 0 = +x, counter-clockwise, wrapped to (−180, 180].
* **Attacking direction** lives in `meta.periods[i].home_attack_sign` (+1: home attacks +x). Use
  `fmi_analytics.common.attack_sign(meta, period, team)`; never assume.
* **Time**: `period` + `t` (seconds since the period start); `frame` is the global 10 Hz index
  (`meta.periods[i].start_frame + round(t * 10)`). Pipeline stages before `project` work in proxy-video frames
  (25 fps).
* **Ids**: tracking `id` is a roster `player_id`, a track id (`L12`) while unidentified, or `ball`.
* **Confidence everywhere**: tracking `confidence`, `estimated`, `yaw_conf`; events `confidence` + `source`
  (`auto` | `provider` | `human`); every stage writes a `StageQA` with review items.

## Stage contract and caching

`Stage.run(ctx, inputs) -> StageResult(artifacts, qa)`. The runner hashes (stage name, version, config sections,
dependency hashes, consumed corrections, source digest). Outputs live in
`storage/matches/<id>/stages/<stage>/<hash>/` with a `manifest.json`; an unchanged hash is a cache hit. A
correction changes only the hash of the stage that consumes it (and its descendants), so fixes re-run only the
affected stages. Bump `version` when a stage's logic changes.

| Correction kind | Consumed by |
|---|---|
| `calibration_points` | calib |
| `split`, `merge` | link |
| `team` | team |
| `assign` | identity |
| `event` | events |

## Decisions carried over from the reviewed system (plan §6)

Gates in pitch metres (calibrate before tracking) · weak boxes in a second pass and new IDs only ≥ 0.6 · ball box
inflated for matching only · global assignment for linking (no greedy stitching) · OSNet + jersey OCR + squad slots
(team constraint blocks cross-team swaps) · one ball path via DP · homography per keyframe + smoothing (not a
similarity model) · calibration checked against painted lines, one review item per failed segment · label-free QA ·
event-based metrics, no player ratings · queue + workers + web, not one script.

## Model choices (pretrained, no training in the MVP)

roboflow/sports YOLOv8 detectors (players/GK/referee/ball, ball-specific) and pitch keypoints; SigLIP for team
embeddings (colour histogram fallback); OSNet (torchreid) for ReID; PARSeq for jersey numbers; Claude
(`claude-opus-5-5`, adaptive thinking, server-side refusal fallback) for the analyst. Fine-tuning only when error
analysis on club footage shows a gap (plan §4 "upgrade later").
