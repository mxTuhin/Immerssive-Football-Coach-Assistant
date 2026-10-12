# Football Match Intelligence: Build Plan

*v2.0 (final) · 12 Oct 2026 · Companion to `01-technical-integration-plan.md` and `02-business-plan.md`*
*We write our own code. Open repos, models and papers are references or pretrained starting points. This plan supersedes the milestone table in the Technical Plan for day-to-day execution.*

---

## 1. Principles

1. **Full pipeline from the start.** Every MVP stage works end to end: video → detection → tracking → identity → pitch metres → ball → facing direction → events → analytics → AI analyst → 3D → XR. Each stage starts with a working v1 and gets upgraded later.
2. **Pretrained first, train later.** The MVP runs on free pretrained football models. We fine-tune only when error analysis on real club footage shows a specific gap.
3. **Small data, measured.** Three test videos to start, one with labelled positions so accuracy is a number, not an impression.
4. **Quality is visible.** Every output carries confidence; wrong data is flagged or sent to review, never shown as fact.
5. **Two tracks in parallel.** The product (analytics, AI, 3D/XR, web) is built on open tracking data while the vision pipeline matures, then the two are joined.
6. **Complex features are parked, not lost.** Body keypoints, head direction, posture and engine animation are listed in §10 with the point at which we pick them up.

---

## 2. MVP definition of done

A coach uploads a full-pitch match (or a 10-minute segment for the demo) and gets, **without anyone touching the command line**:

| # | Output | Quality bar |
|---|---|---|
| 1 | Every player, referee and the ball tracked; team colours; positions in pitch metres | Median position error ≤ 1.5 m on the labelled test video |
| 2 | Stable player identities, named from the squad list | ≥ 85% of player-minutes correct automatically; ≥ 95% after ≤ 20 min review |
| 3 | **Facing direction** of every player on the pitch (`body_yaw`) | Within ±45° on ≥ 70% of hand-labelled samples of moving players (> 2 m/s); slow players reported separately; every value carries a confidence |
| 4 | Physical and team metrics: distance, speed, sprints, heatmaps, team shape, possession | No impossible values (speed ≤ 36 km/h, acceleration ≤ 8 m/s²) |
| 5 | Events: passes, shots, recoveries, set pieces | Pass/shot F1 ≥ 0.7 at ±1 s on labelled data |
| 6 | AI match report: 8–12 key moments, each with a clip and a 3D scene; questions in plain language | Every number comes from a tool call; coach rubric ≥ 4/5 on usefulness |
| 7 | Web 2D minimap and 3D viewer: timeline, free camera, tactical view, player view; players shown facing the right way | Smooth at 60 fps on a laptop |
| 8 | Quest 3S playback of the key moments | Comfortable: teleport movement, fixed horizon |
| 9 | Review screen to fix IDs, teams and calibration | Fixes re-run only the affected stages |

---

## 3. Start data: 3 videos + open tracking data

| # | Source | What it is | Why | Size to download |
|---|---|---|---|---|
| V1 | `roboflow/sports` sample clip (`examples/soccer/setup.sh`) | ~30 s Bundesliga broadcast clip | Smoke test; see the reference flow work in one afternoon | One clip + 3 model files |
| V2 | **SoccerTrack v2**: one half-match only (Hugging Face; accept conditions, download one file) | 4K panoramic amateur university match, ~45 min. Labels: pitch positions in metres, IDs, jersey numbers, roles, team, ball-action events | The one video with **ground truth**: measure position error, ID switches and event F1 | One half-match file, not the 145 GB set |
| V3 | A public full-match upload by a lower-tier club | Real follow-cam or low-angle footage | Shows how the pipeline behaves on what customers actually have | One video |
| D1 | **SkillCorner open data** (GitHub) | 10 A-League 2024/25 matches: player + ball tracking in metres at 10 fps, events, phases of play. No video | Build analytics, facing logic, AI analyst and 3D viewer **from week 1** | A few hundred MB |
| D2 | Metrica sample data (GitHub) | 3 anonymized matches: tracking + synced events. No video | Second format to test the data model and analytics | Small |

**Later, when needed:** a SoccerNet-GSR split (broadcast benchmark), more SoccerTrack v2 halves (when fine-tuning), SkillCorner body-pose sample (when body keypoints are picked up), then club-licensed footage from design partners.

---

## 4. Pipeline: MVP v1 and upgrade path for every stage

| # | Stage | MVP v1 (pretrained, no training) | Upgrade later (trigger) | Output |
|---|---|---|---|---|
| 1 | **Ingest** | Upload + URL import; ffmpeg to 25 fps proxy; shot-cut detection; squad CSV | Resumable chunks; direct Pixellot/Veo export import | Video assets, segments |
| 2 | **Detection** | Roboflow pretrained football detector (player, GK, referee, ball) at 1280 px; **tiling** for 4K panoramic | Fine-tune RF-DETR/YOLO on club footage (when recall drops on low angles, rain, floodlights) | Boxes + scores |
| 3 | **Calibration** | Roboflow pretrained 32-point pitch-keypoint model → RANSAC homography. **Line-alignment check** against painted lines; static camera: solve once and verify; moving camera: keyframes + smoothing. Manual 4-point fallback in UI | PnLCalib-style refinement; own keypoint fine-tune (when line score < 0.7 on > 10% of frames) | Homography per frame + quality score |
| 4 | **Online tracking** | Our BoT-SORT-style tracker: cost = IoU + distance in **pitch metres** + appearance; weak boxes into a second pass; new IDs only above ~0.6 | Mask-assisted matching (McByte/SAM 2) for crowded set pieces | Tracklets |
| 5 | **Offline linking** | Tracklet graph over the whole match; global assignment; football rules: ≤ 11 per team on the pitch, feasible speed ≤ 10 m/s across gaps, no overlap | Learned link scores from correction logs | Long tracks |
| 6 | **Team & roles** | SigLIP embeddings + UMAP + KMeans (as in `roboflow/sports`); GK by position relative to own team; referees by outlier colour | Per-match fine-tuned classifier | Team, role per track |
| 7 | **Identity** | **Pretrained OSNet ReID** for linking; **pretrained PARSeq** jersey reading with voting across each track; squad list → fixed slots 1–11 per team; review screen for the rest | Fine-tune ReID and OCR on our crops (when review time > 15 min) | Player identity per track |
| 8 | **Ball** | Roboflow pretrained ball model + tiling, low threshold → **one ball path** picked by dynamic programming; in-air flag; gap interpolation | Temporal ball model (TrackNet-style) | Ball x, y + confidence |
| 9 | **Projection & smoothing** | Foot point → metres; RTS smoother; speed/acceleration caps; interpolated points flagged `estimated` | — | Clean tracking table |
| 10 | **Facing direction** (details in §4.1) | From tracking only: movement direction + ball direction, smoothed per player | Body keypoints for side steps and backpedalling (§10) | `body_yaw`, confidence, source |
| 11 | **Events** | Rules on tracking: possession changes → passes, recoveries; shots from ball speed + direction to goal; set pieces from restarts | Learned action spotting (SoccerNet-style) | Event list (SPADL-compatible) |
| 12 | **Analytics** | Physical, shape, line heights, compactness, possession with hysteresis, phases of play, pressing, xT | Ghost player v2 (learned) | Metrics with reasons |
| 13 | **AI analyst** | Claude with tools over the match data: report, key moments, plain-language search, session plan | Opponent scouting across matches | Report, moments, answers |
| 14 | **3D / XR** | Web three.js viewer; Quest 3S Unity app; stylized kit-coloured players placed from tracking and **rotated to `body_yaw`** | Engine locomotion animation (idle/walk/run from speed) and pose-driven bodies (§10) | Scene package |
| 15 | **Review UI** | Web queue sorted by low confidence: merge/split, assign player, fix team, add 4 pitch points | Corrections feed retraining | Corrected match + log |

### 4.1 Facing direction (stage 10)

**Goal:** know which way each player faces on the pitch, from data we already have. No body keypoints in the MVP.

**Inputs per player per frame:** smoothed velocity (from stage 9), ball position (stage 8), team attacking direction.

**Rule set:**

| Situation | Facing estimate | Confidence |
|---|---|---|
| Moving > 2 m/s | Direction of movement | High |
| Moving 0.5–2 m/s | Blend of movement and ball direction | Medium |
| Slow or standing (< 0.5 m/s) | Direction to the ball; at set pieces and restarts, toward play | Low–medium |
| Goalkeeper, slow | Toward the ball | Medium |
| Ball not tracked | Hold the last estimate; decay toward the team's attacking direction | Low |

**Smoothing:** a per-player angle filter (circular Kalman) removes jitter and caps turn speed (e.g. ≤ 720°/s), so avatars never snap around.

**Output:** `body_yaw` (degrees on the pitch), `yaw_conf`, `source` (motion / ball / hold).

**Known limit:** backpedalling and side steps read as facing the wrong way. This is the gap body keypoints close later (§10).

**Works on open data too:** the same logic runs on SkillCorner tracking from week 2, so the 3D viewer shows facing players before our own video pipeline is ready.

---

## 5. Architecture

```text
Web app (React + three.js) ── API (FastAPI) ── Postgres · Redis queue · R2 storage
        ▲                          │
        │                   GPU worker (Docker; cloud GPU or Colab/Kaggle for dev)
 Quest 3S (Unity)   ingest → calibrate → detect → track → link → team/ID → ball → project → facing → events
        ▲                          │
        └──── Match Package (Parquet + JSON) ──► analytics ──► AI analyst (Claude + tools)
```

**Stage contract (every module):**
- `run(match_id, inputs) -> artifacts`; idempotent; cached by input hash.
- Parquet/JSON in storage; no raw frames passed between stages.
- Every stage writes confidence + a QA record.
- Any stage can be re-run alone after a correction.
- Unit tests use fake models, so CI runs without a GPU.

**Match Package (core schema):**
- `meta`: competition, teams, pitch size, video tier
- `roster`: player, team, shirt number
- `calibration`: per-frame homography + score
- `tracking` (10 Hz): frame, t, id, x, y, vx, vy, body_yaw, yaw_conf, confidence, estimated
- `events`: t, type, team, player, start/end xy, outcome, source (auto/human)
- `qa`: per-stage metrics, review queue
- `derived`: metrics, scenes, AI outputs

It is kloppy-compatible, so SkillCorner and Metrica data load into the same schema. A `body` table (keypoints, head direction, posture) is reserved for §10.

---

## 6. Lessons taken from the reviewed system (`Mhmdwael77/ai-football-analytics`)

**Re-implemented in our code:**
- Weak detections go into the tracker's second pass, and new IDs need ~0.6 confidence. Their measurements: ID switches 12 → 3 and 28 → 8.
- The ball box is inflated for matching only, so a small fast ball still links between frames.
- Calibration is checked against painted lines it wasn't fitted to, and a human is asked once per failed segment, not per frame.
- Fixed squad slots per team; the team constraint blocks cross-team swaps.
- Label-free QA: mid-pitch ID births, impossible jumps, kit-colour changes within a track.
- Typed stage contracts, re-runnable stages, tests with fakes.

**Done differently:**

| Their approach | Problem | Ours |
|---|---|---|
| All gates in pixels | Changes with resolution and zoom | Gates in pitch metres; calibrate before tracking |
| Greedy stitching | One bad early link cascades | Global assignment with football rules |
| ReID off; HSV colour only | Teammate swaps unsolved | Pretrained OSNet + jersey OCR + squad slots |
| Several ball guesses at once | Downstream takes the first | One ball path per match via DP; in-air flag |
| Similarity camera-motion model | Wrong for pan/tilt/zoom | Homography between frames, or keyframe calibration + smoothing |
| Ratings from distance and speed | Coaches won't trust them | Event-based metrics; no player ratings in MVP |
| Single 1,800-line script; simulated web jobs | Not a product | Queue + workers + web end to end |

---

## 7. Two parallel tracks, 16 weeks

| Weeks | Vision track (V1–V3) | Product track (D1–D2, then live data) | Showcase at end of block |
|---|---|---|---|
| **1** | Run `roboflow/sports` RADAR on V1. Set up repo, stage contract, eval loader for V2 labels | Match Package schema; load SkillCorner + Metrica into it | Radar video of V1; SkillCorner match in our schema |
| **2–3** | Our v1 pipeline: detection (tiled) → calibration with line check → online tracker in metres → team clustering → projection + smoothing. Run on V1 and the first 10 min of V2 | Analytics v1 (physical, shape, possession) on SkillCorner; web 2D minimap; **facing logic (§4.1) on SkillCorner tracking** | First number: position error on V2 |
| **4–5** | Offline linking; ball path (DP); OSNet ReID; squad slots; label-free QA; **facing on our own tracking**. Full V2 half | Web 3D viewer (three.js): players placed and rotated to `body_yaw`; scene package; events v1 on SkillCorner | **Demo 1:** V2 half in 2D + 3D with metrics and player facing |
| **6–7** | Jersey OCR (PARSeq) + voting; events from our tracking; V3 lower-tier match; **hand-label 300 facing samples from V2/V3 and measure** | AI analyst v1: report + key moments + clip/scene links | AI report on our own tracked match |
| **8–9** | FastAPI + Redis queue + GPU worker end to end; storage; job status | Review UI: merge/split, assign, team fix, calibration points; partial re-run | Upload → result with no CLI |
| **10–11** | Moving-camera calibration (keyframes + smoothing) on V3; error analysis report | Quest 3S app: load scene package, walk-in freeze frames, timeline | **Demo 2 (MVP showcase):** web + XR on all 3 videos |
| **12–14** | First design-partner club footage; measure; fine-tune only where error analysis shows gaps | Plain-language search; session plan; phases of play, pressing, xT | Partner-club match fully processed |
| **15–16** | Hardening: < 1 h processing per match; cost per match < US$4 | Accounts, teams, billing hooks; PDF/CSV export | **Pilot-ready** |

---

## 8. Evaluation with small data

| What | How we measure |
|---|---|
| V2 (labelled) | Position error (m), HOTA/IDF1, ID switches per 10 min, % player-minutes correctly identified, ball within 2 m, event F1 |
| V1, V3 (no labels) | Label-free QA (births, impossible jumps, kit changes, calibration line score) + hand-check of 200 sampled frames |
| Facing (V2, V3) | 300 hand-labelled player crops in 8 directions: % within ±45° for moving players (target ≥ 70%) and for slow players (reported); median angle error |
| Every job | Plausibility gates: speed > 36 km/h, acceleration > 8 m/s², > 11 players per team on the pitch, players off the pitch for > 2 s |
| AI analyst | 30-question test set per match; coach rubric (correct, useful, actionable); fail on any number without a tool call |

Every change runs the V2 evaluation in CI and blocks a merge on regression.

---

## 9. Compute and infrastructure

| Item | MVP choice |
|---|---|
| Development GPU | Free Colab/Kaggle GPU for short clips; one cloud L4/A10 instance for full matches (startup credits) |
| Control plane | Existing VPS: Docker, Postgres, Redis |
| Storage | Cloudflare R2 for video, Match Packages, scenes |
| Queue | Celery + Redis |
| Target | < 3 h per full match at MVP; < 1 h at pilot; < US$4 per match |

---

## 10. Parked for after the MVP

| Feature | What it adds | Starting point | Pick up when |
|---|---|---|---|
| **Body keypoints** | Correct facing during backpedalling and side steps | Pretrained RTMPose via `rtmlib`, or ViTPose; shoulders + hips projected to the pitch (published method: median 27° error vs. sensors) | Facing accuracy on slow players is the top coach complaint |
| **Head direction** | Where the player is looking | Nose, eyes, ears from the same pose model | Body keypoints are in |
| **Rough posture** | Kicking, jumping, sliding, on the ground | Rules on keypoints + ball events | Body keypoints are in |
| **Scanning metric** | Head checks before receiving | Head direction + ball timeline | Head direction is measured reliable |
| **Engine animation** | Idle/walk/run/sprint blended from speed; turn-in-place | Unity animation controller + standard locomotion clips | After Demo 2, before pilots |
| **Pose-driven bodies** | Real limb movement in key moments | 3D pose lifting; SkillCorner body-pose data for testing | Coaches ask for it in XR reviews |

---

## 11. Repo layout

```text
fmi/
├─ apps/
│  ├─ api/            FastAPI: matches, jobs, uploads
│  ├─ web/            React + three.js: library, review UI, 2D/3D viewer, reports
│  └─ quest/          Unity: scene loader, XR UX
├─ workers/pipeline/  stages/{ingest,calib,detect,track,link,team,identity,ball,project,facing,events}
├─ packages/
│  ├─ schema/         Match Package + kloppy adapters (SkillCorner, Metrica)
│  ├─ analytics/      physical, shape, possession, phases, pressing, xT
│  └─ agent/          tools, prompts, eval set
├─ eval/              V2 label loader, metrics, label-free QA, facing labels, reports
└─ infra/             docker-compose, GPU worker image
```

---

## 12. Resources

**Start here**
- roboflow/sports (pretrained detector, ball and pitch-keypoint models; sample clips; team classifier): https://github.com/roboflow/sports
- supervision: https://github.com/roboflow/supervision
- trackers (BoT-SORT, ByteTrack, OC-SORT, McByte): https://github.com/roboflow/trackers

**Models used pretrained in the MVP**
- torchreid (OSNet ReID): https://github.com/KaiyangZhou/deep-person-reid
- PARSeq (jersey numbers): https://github.com/baudm/parseq
- SAHI (tiled detection): https://github.com/obss/sahi
- SigLIP (team embeddings): https://huggingface.co/docs/transformers/model_doc/siglip

**For upgrades**
- RF-DETR: https://github.com/roboflow/rf-detr
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- PnLCalib: https://github.com/mguti97/PnLCalib
- TrackNetV3: https://github.com/alenzenx/TrackNetV3
- sn-gamestate (SoccerNet baseline): https://github.com/SoccerNet/sn-gamestate
- SAM 2: https://github.com/facebookresearch/sam2

**Parked features (§10)**
- rtmlib (RTMPose via ONNX): https://github.com/Tau-J/rtmlib
- ViTPose in transformers: https://huggingface.co/docs/transformers/model_doc/vitpose
- SkillCorner body pose: https://huggingface.co/datasets/SkillCorner/opendata-bodypose

**Data model and analytics**
- kloppy: https://github.com/PySport/kloppy
- socceraction (SPADL, xT): https://github.com/ML-KULeuven/socceraction
- floodlight: https://github.com/floodlight-sports/floodlight

**Data**
- SoccerTrack v2: https://huggingface.co/datasets/atomscott/soccertrack-v2
- SkillCorner open data: https://github.com/SkillCorner/opendata
- Metrica sample data: https://github.com/metrica-sports/sample-data
- TeamTrack (later): https://www.kaggle.com/datasets/atomscott/teamtrack

**Papers**
- SoccerTrack v2: https://arxiv.org/abs/2508.01802
- SoccerNet-GSR: https://arxiv.org/abs/2404.11335
- BoT-SORT: https://arxiv.org/abs/2206.14651
- McByte: https://arxiv.org/abs/2506.01373
- Player orientation from pose + ball context: https://arxiv.org/abs/2003.00943

**Reviewed system:** https://github.com/Mhmdwael77/ai-football-analytics
