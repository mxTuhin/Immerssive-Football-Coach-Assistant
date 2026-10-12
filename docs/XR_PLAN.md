# 3D / XR plan (phase 2: pending)

Status: **pending by decision**. The backend already produces everything the viewers need (`scene.v1`, frame
snapshots, key moments, review queue). This document is the build plan for the two clients so work can start
from a clear contract.

## Contract the clients consume (already built)

| Endpoint | Returns |
|---|---|
| `GET /matches/{id}/summary` | meta (teams, kit colours, pitch size, periods), headline metrics, QA |
| `GET /matches/{id}/key-moments` | 8–12 moments with `period`, `t_start`, `t_end`, reason |
| `GET /matches/{id}/scenes/{moment_id}` | `scene.v1` for a key moment (pre-built in the package) |
| `GET /matches/{id}/tracking?period&t0&t1&fps` | `scene.v1` for any window ≤ 120 s |
| `GET /matches/{id}/frame?period&t` | one frame (freeze frames, minimap) |
| `GET /matches/{id}/review`, `POST /corrections`, `POST /rerun` | review loop |

`scene.v1` (see `packages/analytics/fmi_analytics/scenes.py`): `frames[]` (seconds), per player `x[]`, `y[]`,
`yaw[]` (degrees, 0 = +x, counter-clockwise), `yaw_conf[]`, `speed[]`, `estimated[]`, team, number, name; ball
`x[]`, `y[]`, `z[]`; events in the window; team names and kit colours.

**Coordinate mapping for y-up engines** (three.js and Unity): world `X = x`, world `Z = -y`, world `Y = 0` (ball:
`Y = z` when known). Rotation about the up axis: three.js `rotation.y = yaw_rad` for a model whose forward is
+X (right-handed, y-up: rotating +X by θ gives (cos θ, 0, −sin θ) = (cos yaw, 0, −sin yaw)); Unity
`Quaternion.Euler(0, 90 + yaw_deg, 0)` (Unity's forward is +Z and Euler-y turns +Z toward +X, so forward =
(sin(90+yaw), 0, cos(90+yaw)) = (cos yaw, 0, −sin yaw)). Write one unit test per client that a player with
yaw 0 faces world +X and yaw 90 faces world −Z.

## Web app (`apps/web`): React + three.js

Order (plan weeks 2–9):
1. **2D minimap** (week 2–3): canvas pitch, players as kit-coloured dots with a facing tick, timeline scrubber,
   play/pause at 10 fps with interpolation to 60 fps; data from `/tracking` windows (prefetch next window).
2. **3D viewer** (weeks 4–5): three.js scene with the pitch from `meta.pitch_length/width`, stylised capsule
   players coloured by kit, number on the back, rotated to `yaw`; ball sphere; camera modes: free orbit, tactical
   (top-down orthographic), broadcast, player view (camera at head height looking along `yaw`). Show
   low-confidence/estimated samples semi-transparent. Target 60 fps on a laptop (instanced meshes, one draw call
   per team).
3. **Library + report** (weeks 6–7): match list, job progress (poll `/jobs/{id}`), AI report with moment cards
   that open the clip window and the 3D scene.
4. **Review UI** (weeks 8–9): queue from `/review` sorted by priority; actions → `POST /corrections`
   (`merge`, `split`, `assign`, `team`, `calibration_points` with a 4-point picker over the video frame,
   `event` edits); show "will re-run: …" from the response; `POST /rerun`.

Stack suggestion: Vite + React + TypeScript, @react-three/fiber + drei, TanStack Query, a generated OpenAPI client
from `/openapi.json`.

## Quest 3S app (`apps/quest`): Unity

Order (plan weeks 10–11, after the web viewer):
1. Unity 6 LTS + OpenXR + Meta XR SDK; URP; target Quest 3S at 72/90 Hz.
2. Scene loader: download `scene.v1` JSON for the key moments (cache on device), build the pitch, spawn
   stylised players (kit colour, number), animate positions with interpolation, rotate to yaw.
3. Comfort (plan §2 #8): teleport locomotion only, fixed horizon, no camera motion the user did not cause,
   vignette on teleport, seated/standing modes.
4. Walk-in freeze frames: pause at a moment, teleport onto the pitch, look from a player's position along `yaw`.
5. Timeline: moment list, scrub bar, play/pause, speed 0.25–1×.
6. Later (plan §10): Animator with idle/walk/run/sprint blended from `speed`, turn-in-place from yaw rate;
   pose-driven bodies when body keypoints arrive.

Performance budget: ≤ 23 players + ball, GPU instancing, no real-time shadows on players, baked pitch lighting.

## Acceptance (from the plan's MVP definition of done)
* Web: smooth at 60 fps on a laptop; players face the right way (spot-check with the facing labels).
* Quest: comfortable playback of the key moments with teleport and fixed horizon.
