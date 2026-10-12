---
name: xr-quest
description: PENDING (phase 2, after the web viewer). Build the Quest 3S Unity app in apps/quest that plays FMI key moments from scene.v1 packages with comfortable XR (teleport, fixed horizon, freeze-frame walk-in, timeline).
---

# Quest 3S app (phase 2 — not started)

Plan: `docs/XR_PLAN.md` ("Quest 3S app"). Inputs: `GET /matches/{id}/key-moments` and `/scenes/{moment_id}`
(`scene.v1`).

Must-haves:
- Unity 6 LTS, URP, OpenXR + Meta XR SDK; 72/90 Hz on Quest 3S.
- Mapping: world X = x, Z = −y; `Quaternion.Euler(0, 90 + yaw_deg, 0)` (Unity forward +Z). Test yaw 0 → +X,
  yaw 90 → −Z.
- Comfort: teleport only, fixed horizon, no unrequested camera motion, vignette on teleport.
- Cache scene JSON on device; GPU instancing for 23 players; no real-time shadows on players.
- Later: Animator blend (idle/walk/run/sprint by speed), turn-in-place; pose-driven bodies when the `body` table
  exists (plan §10).
