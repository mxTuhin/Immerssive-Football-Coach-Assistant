---
name: web-viewer
description: PENDING (phase 2). Build the FMI web app in apps/web - React + three.js 2D minimap, 3D viewer with players rotated to body_yaw, match library, AI report and review UI - against the existing API and scene.v1 contract.
---

# Web viewer (phase 2 — not started)

Read `docs/XR_PLAN.md` first; the backend contract is done. Order: 2D minimap → 3D viewer → library/report →
review UI. Use the API (`/tracking`, `/scenes/{id}`, `/key-moments`, `/review`, `/corrections`, `/rerun`).

Must-haves:
- Coordinates: world X = x, world Z = −y; `rotation.y = yaw_rad` for a model facing +X. Unit-test yaw 0 → +X.
- Render `estimated` / low `yaw_conf` samples semi-transparent; never show flagged data as fact.
- 60 fps on a laptop: instanced meshes, interpolate 10 Hz data on the render clock.
- Generate the API client from `/openapi.json`; keep the API token out of the bundle (use a session/proxy).
