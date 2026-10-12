"""Scene packages for the 2D minimap, the three.js viewer and the Quest app.

A scene is a time window of the match with everything a renderer needs: players (team, number, name, kit
colour) with per-frame x, y, body_yaw and speed, plus the ball path. Coordinates are centred metres; the
renderer maps x -> world X, y -> world -Z (Unity/three.js are y-up).

Format (``scene.v1``)::

    {"format": "scene.v1", "match_id", "period", "t_start", "t_end", "fps", "pitch": {length, width},
     "teams": {"home": {name, color}, "away": {...}},
     "frames": [t0, t1, ...],
     "players": [{"id", "team", "number", "name", "x": [...], "y": [...], "yaw": [...], "speed": [...],
                  "yaw_conf": [...], "estimated": [...]}],
     "ball": {"x": [...], "y": [...], "z": [...]},
     "events": [...]}
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_schema.io import to_jsonable


def build_scene(pkg, period: int, t_start: float, t_end: float, fps: float | None = None,
                label: str | None = None) -> dict:
    fps = fps or pkg.meta.fps
    stride = max(1, int(round(pkg.meta.fps / fps)))
    tr = pkg.tracking
    win = tr[(tr["period"] == period) & (tr["t"] >= t_start) & (tr["t"] <= t_end)]
    frames = np.sort(win["frame"].unique())[::stride]
    win = win[win["frame"].isin(frames)]
    fidx = {f: i for i, f in enumerate(frames)}
    roster = pkg.roster.set_index("player_id") if not pkg.roster.empty else pd.DataFrame()

    def series(g, col, nd=2):
        arr = np.full(len(frames), np.nan)
        arr[[fidx[f] for f in g["frame"]]] = g[col].to_numpy(dtype=float)
        return [None if not np.isfinite(v) else round(float(v), nd) for v in arr]

    players = []
    for pid, g in win[win["team"].isin(["home", "away", "referee"])].groupby("id", sort=False):
        info = roster.loc[pid] if pid in roster.index else None
        est = np.zeros(len(frames), dtype=bool)
        est[[fidx[f] for f in g["frame"]]] = g["estimated"].fillna(False).to_numpy(dtype=bool)
        players.append({
            "id": str(pid), "team": g["team"].iloc[0],
            "number": int(info["number"]) if info is not None and pd.notna(info["number"]) else None,
            "name": str(info["name"]) if info is not None else str(pid),
            "x": series(g, "x"), "y": series(g, "y"), "yaw": series(g, "body_yaw", 1),
            "yaw_conf": series(g, "yaw_conf"), "speed": series(g, "speed"),
            "estimated": est.tolist(),
        })
    ball = win[win["id"] == "ball"]
    ev = pkg.events
    ev = ev[(ev["period"] == period) & (ev["t"] >= t_start) & (ev["t"] <= t_end)] if not ev.empty else ev
    scene = {
        "format": "scene.v1", "match_id": pkg.match_id, "label": label, "period": period,
        "t_start": t_start, "t_end": t_end, "fps": pkg.meta.fps / stride,
        "pitch": {"length": pkg.meta.pitch_length, "width": pkg.meta.pitch_width},
        "teams": {s: {"name": pkg.meta.team_info(s).name, "color": pkg.meta.team_info(s).jersey_color}
                  for s in ("home", "away")},
        "frames": [round(float(t), 2) for t in win.drop_duplicates("frame").sort_values("frame")["t"]],
        "players": players,
        "ball": {"x": series(ball, "x"), "y": series(ball, "y"), "z": series(ball, "z")} if not ball.empty else None,
        "events": ev.to_dict(orient="records") if not ev.empty else [],
    }
    return to_jsonable(scene)


def frame_snapshot(pkg, period: int, t: float) -> dict:
    """Single frame (for the minimap / freeze frames)."""
    frame = pkg.frame_at(period, t)
    tr = pkg.tracking[pkg.tracking["frame"] == frame]
    cols = ["id", "team", "x", "y", "body_yaw", "yaw_conf", "speed", "estimated", "confidence"]
    return to_jsonable({"match_id": pkg.match_id, "period": period, "t": t, "frame": frame,
                        "objects": tr[cols].to_dict(orient="records")})
