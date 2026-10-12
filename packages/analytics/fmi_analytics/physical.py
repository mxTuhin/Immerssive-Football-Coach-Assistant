"""Physical metrics per player: distance, speed zones, top speed, sprints, accelerations, heatmaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign

# m/s thresholds
ZONES = [("walk", 0.0, 2.0), ("jog", 2.0, 4.0), ("run", 4.0, 5.5), ("hsr", 5.5, 7.0), ("sprint", 7.0, 99.0)]
SPRINT_SPEED = 7.0      # 25.2 km/h
SPRINT_MIN_S = 1.0
ACCEL_THRESHOLD = 3.0   # m/s^2


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def player_physical(tracking: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
    dt = 1.0 / fps
    rows = []
    players = tracking[tracking["team"].isin(["home", "away"])]
    for pid, g in players.groupby("id", sort=False):
        g = g.sort_values("frame")
        speed = g["speed"].to_numpy(dtype=float)
        speed = np.nan_to_num(speed, nan=0.0)
        frames = g["frame"].to_numpy()
        contiguous = np.diff(frames, prepend=frames[0] - 1) == 1
        step = speed * dt * contiguous
        rec = {"player_id": pid, "team": g["team"].iloc[0], "minutes": len(g) * dt / 60.0,
               "distance_m": float(step.sum()), "estimated_fraction": float(g["estimated"].mean()),
               "top_speed_kmh": float(np.percentile(speed, 99.5) * 3.6) if len(speed) else 0.0}
        for name, lo, hi in ZONES:
            m = (speed >= lo) & (speed < hi)
            rec[f"dist_{name}_m"] = float(step[m].sum())
        sprints = [(a, b) for a, b in _runs(speed >= SPRINT_SPEED) if (b - a) * dt >= SPRINT_MIN_S]
        rec["sprints"] = len(sprints)
        acc = np.gradient(speed, dt) if len(speed) > 1 else np.zeros_like(speed)
        rec["accelerations"] = len([r for r in _runs(acc >= ACCEL_THRESHOLD) if (r[1] - r[0]) * dt >= 0.5])
        rec["decelerations"] = len([r for r in _runs(acc <= -ACCEL_THRESHOLD) if (r[1] - r[0]) * dt >= 0.5])
        rec["distance_per_90_m"] = rec["distance_m"] / max(rec["minutes"], 1e-6) * 90.0
        rows.append(rec)
    return pd.DataFrame(rows)


def sprint_list(tracking: pd.DataFrame, fps: float = 10.0) -> pd.DataFrame:
    dt = 1.0 / fps
    rows = []
    for pid, g in tracking[tracking["team"].isin(["home", "away"])].groupby("id", sort=False):
        g = g.sort_values("frame")
        sp = np.nan_to_num(g["speed"].to_numpy(dtype=float))
        for a, b in _runs(sp >= SPRINT_SPEED):
            if (b - a) * dt < SPRINT_MIN_S:
                continue
            seg = g.iloc[a:b]
            rows.append({"player_id": pid, "team": seg["team"].iloc[0], "period": int(seg["period"].iloc[0]),
                         "t_start": float(seg["t"].iloc[0]), "t_end": float(seg["t"].iloc[-1]),
                         "max_speed_kmh": float(sp[a:b].max() * 3.6),
                         "distance_m": float(sp[a:b].sum() * dt),
                         "start_x": float(seg["x"].iloc[0]), "start_y": float(seg["y"].iloc[0]),
                         "end_x": float(seg["x"].iloc[-1]), "end_y": float(seg["y"].iloc[-1])})
    return pd.DataFrame(rows)


def heatmap(tracking: pd.DataFrame, player_id: str, meta, bins=(21, 14), normalize_attack: bool = True) -> dict:
    """Occupancy grid (fraction of time) for one player; x normalised so the team attacks +x."""
    g = tracking[tracking["id"] == player_id]
    if g.empty:
        return {"bins": list(bins), "grid": []}
    team = g["team"].iloc[0]
    sign = g["period"].map(lambda p: attack_sign(meta, int(p), team)).to_numpy() if normalize_attack else 1
    x, y = g["x"].to_numpy() * sign, g["y"].to_numpy() * sign
    L, W = meta.pitch_length, meta.pitch_width
    H, _, _ = np.histogram2d(x, y, bins=bins, range=[[-L / 2, L / 2], [-W / 2, W / 2]])
    H = H / max(H.sum(), 1)
    return {"bins": list(bins), "x_range": [-L / 2, L / 2], "y_range": [-W / 2, W / 2], "grid": H.round(5).tolist()}
