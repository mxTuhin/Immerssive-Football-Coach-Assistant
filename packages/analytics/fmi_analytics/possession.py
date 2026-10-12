"""Ball possession per frame with hysteresis.

Player control: the nearest player within ``control_radius`` of the ball (and ball low, when height is known)
becomes the candidate; the owner only changes after the candidate holds for ``min_frames``. Team possession
persists through loose-ball periods until the other team gains control.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmi_analytics.common import to_wide


@dataclass
class PossessionConfig:
    control_radius: float = 1.5     # m
    max_ball_height: float = 1.8    # m, when z is known
    min_frames: int = 2             # hysteresis for a player change
    team_min_frames: int = 3        # hysteresis for a team change
    fps: float = 10.0


def compute_possession(tracking: pd.DataFrame, cfg: PossessionConfig | None = None) -> pd.DataFrame:
    """Per-frame table: frame, period, t, ball_x, ball_y, owner, owner_team, team_in_possession, dist."""
    cfg = cfg or PossessionConfig()
    out = []
    for period, df in tracking.groupby("period", sort=True):
        ball = df[df["id"] == "ball"].set_index("frame").sort_index()
        players = df[df["team"].isin(["home", "away"])]
        if ball.empty or players.empty:
            continue
        frames, ids, w = to_wide(players, values=("x", "y"))
        team = players.drop_duplicates("id").set_index("id")["team"].reindex(ids).to_numpy()
        bx = ball["x"].reindex(frames).to_numpy()
        by = ball["y"].reindex(frames).to_numpy()
        bz = ball["z"].reindex(frames).to_numpy() if "z" in ball else np.full(len(frames), np.nan)
        d = np.hypot(w["x"] - bx[:, None], w["y"] - by[:, None])
        d = np.where(np.isfinite(d), d, np.inf)
        nearest = np.argmin(d, axis=1)
        dmin = d[np.arange(len(frames)), nearest]
        low = ~(np.nan_to_num(bz, nan=0.0) > cfg.max_ball_height)
        cand = np.where((dmin <= cfg.control_radius) & low & np.isfinite(bx), nearest, -1)

        owner = np.full(len(frames), -1)
        cur, run_c, run_n = -1, -1, 0
        for i, c in enumerate(cand):
            if c == run_c:
                run_n += 1
            else:
                run_c, run_n = c, 1
            if c >= 0 and c != cur and run_n >= cfg.min_frames:
                cur = c
            if c == -1 and cur >= 0 and d[i, cur] > 3.0:
                cur = -1  # ball clearly left the owner
            owner[i] = cur if (cur >= 0 and d[i, cur] <= 3.0) else -1

        tip = np.empty(len(frames), dtype=object)
        cur_t, run_t, run_tn = None, None, 0
        for i, o in enumerate(owner):
            t = team[o] if o >= 0 else None
            if t is not None and t == run_t:
                run_tn += 1
            else:
                run_t, run_tn = t, 1 if t is not None else 0
            if t is not None and t != cur_t and (cur_t is None or run_tn >= cfg.team_min_frames):
                cur_t = t
            tip[i] = cur_t
        tf = df.groupby("frame")["t"].first()
        t_clock = float(tf.iloc[0]) + (frames - int(tf.index[0])) / cfg.fps
        out.append(pd.DataFrame({
            "frame": frames, "period": period, "t": t_clock, "ball_x": bx, "ball_y": by,
            "owner": [ids[o] if o >= 0 else None for o in owner],
            "owner_team": [team[o] if o >= 0 else None for o in owner],
            "team_in_possession": tip, "dist": np.where(np.isfinite(dmin), dmin, np.nan),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["frame", "period", "t", "ball_x", "ball_y", "owner", "owner_team", "team_in_possession", "dist"])


def possession_share(poss: pd.DataFrame) -> dict[str, float]:
    tip = poss["team_in_possession"].dropna()
    if tip.empty:
        return {"home": float("nan"), "away": float("nan")}
    vc = tip.value_counts(normalize=True)
    return {"home": float(vc.get("home", 0.0)), "away": float(vc.get("away", 0.0))}
