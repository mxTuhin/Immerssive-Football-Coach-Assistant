"""Stage 8: one ball path per match by dynamic programming over all ball candidates.

Nodes are ball candidates (top-K per frame, low detection threshold). An edge joins candidates up to
``max_skip`` frames apart when the implied speed is feasible (<= 40 m/s). Reward = detection evidence minus
motion and skip penalties; a long "jump" edge from the best path so far lets the path re-acquire the ball
after long occlusions. Backtracking gives exactly one ball position per frame (never several guesses).
Gaps up to 2 s are interpolated and flagged ``estimated``; implied speeds above 30 m/s flag ``in_air``
(a 2D projection of an aerial ball moves too fast/irregularly on the ground plane).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_pipeline.config import BallConfig


def best_path(cands: pd.DataFrame, fps: float, cfg: BallConfig) -> pd.DataFrame:
    """cands: frame, px, py, score. Returns the chosen candidate per frame (subset of rows)."""
    c = cands.dropna(subset=["px", "py"]).sort_values(["frame", "score"], ascending=[True, False])
    c = c.groupby("frame").head(cfg.max_candidates).reset_index(drop=True)
    if c.empty:
        return c
    frames = c["frame"].to_numpy()
    xy = c[["px", "py"]].to_numpy(float)
    reward = np.log(np.clip(c["score"].to_numpy(float), 1e-3, 1.0)) + 2.0   # > 0 for score > ~0.14
    n = len(c)
    best = np.full(n, -np.inf)
    back = np.full(n, -1)
    max_skip = max(1, int(cfg.max_skip_s * fps))
    dt = 1.0 / fps
    starts = np.searchsorted(frames, frames, side="left")
    # running global best (for long jumps): best value & index among nodes with frame < f - max_skip
    order_idx = np.arange(n)
    glob_val, glob_idx, gptr = -np.inf, -1, 0
    for i in range(n):
        f = frames[i]
        while gptr < n and frames[gptr] < f - max_skip:
            if best[gptr] > glob_val:
                glob_val, glob_idx = best[gptr], gptr
            gptr += 1
        lo = np.searchsorted(frames, f - max_skip, side="left")
        hi = starts[i]
        cand_best, cand_back = reward[i] - 1.0, -1   # start a new path (small start penalty)
        if glob_idx >= 0 and glob_val - cfg.jump_penalty + reward[i] > cand_best:
            cand_best, cand_back = glob_val - cfg.jump_penalty + reward[i], glob_idx
        if hi > lo:
            j = order_idx[lo:hi]
            gap = (f - frames[j]).astype(float)
            d = np.linalg.norm(xy[j] - xy[i], axis=1)
            speed = d / (gap * dt)
            ok = speed <= cfg.max_speed
            if ok.any():
                score = best[j] + reward[i] - 0.02 * (d ** 2) / gap - 0.3 * (gap - 1)
                score[~ok] = -np.inf
                k = int(np.argmax(score))
                if score[k] > cand_best:
                    cand_best, cand_back = score[k], int(j[k])
        best[i], back[i] = cand_best, cand_back
    # backtrack from the best node overall
    i = int(np.argmax(best))
    path = []
    while i >= 0:
        path.append(i)
        i = back[i]
    path = path[::-1]
    return c.iloc[path].reset_index(drop=True)


def ball_track(cands: pd.DataFrame, fps: float, cfg: BallConfig, frame_range: tuple[int, int]) -> pd.DataFrame:
    """Dense per-frame ball table: frame, x, y, confidence, estimated, in_air."""
    path = best_path(cands, fps, cfg)
    f0, f1 = frame_range
    frames = np.arange(f0, f1 + 1)
    out = pd.DataFrame({"frame": frames})
    if path.empty:
        out[["x", "y", "confidence"]] = np.nan
        out["estimated"] = True
        out["in_air"] = False
        return out
    p = path.drop_duplicates("frame").set_index("frame")
    x = p["px"].reindex(frames).to_numpy()
    y = p["py"].reindex(frames).to_numpy()
    conf = p["score"].reindex(frames).to_numpy()
    obs = np.isfinite(x)
    # interpolate short gaps only
    max_gap = int(cfg.interp_max_gap_s * fps)
    idx = np.arange(len(frames))
    if obs.sum() >= 2:
        xi = np.interp(idx, idx[obs], x[obs])
        yi = np.interp(idx, idx[obs], y[obs])
        last = np.maximum.accumulate(np.where(obs, idx, -10 ** 9))
        nxt = np.flip(np.minimum.accumulate(np.flip(np.where(obs, idx, 10 ** 9))))
        fill = ~obs & (nxt - last - 1 <= max_gap) & (last >= 0) & (nxt < 10 ** 9)
        x = np.where(fill, xi, x)
        y = np.where(fill, yi, y)
        conf = np.where(fill, 0.3, conf)
    speed = np.r_[0, np.hypot(np.diff(x), np.diff(y)) * fps]
    out["x"], out["y"], out["confidence"] = x, y, conf
    out["estimated"] = ~obs
    out["in_air"] = np.nan_to_num(speed) > cfg.in_air_speed
    return out
