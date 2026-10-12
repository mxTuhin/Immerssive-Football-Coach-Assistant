"""Stage 10: facing direction (``body_yaw``) from tracking only (build plan section 4.1).

Rules per player per frame:
    moving > 2 m/s          -> direction of movement                  (high confidence)
    moving 0.5 .. 2 m/s     -> blend of movement and ball direction   (medium)
    slow / standing         -> direction to the ball                  (low-medium; GK medium)
    ball not tracked        -> hold last estimate, decay toward the team's attacking direction (low)

Smoothing: a circular Kalman filter per player (random-walk angle, wrapped innovation, measurement noise
from the rule confidence), run forward and backward and fused, with the turn rate capped (720 deg/s).
Known limit: backpedalling and side steps read as facing the wrong way (fixed later by body keypoints).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign, goalkeepers, to_wide
from fmi_schema.geom import wrap_deg

SOURCES = np.array(["none", "motion", "blend", "ball", "hold"], dtype=object)


@dataclass
class FacingConfig:
    fps: float = 10.0
    fast: float = 2.0          # m/s
    slow: float = 0.5          # m/s
    max_turn_dps: float = 720.0
    hold_decay_s: float = 3.0  # time constant for decaying toward attacking direction
    q_dps: float = 120.0       # process noise, deg per sqrt(s)


def _rule_measurements(vx, vy, speed, bx, by, px, py, is_gk, attack_deg, cfg: FacingConfig):
    """Return (yaw_meas, conf, source_idx) matrices; NaN yaw where no measurement."""
    T, N = vx.shape
    motion = np.degrees(np.arctan2(vy, vx))
    ball_ok = np.isfinite(bx)[:, None] & np.isfinite(px)
    dxb, dyb = bx[:, None] - px, by[:, None] - py
    dist_b = np.hypot(dxb, dyb)
    to_ball = np.degrees(np.arctan2(dyb, dxb))
    ball_ok &= dist_b > 0.3

    yaw = np.full((T, N), np.nan)
    conf = np.zeros((T, N))
    src = np.zeros((T, N), dtype=int)

    fast = speed > cfg.fast
    yaw[fast], conf[fast], src[fast] = motion[fast], 0.9, 1

    mid = (speed > cfg.slow) & ~fast
    w = np.clip((speed - cfg.slow) / (cfg.fast - cfg.slow), 0, 1)
    # circular blend of motion and ball directions
    bx_u = np.where(ball_ok, np.cos(np.radians(to_ball)), 0.0)
    by_u = np.where(ball_ok, np.sin(np.radians(to_ball)), 0.0)
    mx_u, my_u = np.cos(np.radians(motion)), np.sin(np.radians(motion))
    blend = np.degrees(np.arctan2(w * my_u + (1 - w) * by_u, w * mx_u + (1 - w) * bx_u))
    m_sel = mid & ball_ok
    yaw[m_sel], conf[m_sel], src[m_sel] = blend[m_sel], 0.6, 2
    m_noball = mid & ~ball_ok
    yaw[m_noball], conf[m_noball], src[m_noball] = motion[m_noball], 0.45, 1

    slow = ~(speed > cfg.slow) & np.isfinite(px)
    s_sel = slow & ball_ok
    gk_conf = np.where(is_gk[None, :], 0.6, 0.45)
    yaw[s_sel] = to_ball[s_sel]
    conf[s_sel] = np.broadcast_to(gk_conf, (T, N))[s_sel]
    src[s_sel] = 3
    # slow without ball: no measurement -> hold (handled in the filter)
    return yaw, conf, src, attack_deg


def _circular_filter(meas, conf, present, prior_deg, cfg: FacingConfig, reverse=False):
    """Random-walk circular Kalman over time for all players at once. Returns (yaw, var)."""
    T, N = meas.shape
    dt = 1.0 / cfg.fps
    q = (cfg.q_dps ** 2) * dt
    max_step = cfg.max_turn_dps * dt
    order = range(T - 1, -1, -1) if reverse else range(T)
    yaw = np.full(N, np.nan)
    var = np.full(N, 1e4)
    out_y = np.full((T, N), np.nan)
    out_v = np.full((T, N), np.nan)
    alpha = 1 - np.exp(-dt / cfg.hold_decay_s)
    for t in order:
        m = meas[t]
        has = np.isfinite(m)
        # initialise on first measurement (or attacking direction when present without one)
        init = ~np.isfinite(yaw) & present[t]
        yaw = np.where(init & has, m, np.where(init, prior_deg[t], yaw))
        var = np.where(init, np.where(has, 900.0, 4e3), var)
        var = var + q
        # measurement noise from confidence: conf 0.9 -> ~20 deg, 0.45 -> ~60 deg
        r = (10 + 110 * (1 - np.clip(conf[t], 0, 1))) ** 2
        k = np.where(has, var / (var + r), 0.0)
        innov = np.where(has, wrap_deg(m - yaw), 0.0)
        # no measurement: decay toward attacking direction
        decay = np.where(~has & np.isfinite(yaw), alpha * wrap_deg(prior_deg[t] - yaw), 0.0)
        step = np.clip(k * innov + decay, -max_step, max_step)
        yaw = np.where(present[t], wrap_deg(yaw + step), np.nan)
        var = np.where(has, (1 - k) * var, var)
        out_y[t], out_v[t] = yaw, var
    return out_y, out_v


def _rate_limit(yaw: np.ndarray, present: np.ndarray, max_step: float) -> np.ndarray:
    """Final pass: no frame-to-frame turn above ``max_step`` degrees (fusing two filters can exceed it)."""
    out = yaw.copy()
    for t in range(1, len(out)):
        prev, cur = out[t - 1], out[t]
        ok = present[t] & present[t - 1] & np.isfinite(prev) & np.isfinite(cur)
        step = np.clip(wrap_deg(cur - prev), -max_step, max_step)
        out[t] = np.where(ok, wrap_deg(prev + step), cur)
    return out


def compute_facing(tracking: pd.DataFrame, pkg_meta, gk_ids: set[str] | None = None,
                   cfg: FacingConfig | None = None) -> pd.DataFrame:
    """Add ``body_yaw``, ``yaw_conf``, ``yaw_source`` for players. Needs smoothed vx, vy, speed."""
    cfg = cfg or FacingConfig()
    gk_ids = gk_ids or set()
    tracking = tracking.copy()
    for col, default in (("body_yaw", np.nan), ("yaw_conf", np.nan), ("yaw_source", "none")):
        if col not in tracking:
            tracking[col] = default
    tracking["yaw_source"] = tracking["yaw_source"].astype(object)
    results = []
    for period, df in tracking.groupby("period", sort=True):
        players = df[df["team"].isin(["home", "away"])]
        ball = df[df["id"] == "ball"]
        if players.empty:
            continue
        frames, ids, w = to_wide(players, values=("x", "y", "vx", "vy", "speed"))
        team = players.drop_duplicates("id").set_index("id")["team"].reindex(ids).to_numpy()
        bx = np.full(len(frames), np.nan)
        by = np.full(len(frames), np.nan)
        if not ball.empty:
            b = ball.set_index("frame")
            idx = b.index.intersection(frames)
            bx[(idx - frames[0]).astype(int)] = b.loc[idx, "x"].to_numpy()
            by[(idx - frames[0]).astype(int)] = b.loc[idx, "y"].to_numpy()
        is_gk = np.array([oid in gk_ids for oid in ids])
        attack = np.array([0.0 if attack_sign(pkg_meta, int(period), t) >= 0 else 180.0 for t in team])
        attack_deg = np.broadcast_to(attack, (len(frames), len(ids)))
        speed = np.nan_to_num(w["speed"], nan=0.0)
        yaw_m, conf, src, _ = _rule_measurements(w["vx"], w["vy"], speed, bx, by, w["x"], w["y"], is_gk,
                                                 attack_deg, cfg)
        present = np.isfinite(w["x"])
        fy, fv = _circular_filter(yaw_m, conf, present, attack_deg, cfg)
        ry, rv = _circular_filter(yaw_m, conf, present, attack_deg, cfg, reverse=True)
        # fuse forward/backward estimates (inverse-variance weighted circular mean)
        wf, wr = 1 / fv, 1 / rv
        fused = np.degrees(np.arctan2(wf * np.sin(np.radians(fy)) + wr * np.sin(np.radians(ry)),
                                      wf * np.cos(np.radians(fy)) + wr * np.cos(np.radians(ry))))
        fused_var = 1 / (wf + wr)
        fused = _rate_limit(fused, present, cfg.max_turn_dps / cfg.fps)
        out_conf = np.where(np.isfinite(yaw_m), conf, np.clip(0.35 - fused_var / 4e4, 0.1, 0.35))
        source = np.where(np.isfinite(yaw_m), src, 4)
        r, c = np.nonzero(present)
        results.append(pd.DataFrame({
            "frame": frames[r], "id": np.asarray(ids, dtype=object)[c], "_yaw": fused[r, c],
            "_conf": out_conf[r, c], "_src": SOURCES[source[r, c]],
        }))
    if not results:
        return tracking
    res = pd.concat(results, ignore_index=True)
    merged = tracking.merge(res, on=["frame", "id"], how="left")
    has = merged["_yaw"].notna()
    merged.loc[has, "body_yaw"] = merged.loc[has, "_yaw"]
    merged.loc[has, "yaw_conf"] = merged.loc[has, "_conf"]
    merged.loc[has, "yaw_source"] = merged.loc[has, "_src"]
    return merged.drop(columns=["_yaw", "_conf", "_src"])


def facing_for_package(pkg, cfg: FacingConfig | None = None) -> pd.DataFrame:
    gks = set(goalkeepers(pkg).values())
    return compute_facing(pkg.tracking, pkg.meta, gks, cfg)


def facing_accuracy(pred_deg, true_deg, speed=None, tol: float = 45.0, moving: float = 2.0) -> dict:
    """Evaluation helper (build plan section 8): % within +-tol for moving and slow players, median error."""
    err = np.abs(wrap_deg(np.asarray(pred_deg, float) - np.asarray(true_deg, float)))
    out = {"n": int(np.isfinite(err).sum()), "within_tol": float(np.nanmean(err <= tol)) if err.size else np.nan,
           "median_err_deg": float(np.nanmedian(err)) if err.size else np.nan}
    if speed is not None:
        speed = np.asarray(speed, float)
        for name, mask in (("moving", speed > moving), ("slow", speed <= moving)):
            e = err[mask]
            out[f"{name}_n"] = int(mask.sum())
            out[f"{name}_within_tol"] = float(np.nanmean(e <= tol)) if e.size else np.nan
            out[f"{name}_median_err_deg"] = float(np.nanmedian(e)) if e.size else np.nan
    return out
