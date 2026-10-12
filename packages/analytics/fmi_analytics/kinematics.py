"""Stage 9 core: constant-velocity Kalman filter + RTS smoother on pitch positions.

Runs vectorised over all objects of a period at once (state per object and axis: position, velocity).
Short gaps (<= ``max_gap_s``) are filled from the smoother and flagged ``estimated``; longer gaps split the
track and are left empty. Speeds above the physical cap are clipped and counted for QA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmi_analytics.common import MAX_BALL_SPEED, MAX_PLAYER_ACCEL, MAX_PLAYER_SPEED, to_wide


@dataclass
class SmoothConfig:
    fps: float = 10.0
    sigma_acc_player: float = 4.0   # m/s^2 process noise (white-noise acceleration)
    sigma_acc_ball: float = 25.0
    meas_std: float = 0.35          # m, observation noise for players
    meas_std_ball: float = 0.6
    max_gap_s: float = 2.0
    max_player_speed: float = MAX_PLAYER_SPEED
    max_ball_speed: float = MAX_BALL_SPEED


def rts_smooth(z: np.ndarray, dt: float, sigma_acc: np.ndarray | float, meas_std: np.ndarray | float,
               max_gap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth observations ``z`` of shape (T, N) with NaN for missing.

    Returns (pos, vel, valid) where ``valid`` marks frames within ``max_gap`` frames of an observation inside
    a segment (a gap longer than ``max_gap`` restarts the filter).
    """
    T, N = z.shape
    s2 = np.broadcast_to(np.asarray(sigma_acc, dtype=float) ** 2, (N,))
    R = np.broadcast_to(np.asarray(meas_std, dtype=float) ** 2, (N,))
    q00, q01, q11 = dt ** 4 / 4 * s2, dt ** 3 / 2 * s2, dt ** 2 * s2

    pf = np.full((T, N), np.nan); vf = np.full((T, N), np.nan)
    a_f = np.zeros((T, N)); b_f = np.zeros((T, N)); c_f = np.zeros((T, N))   # filtered cov
    a_p = np.ones((T, N)); b_p = np.zeros((T, N)); c_p = np.ones((T, N))     # predicted cov (t | t-1)
    pp = np.full((T, N), np.nan); vp = np.full((T, N), np.nan)
    restart = np.zeros((T, N), dtype=bool)

    p = np.full(N, np.nan); v = np.zeros(N)
    a = np.zeros(N); b = np.zeros(N); c = np.zeros(N)
    since = np.full(N, 10 ** 9)
    for t in range(T):
        obs = np.isfinite(z[t])
        active = np.isfinite(p)
        # predict
        p_pred = p + v * dt
        a_pr = a + 2 * dt * b + dt * dt * c + q00
        b_pr = b + dt * c + q01
        c_pr = c + q11
        # (re)start where an observation arrives after a long gap or for the first time
        start = obs & (~active | (since > max_gap))
        upd = obs & ~start
        # update
        S = a_pr + R
        k0 = np.where(upd, a_pr / S, 0.0)
        k1 = np.where(upd, b_pr / S, 0.0)
        innov = np.where(upd, z[t] - p_pred, 0.0)
        p_new = p_pred + k0 * innov
        v_new = v + k1 * innov
        a_new = (1 - k0) * a_pr
        b_new = (1 - k0) * b_pr
        c_new = c_pr - k1 * b_pr
        # starts
        p_new = np.where(start, z[t], p_new)
        v_new = np.where(start, 0.0, v_new)
        a_new = np.where(start, R, a_new)
        b_new = np.where(start, 0.0, b_new)
        c_new = np.where(start, 25.0, c_new)
        since = np.where(obs, 0, since + 1)
        # drop tracks that have been unobserved for too long
        dead = since > max_gap
        p_new = np.where(dead, np.nan, p_new)

        pp[t], vp[t] = p_pred, v
        a_p[t], b_p[t], c_p[t] = a_pr, b_pr, c_pr
        restart[t] = start
        p, v, a, b, c = p_new, v_new, a_new, b_new, c_new
        pf[t], vf[t] = p, v
        a_f[t], b_f[t], c_f[t] = a, b, c

    ps = pf.copy(); vs = vf.copy()
    for t in range(T - 2, -1, -1):
        ok = np.isfinite(pf[t]) & np.isfinite(ps[t + 1]) & ~restart[t + 1]
        if not ok.any():
            continue
        # C = P_f F^T Ppred^-1 with F = [[1, dt], [0, 1]]
        m00 = a_f[t] + dt * b_f[t]; m01 = b_f[t]
        m10 = b_f[t] + dt * c_f[t]; m11 = c_f[t]
        A, B, Cc = a_p[t + 1], b_p[t + 1], c_p[t + 1]
        det = A * Cc - B * B
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        i00, i01, i11 = Cc / det, -B / det, A / det
        C00 = m00 * i00 + m01 * i01; C01 = m00 * i01 + m01 * i11
        C10 = m10 * i00 + m11 * i01; C11 = m10 * i01 + m11 * i11
        dp = ps[t + 1] - pp[t + 1]
        dv = vs[t + 1] - vp[t + 1]
        ps[t] = np.where(ok, pf[t] + C00 * dp + C01 * dv, ps[t])
        vs[t] = np.where(ok, vf[t] + C10 * dp + C11 * dv, vs[t])

    # keep observed frames and frames inside gaps no longer than max_gap (interpolation, never extrapolation)
    obs = np.isfinite(z)
    idx = np.arange(T)[:, None]
    last = np.where(obs, idx, -10 ** 9)
    last = np.maximum.accumulate(last, axis=0)
    nxt = np.where(obs, idx, 10 ** 9)
    nxt = np.flip(np.minimum.accumulate(np.flip(nxt, 0), axis=0), 0)
    gap_len = nxt - last - 1
    valid = np.isfinite(ps) & (obs | (gap_len <= max_gap))
    ps[~valid] = np.nan
    vs[~valid] = np.nan
    return ps, vs, valid


def smooth_tracking(tracking: pd.DataFrame, cfg: SmoothConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Smooth every object of every period. Returns (tracking with x, y, vx, vy, speed, estimated, qa_metrics)."""
    cfg = cfg or SmoothConfig()
    dt = 1.0 / cfg.fps
    max_gap = int(round(cfg.max_gap_s * cfg.fps))
    out = []
    n_speed_capped = 0
    n_accel_flags = 0
    for period, df in tracking.groupby("period", sort=True):
        frames, ids, w = to_wide(df, values=("x", "y", "confidence") if "confidence" in df else ("x", "y"))
        meta_cols = df.drop_duplicates("id").set_index("id")
        is_ball = np.array([oid == "ball" for oid in ids])
        sig = np.where(is_ball, cfg.sigma_acc_ball, cfg.sigma_acc_player)
        meas = np.where(is_ball, cfg.meas_std_ball, cfg.meas_std)
        observed = np.isfinite(w["x"])
        xs, vxs, valid = rts_smooth(w["x"], dt, sig, meas, max_gap)
        ys, vys, valid_y = rts_smooth(w["y"], dt, sig, meas, max_gap)
        valid &= valid_y
        speed = np.hypot(vxs, vys)
        vmax = np.where(is_ball, cfg.max_ball_speed, cfg.max_player_speed)
        over = speed > vmax[None, :]
        n_speed_capped += int(np.nansum(over & ~is_ball[None, :]))
        scale = np.where(over, vmax[None, :] / np.maximum(speed, 1e-9), 1.0)
        vxs, vys = vxs * scale, vys * scale
        speed = np.hypot(vxs, vys)
        acc = np.hypot(np.gradient(vxs, dt, axis=0), np.gradient(vys, dt, axis=0)) if len(frames) > 1 else np.zeros_like(vxs)
        n_accel_flags += int(np.nansum((acc > MAX_PLAYER_ACCEL) & ~is_ball[None, :]))

        r, c = np.nonzero(valid)
        conf = w.get("confidence")
        part = pd.DataFrame({
            "frame": frames[r], "period": period, "id": np.asarray(ids, dtype=object)[c],
            "x": xs[r, c], "y": ys[r, c], "vx": vxs[r, c], "vy": vys[r, c], "speed": speed[r, c],
            "estimated": ~observed[r, c],
            "confidence": np.where(observed[r, c], conf[r, c] if conf is not None else 1.0, 0.5),
        })
        part["confidence"] = part["confidence"].fillna(0.5)
        for col in ("team", "z"):
            if col in df.columns:
                if col == "team":
                    part[col] = part["id"].map(meta_cols[col])
                else:
                    zmap = df.set_index(["frame", "id"])[col]
                    part[col] = zmap.reindex(pd.MultiIndex.from_arrays([part["frame"], part["id"]])).to_numpy()
        # carry the period clock
        t0 = df["frame"].min() - df.loc[df["frame"].idxmin(), "t"] * cfg.fps
        part["t"] = (part["frame"] - t0) / cfg.fps
        out.append(part)
    res = pd.concat(out, ignore_index=True) if out else tracking.iloc[0:0].copy()
    extra = [c for c in tracking.columns if c not in res.columns]
    for c in extra:
        res[c] = np.nan if tracking[c].dtype.kind == "f" else pd.NA
    qa = {"speed_capped_samples": n_speed_capped, "accel_over_cap_samples": n_accel_flags,
          "estimated_fraction": float(res["estimated"].mean()) if len(res) else 0.0}
    return res, qa
