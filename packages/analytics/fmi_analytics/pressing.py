"""Pressing: pressure on the ball carrier and PPDA.

Pressure: opponents within 5 m of the ball carrier; a *press* is >= 1 opponent within 3 m closing at > 1.5 m/s
or >= 2 within 5 m. PPDA = opponent passes in their own 60% of the pitch / our defensive actions there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign, to_wide


def pressure_timeseries(tracking: pd.DataFrame, possession: pd.DataFrame, step: int = 2) -> pd.DataFrame:
    """Pressure on the ball carrier every ``step`` frames (vectorised per period)."""
    out = []
    for period, pp in possession.groupby("period", sort=True):
        pp = pp[pp["owner"].notna() & (pp["frame"] % step == 0)]
        players = tracking[(tracking["period"] == period) & tracking["team"].isin(["home", "away"])]
        if pp.empty or players.empty:
            continue
        frames, ids, w = to_wide(players, values=("x", "y", "vx", "vy"))
        team = players.drop_duplicates("id").set_index("id")["team"].reindex(ids).to_numpy()
        col = {oid: i for i, oid in enumerate(ids)}
        keep = pp["owner"].isin(col) & pp["frame"].between(frames[0], frames[-1])
        pp = pp[keep]
        if pp.empty:
            continue
        r = (pp["frame"].to_numpy() - frames[0]).astype(int)
        c = pp["owner"].map(col).to_numpy()
        cx, cy = w["x"][r, c], w["y"][r, c]
        dx, dy = cx[:, None] - w["x"][r], cy[:, None] - w["y"][r]
        d = np.hypot(dx, dy)
        opp = team[None, :] != np.asarray(pp["owner_team"].to_numpy(), dtype=object)[:, None]
        d = np.where(opp & np.isfinite(d), d, np.inf)
        closing = (w["vx"][r] * dx + w["vy"][r] * dy) / np.maximum(d, 1e-6)
        n5 = (d < 5).sum(1)
        n3c = ((d < 3) & (np.nan_to_num(closing) > 1.5)).sum(1)
        nearest = d.min(1)
        out.append(pd.DataFrame({
            "frame": pp["frame"].to_numpy(), "period": int(period), "t": pp["t"].to_numpy(),
            "carrier": pp["owner"].to_numpy(), "carrier_team": pp["owner_team"].to_numpy(),
            "pressing_team": np.where(pp["owner_team"].to_numpy() == "home", "away", "home"),
            "opp_within_5m": n5, "closing_within_3m": n3c,
            "nearest_opp_m": np.where(np.isfinite(nearest), nearest, np.nan),
            "pressed": (n3c >= 1) | (n5 >= 2),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def ppda(events: pd.DataFrame, meta) -> dict:
    """Passes allowed per defensive action, computed in the opponent's own 60% of the pitch."""
    out = {}
    L = meta.pitch_length
    if events.empty:
        return out
    for team in ("home", "away"):
        opp = "away" if team == "home" else "home"
        passes = events[(events["team"] == opp) & events["type"].isin(["pass", "cross"])]
        sign_opp = passes["period"].map(lambda p: attack_sign(meta, int(p), opp))
        n_pass = int(((passes["start_x"] * sign_opp) < 0.1 * L).sum())
        acts = events[(events["team"] == team) & events["type"].isin(["interception", "tackle", "recovery", "foul"])]
        sign = acts["period"].map(lambda p: attack_sign(meta, int(p), team))
        n_def = int(((acts["start_x"] * sign) > -0.1 * L).sum())
        out[team] = round(n_pass / n_def, 2) if n_def else None
    return out


def press_summary(pressure: pd.DataFrame) -> dict:
    if pressure.empty:
        return {}
    out = {}
    for team, g in pressure.groupby("pressing_team"):
        out[team] = {"pressed_share": round(float(g["pressed"].mean()), 3),
                     "mean_opp_within_5m": round(float(g["opp_within_5m"].mean()), 2),
                     "mean_nearest_opp_m": round(float(g["nearest_opp_m"].mean()), 2)}
    return out
