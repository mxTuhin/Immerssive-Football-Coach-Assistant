"""Helpers shared by provider adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_schema.models import MatchMeta, PeriodInfo


def resample_long(df: pd.DataFrame, fps_in: float, fps_out: float, value_cols=("x", "y", "z")) -> pd.DataFrame:
    """Resample long-format tracking (columns period, t, id, team, values) to ``fps_out`` by linear interpolation.

    Interpolation never bridges gaps longer than 0.5 s: those samples stay NaN.
    """
    if fps_in == fps_out:
        return df
    out = []
    for (period, oid), g in df.groupby(["period", "id"], sort=False):
        g = g.sort_values("t")
        t_in = g["t"].to_numpy()
        if len(t_in) < 2:
            continue
        t_out = np.arange(np.ceil(t_in[0] * fps_out), np.floor(t_in[-1] * fps_out) + 1) / fps_out
        rec = {"period": period, "id": oid, "t": t_out}
        # gap mask: nearest input sample further than 0.5 s away
        idx = np.clip(np.searchsorted(t_in, t_out), 1, len(t_in) - 1)
        gap = (t_in[idx] - t_in[idx - 1]) > 0.5
        for c in value_cols:
            if c not in g:
                continue
            v = g[c].to_numpy(dtype=float)
            ok = np.isfinite(v)
            if ok.sum() < 2:
                rec[c] = np.full(len(t_out), np.nan)
                continue
            r = np.interp(t_out, t_in[ok], v[ok], left=np.nan, right=np.nan)
            r[gap] = np.nan
            rec[c] = r
        part = pd.DataFrame(rec)
        for c in g.columns:
            if c not in part.columns and c not in value_cols:
                part[c] = g[c].iloc[0]
        out.append(part)
    if not out:
        return df.iloc[0:0]
    return pd.concat(out, ignore_index=True)


def assign_frames(tracking: pd.DataFrame, meta: MatchMeta) -> pd.DataFrame:
    """Set the global 10 Hz ``frame`` column from (period, t) using ``meta.periods``."""
    starts = {p.period: p.start_frame for p in meta.periods}
    tracking = tracking.copy()
    tracking["frame"] = (tracking["period"].map(starts) + np.round(tracking["t"] * meta.fps)).astype("int64")
    return tracking


def infer_home_attack_sign(tracking: pd.DataFrame, period: int) -> int:
    """+1 if the home team attacks +x in ``period``: at the start of a period each team stands in its own half."""
    tr = tracking[(tracking["period"] == period) & tracking["team"].isin(["home", "away"])]
    if tr.empty:
        return 1
    first = tr[tr["t"] <= tr["t"].min() + 2.0]
    mx = first.groupby("team")["x"].mean()
    if "home" in mx and "away" in mx and np.isfinite(mx["home"]) and np.isfinite(mx["away"]):
        return 1 if mx["home"] < mx["away"] else -1
    return 1


def build_periods(tracking: pd.DataFrame, fps: float) -> list[PeriodInfo]:
    periods = []
    frame = 0
    for p in sorted(tracking["period"].dropna().unique()):
        tmax = tracking.loc[tracking["period"] == p, "t"].max()
        n = int(round(tmax * fps))
        periods.append(PeriodInfo(period=int(p), start_frame=frame, end_frame=frame + n,
                                  home_attack_sign=infer_home_attack_sign(tracking, int(p))))
        frame += n + int(fps * 60)  # leave a one-minute gap between periods
    return periods
