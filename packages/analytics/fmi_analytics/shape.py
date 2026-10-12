"""Team shape: centroid, width, length, compactness (hull area), defensive line height, by possession phase."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign, to_wide
from fmi_schema.geom import convex_hull_area


def team_shape_timeseries(tracking: pd.DataFrame, meta, gk_ids: set[str], possession: pd.DataFrame | None = None,
                          step: int = 10) -> pd.DataFrame:
    """Shape per team every ``step`` frames (default 1 Hz at 10 fps). Outfield players only.

    All x values are normalised so the team attacks +x; ``line_height`` is the distance (m) from the team's
    own goal line to the deepest outfield line (mean x of the deepest 3 players).
    """
    tr = tracking[tracking["team"].isin(["home", "away"]) & ~tracking["id"].isin(gk_ids)]
    tr = tr[tr["frame"] % step == 0]
    tip = possession.set_index("frame")["team_in_possession"] if possession is not None and not possession.empty else None
    half_l = meta.pitch_length / 2
    rows = []
    for (period, team), g in tr.groupby(["period", "team"], sort=True):
        frames, ids, w = to_wide(g, values=("x", "y"))
        sel = np.isin(frames, g["frame"].unique())
        frames = frames[sel]
        s = attack_sign(meta, int(period), team)
        X, Y = w["x"][sel] * s, w["y"][sel] * s
        n = np.isfinite(X).sum(1)
        ok = n >= 5
        frames, X, Y, n = frames[ok], X[ok], Y[ok], n[ok]
        if not len(frames):
            continue
        cx, cy = np.nanmean(X, 1), np.nanmean(Y, 1)
        deepest = np.sort(np.where(np.isfinite(X), X, np.inf), axis=1)[:, :3].mean(1)
        hull = [convex_hull_area(np.stack([x[np.isfinite(x)], y[np.isfinite(y)]], 1)) for x, y in zip(X, Y)]
        tmap = g.groupby("frame")["t"].first()
        state = [None] * len(frames)
        if tip is not None:
            tv = tip.reindex(frames).to_numpy(dtype=object)
            state = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else ("in" if v == team else "out")
                     for v in tv]
        rows.append(pd.DataFrame({
            "frame": frames, "period": int(period), "t": tmap.reindex(frames).to_numpy(), "team": team,
            "centroid_x": cx, "centroid_y": cy,
            "width": np.nanmax(Y, 1) - np.nanmin(Y, 1), "length": np.nanmax(X, 1) - np.nanmin(X, 1),
            "hull_area": hull, "line_height": deepest + half_l,
            "stretch_index": np.nanmean(np.hypot(X - cx[:, None], Y - cy[:, None]), 1),
            "possession": state,
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["frame", "team"]).reset_index(drop=True) if rows else pd.DataFrame()


def shape_summary(ts: pd.DataFrame) -> dict:
    if ts.empty:
        return {}
    out = {}
    cols = ["width", "length", "hull_area", "line_height", "centroid_x", "stretch_index"]
    for team, g in ts.groupby("team"):
        out[team] = {"all": g[cols].mean().round(2).to_dict()}
        for state, gg in g.groupby("possession"):
            out[team][f"{state}_possession"] = gg[cols].mean().round(2).to_dict()
    return out
