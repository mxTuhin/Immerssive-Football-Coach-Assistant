"""Phases of play per team from possession + shape.

Frames without ball or shape data are labelled ``unknown`` rather than guessed.

In possession:  build_up (ball in own third) | progression (middle third) | final_third
                transition_attack in the first 5 s after winning the ball
Out of possession: high_block | mid_block | low_block by defensive line height
                   transition_defence in the first 5 s after losing the ball
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign

TRANSITION_S = 5.0


def phases_timeseries(possession: pd.DataFrame, shape_ts: pd.DataFrame, meta, fps: float = 10.0) -> pd.DataFrame:
    if possession.empty:
        return pd.DataFrame()
    L = meta.pitch_length
    lh = {}
    if not shape_ts.empty:
        lh = shape_ts.set_index(["frame", "team"])["line_height"].to_dict()
    rows = []
    for period, pp in possession.groupby("period", sort=True):
        pp = pp.sort_values("frame").copy()
        # bridge short ball-tracking gaps (<= 5 s) with the last seen ball position
        pp["ball_x"] = pp["ball_x"].ffill(limit=int(5 * fps))
        tip = pp["team_in_possession"].to_numpy(dtype=object)
        change = np.zeros(len(tip), dtype=bool)
        change[1:] = tip[1:] != tip[:-1]
        last_change = -10 ** 9
        for i, r in enumerate(pp.itertuples(index=False)):
            if change[i]:
                last_change = i
            if r.team_in_possession is None or (isinstance(r.team_in_possession, float) and np.isnan(r.team_in_possession)):
                continue
            att = r.team_in_possession
            dfn = "away" if att == "home" else "home"
            in_transition = (i - last_change) / fps < TRANSITION_S and last_change > 0
            s = attack_sign(meta, int(period), att)
            bx = r.ball_x * s if np.isfinite(r.ball_x) else np.nan
            if in_transition:
                pa, pd_ = "transition_attack", "transition_defence"
            else:
                if not np.isfinite(bx):
                    pa = "unknown"
                elif bx < -L / 6:
                    pa = "build_up"
                elif bx < L / 6:
                    pa = "progression"
                else:
                    pa = "final_third"
                key = (int(r.frame) - int(r.frame) % 10, dfn)
                h = lh.get(key)
                if h is None:
                    pd_ = "unknown"
                elif h > 0.55 * L:
                    pd_ = "high_block"
                elif h > 0.35 * L:
                    pd_ = "mid_block"
                else:
                    pd_ = "low_block"
            rows.append({"frame": int(r.frame), "period": int(period), "t": float(r.t), "attacking": att,
                         "phase_attack": pa, "phase_defence": pd_})
    return pd.DataFrame(rows)


def phase_segments(ph: pd.DataFrame, fps: float = 10.0, min_s: float = 2.0) -> pd.DataFrame:
    """Collapse the per-frame table into contiguous segments."""
    if ph.empty:
        return ph
    key = ph["attacking"] + "|" + ph["phase_attack"] + "|" + ph["phase_defence"] + "|" + ph["period"].astype(str)
    grp = (key != key.shift()).cumsum()
    seg = ph.groupby(grp).agg(period=("period", "first"), t_start=("t", "first"), t_end=("t", "last"),
                              attacking=("attacking", "first"), phase_attack=("phase_attack", "first"),
                              phase_defence=("phase_defence", "first"))
    seg["duration_s"] = seg["t_end"] - seg["t_start"] + 1 / fps
    return seg[seg["duration_s"] >= min_s].reset_index(drop=True)


def phase_share(ph: pd.DataFrame) -> dict:
    if ph.empty:
        return {}
    out = {}
    for team in ("home", "away"):
        att = ph[ph["attacking"] == team]["phase_attack"].value_counts(normalize=True).round(3).to_dict()
        dfn = ph[ph["attacking"] != team]["phase_defence"].value_counts(normalize=True).round(3).to_dict()
        out[team] = {"in_possession": att, "out_of_possession": dfn}
    return out
