"""Shared helpers: wide (frame x object) matrices, attack direction, constants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmi_schema.models import MatchMeta

MAX_PLAYER_SPEED = 10.0  # m/s (36 km/h)
MAX_PLAYER_ACCEL = 8.0  # m/s^2
MAX_BALL_SPEED = 40.0  # m/s


@dataclass
class Wide:
    """Tracking pivoted to matrices of shape (n_frames, n_objects) for one period."""

    frames: np.ndarray
    ids: list[str]
    teams: list[str]
    x: np.ndarray
    y: np.ndarray

    def col(self, oid: str) -> int:
        return self.ids.index(oid)


def to_wide(df: pd.DataFrame, values=("x", "y")) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Pivot long tracking of ONE period to a dense frame grid. Missing samples are NaN."""
    frames = np.arange(int(df["frame"].min()), int(df["frame"].max()) + 1)
    ids = list(pd.unique(df["id"]))
    col = {oid: i for i, oid in enumerate(ids)}
    r = (df["frame"].to_numpy() - frames[0]).astype(int)
    c = df["id"].map(col).to_numpy()
    out = {}
    for v in values:
        m = np.full((len(frames), len(ids)), np.nan)
        m[r, c] = df[v].to_numpy(dtype=float)
        out[v] = m
    return frames, ids, out


def team_of(df: pd.DataFrame) -> dict[str, str]:
    return df.drop_duplicates("id").set_index("id")["team"].astype(str).to_dict()


def attack_sign(meta: MatchMeta, period: int, team: str) -> int:
    p = meta.period(period)
    return p.attack_sign(team) if p else 1


def normalize_to_attack(x, y, sign):
    """Rotate coordinates so the team attacks +x."""
    s = np.asarray(sign)
    return np.asarray(x) * s, np.asarray(y) * s


def goalkeepers(pkg) -> dict[str, str]:
    """{team: player_id} for each team's GK: roster position if known, else the player deepest on average."""
    out = {}
    roster = pkg.roster
    if "position" in roster:
        for team in ("home", "away"):
            gk = roster[(roster["team"] == team) & roster["position"].astype(str).str.upper().isin(["GK", "GOALKEEPER"])]
            starters = gk[gk["starter"].fillna(False).astype(bool)] if "starter" in gk else gk
            pick = starters if not starters.empty else gk
            if not pick.empty:
                out[team] = str(pick.iloc[0]["player_id"])
    tr = pkg.players()
    for team in ("home", "away"):
        if team in out or tr.empty:
            continue
        sub = tr[tr["team"] == team]
        if sub.empty:
            continue
        sign = sub["period"].map(lambda p: attack_sign(pkg.meta, int(p), team))
        depth = (sub["x"] * sign).groupby(sub["id"]).mean()
        counts = sub.groupby("id").size()
        depth = depth[counts > counts.max() * 0.3]
        if not depth.empty:
            out[team] = str(depth.idxmin())
    return out
