"""Expected Threat (xT) by value iteration on a grid (Karun Singh's method, as in socceraction).

The grid is fitted from event data with coordinates normalised so the acting team attacks +x. We fit a
default grid from all bundled open-data events (``scripts/fit_xt.py``) and store it at
``packages/analytics/fmi_analytics/data/xt_grid.json``; until then ``fit_xt`` runs on the match itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign

GRID = (16, 12)  # (x cells, y cells)
DEFAULT_PATH = Path(__file__).parent / "data" / "xt_grid.json"
MOVE_TYPES = {"pass", "cross", "dribble"}


def _cells(x, y, L, W, grid=GRID):
    cx = np.clip(((np.asarray(x) + L / 2) / L * grid[0]).astype(int), 0, grid[0] - 1)
    cy = np.clip(((np.asarray(y) + W / 2) / W * grid[1]).astype(int), 0, grid[1] - 1)
    return cx, cy


def normalized_events(events: pd.DataFrame, meta) -> pd.DataFrame:
    ev = events[events["team"].isin(["home", "away"])].copy()
    s = np.array([attack_sign(meta, int(p), t) for p, t in zip(ev["period"], ev["team"])], dtype=float)
    for c in ("start_x", "start_y", "end_x", "end_y"):
        ev[c] = ev[c].astype(float) * s
    return ev


def fit_xt(norm_events: pd.DataFrame, L: float = 105.0, W: float = 68.0, grid=GRID, iters: int = 30) -> np.ndarray:
    ev = norm_events.dropna(subset=["start_x", "start_y"])
    nx, ny = grid
    shots = ev[ev["type"] == "shot"]
    moves = ev[ev["type"].isin(MOVE_TYPES)]
    sx, sy = _cells(shots["start_x"], shots["start_y"], L, W, grid)
    mx, my = _cells(moves["start_x"], moves["start_y"], L, W, grid)
    shot_n = np.zeros(grid); goal_n = np.zeros(grid); move_n = np.zeros(grid)
    np.add.at(shot_n, (sx, sy), 1)
    np.add.at(goal_n, (sx, sy), (shots["outcome"] == "success").to_numpy(dtype=float))
    np.add.at(move_n, (mx, my), 1)
    total = shot_n + move_n
    with np.errstate(invalid="ignore", divide="ignore"):
        p_shot = np.nan_to_num(shot_n / total)
        p_move = np.nan_to_num(move_n / total)
        p_goal = np.nan_to_num(goal_n / shot_n)
    # transition matrix from successful moves
    ok = moves[(moves["outcome"] == "success") & moves["end_x"].notna()]
    T = np.zeros((nx, ny, nx, ny))
    a, b = _cells(ok["start_x"], ok["start_y"], L, W, grid)
    c, d = _cells(ok["end_x"], ok["end_y"], L, W, grid)
    np.add.at(T, (a, b, c, d), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.nan_to_num(T / move_n[:, :, None, None])
    xt = np.zeros(grid)
    for _ in range(iters):
        xt = p_shot * p_goal + p_move * np.einsum("ijkl,kl->ij", T, xt)
    return xt


def load_default_grid() -> np.ndarray | None:
    if DEFAULT_PATH.exists():
        return np.asarray(json.loads(DEFAULT_PATH.read_text())["grid"])
    return None


def xt_added(events: pd.DataFrame, meta, grid_values: np.ndarray | None = None) -> pd.DataFrame:
    """xT added by each successful move: xT(end) - xT(start)."""
    L, W = meta.pitch_length, meta.pitch_width
    ev = normalized_events(events, meta)
    if grid_values is None:
        grid_values = load_default_grid()
    if grid_values is None:
        grid_values = fit_xt(ev, L, W)
    g = grid_values.shape
    moves = ev[ev["type"].isin(MOVE_TYPES) & (ev["outcome"] == "success")].dropna(subset=["end_x"])
    a, b = _cells(moves["start_x"], moves["start_y"], L, W, g)
    c, d = _cells(moves["end_x"], moves["end_y"], L, W, g)
    moves = moves.assign(xt_start=grid_values[a, b], xt_end=grid_values[c, d])
    moves["xt_added"] = moves["xt_end"] - moves["xt_start"]
    return moves[["event_id", "period", "t", "team", "player_id", "type", "xt_start", "xt_end", "xt_added"]]
