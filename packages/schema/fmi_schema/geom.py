"""Small numeric helpers shared across packages: assignment, angles, hulls."""

from __future__ import annotations

import numpy as np

BIG = 1e9


def linear_assignment(cost: np.ndarray, max_cost: float | None = None) -> list[tuple[int, int]]:
    """Minimum-cost bipartite matching. Returns (row, col) pairs, dropping pairs with cost > max_cost.

    Uses scipy when available and a pure-numpy Hungarian algorithm otherwise (identical results).
    Non-finite costs are treated as forbidden.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    c = np.where(np.isfinite(cost), cost, BIG)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(c)
    except ImportError:
        rows, cols = _hungarian(c)
    pairs = []
    for r, k in zip(rows, cols):
        v = c[r, k]
        if v >= BIG:
            continue
        if max_cost is not None and v > max_cost:
            continue
        pairs.append((int(r), int(k)))
    return pairs


def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1, :] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            used_idx = np.nonzero(used)[0]
            u[p[used_idx]] += delta
            v[used_idx] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] != 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows_a, cols_a = np.asarray(rows), np.asarray(cols)
    if transposed:
        rows_a, cols_a = cols_a, rows_a
    order = np.argsort(rows_a)
    return rows_a[order], cols_a[order]


def wrap_deg(a):
    """Wrap degrees to (-180, 180]."""
    a = np.asarray(a, dtype=float)
    w = (a + 180.0) % 360.0 - 180.0
    return np.where(w == -180.0, 180.0, w)


def angle_diff_deg(a, b):
    """Signed smallest difference a - b in degrees."""
    return wrap_deg(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def heading_deg(dx, dy):
    return np.degrees(np.arctan2(dy, dx))


def convex_hull_area(points: np.ndarray) -> float:
    """Area of the convex hull of N x 2 points (monotone chain). 0 for < 3 points."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.asarray(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
