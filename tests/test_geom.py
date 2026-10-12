import itertools

import numpy as np

from fmi_schema.geom import _hungarian, angle_diff_deg, convex_hull_area, linear_assignment, wrap_deg


def brute(cost):
    n, m = cost.shape
    best = np.inf
    for perm in itertools.permutations(range(m), n):
        best = min(best, sum(cost[i, perm[i]] for i in range(n)))
    return best


def test_hungarian_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, m = rng.integers(1, 5), rng.integers(4, 6)
        c = rng.random((n, m))
        r, k = _hungarian(c)
        assert np.isclose(c[r, k].sum(), brute(c))
        r2, k2 = _hungarian(c.T)  # tall matrix
        assert np.isclose(c.T[r2, k2].sum(), brute(c))


def test_linear_assignment_respects_forbidden_and_max_cost():
    c = np.array([[1.0, np.inf], [np.inf, 5.0]])
    assert linear_assignment(c) == [(0, 0), (1, 1)]
    assert linear_assignment(c, max_cost=2.0) == [(0, 0)]
    assert linear_assignment(np.zeros((0, 3))) == []


def test_angles():
    assert wrap_deg(190) == -170
    assert wrap_deg(-180) == 180
    assert np.isclose(angle_diff_deg(10, 350), 20)


def test_hull_area():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]])
    assert np.isclose(convex_hull_area(sq), 4.0)
    assert convex_hull_area(sq[:2]) == 0.0
