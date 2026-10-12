"""Pitch geometry in centred metres.

The 32 keypoints follow the order used by the roboflow/sports pretrained pitch-keypoint model, so the
model's output index ``i`` maps directly to ``PitchDims.keypoints()[i]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PENALTY_BOX_WIDTH = 40.32
PENALTY_BOX_LENGTH = 16.5
GOAL_BOX_WIDTH = 18.32
GOAL_BOX_LENGTH = 5.5
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0
GOAL_WIDTH = 7.32


@dataclass(frozen=True)
class PitchDims:
    length: float = 105.0
    width: float = 68.0

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

    def contains(self, x: np.ndarray, y: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return (np.abs(x) <= self.half_length + margin) & (np.abs(y) <= self.half_width + margin)

    def goal_centre(self, attack_sign: int) -> tuple[float, float]:
        """Centre of the goal a team attacks; ``attack_sign`` is +1 (attacks +x) or -1."""
        return (attack_sign * self.half_length, 0.0)

    def keypoints(self) -> np.ndarray:
        """32 x 2 array of pitch keypoints in centred metres (roboflow/sports order)."""
        L, W = self.length, self.width
        pbw, pbl = PENALTY_BOX_WIDTH, PENALTY_BOX_LENGTH
        gbw, gbl = GOAL_BOX_WIDTH, GOAL_BOX_LENGTH
        r, ps = CENTRE_CIRCLE_RADIUS, PENALTY_SPOT_DISTANCE
        # (x from left goal line, y from top touchline), then shifted to the centre.
        pts = [
            (0, 0), (0, (W - pbw) / 2), (0, (W - gbw) / 2), (0, (W + gbw) / 2), (0, (W + pbw) / 2), (0, W),
            (gbl, (W - gbw) / 2), (gbl, (W + gbw) / 2),
            (ps, W / 2),
            (pbl, (W - pbw) / 2), (pbl, (W - gbw) / 2), (pbl, (W + gbw) / 2), (pbl, (W + pbw) / 2),
            (L / 2, 0), (L / 2, W / 2 - r), (L / 2, W / 2 + r), (L / 2, W),
            (L - pbl, (W - pbw) / 2), (L - pbl, (W - gbw) / 2), (L - pbl, (W + gbw) / 2), (L - pbl, (W + pbw) / 2),
            (L - ps, W / 2),
            (L - gbl, (W - gbw) / 2), (L - gbl, (W + gbw) / 2),
            (L, 0), (L, (W - pbw) / 2), (L, (W - gbw) / 2), (L, (W + gbw) / 2), (L, (W + pbw) / 2), (L, W),
            (L / 2 - r, W / 2), (L / 2 + r, W / 2),
        ]
        arr = np.asarray(pts, dtype=float)
        arr[:, 0] -= L / 2
        arr[:, 1] -= W / 2
        return arr

    def line_segments(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Straight painted lines as segments (centred metres). Used by the calibration line check."""
        k = self.keypoints()
        # 1-based keypoint index pairs (same edges as roboflow/sports config).
        edges = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
            (16, 17), (18, 19), (19, 20), (20, 21), (23, 24), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17), (14, 25), (18, 26), (23, 27), (24, 28), (21, 29),
            (17, 30),
        ]
        return [(tuple(k[a - 1]), tuple(k[b - 1])) for a, b in edges]

    def line_samples(self, step: float = 1.0) -> np.ndarray:
        """Points sampled every ``step`` metres along all painted lines, incl. the centre circle."""
        out = []
        for (x0, y0), (x1, y1) in self.line_segments():
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step) + 1)
            t = np.linspace(0, 1, n)
            out.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1))
        n = int(2 * np.pi * CENTRE_CIRCLE_RADIUS / step)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        out.append(np.stack([CENTRE_CIRCLE_RADIUS * np.cos(a), CENTRE_CIRCLE_RADIUS * np.sin(a)], axis=1))
        return np.concatenate(out, axis=0)
