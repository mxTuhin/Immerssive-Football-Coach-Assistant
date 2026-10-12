"""Model interfaces. Stages only see these; real pretrained models and test fakes implement them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}


@dataclass
class Detections:
    xyxy: np.ndarray   # (N, 4) float pixels
    score: np.ndarray  # (N,)
    cls: np.ndarray    # (N,) int, see CLASS_NAMES

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))

    def __len__(self) -> int:
        return len(self.score)

    def select(self, mask) -> "Detections":
        return Detections(self.xyxy[mask], self.score[mask], self.cls[mask])

    @classmethod
    def concat(cls, items: list["Detections"]) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return cls.empty()
        return cls(np.concatenate([d.xyxy for d in items]), np.concatenate([d.score for d in items]),
                   np.concatenate([d.cls for d in items]))


class Detector(Protocol):
    def detect(self, frame: np.ndarray, conf: float = 0.1) -> Detections: ...


class KeypointModel(Protocol):
    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (32 x 2 pixel coords, 32 confidences) in the roboflow/sports keypoint order."""
        ...


class Embedder(Protocol):
    dim: int

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


class JerseyOCR(Protocol):
    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]: ...


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) boxes."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def nms(det: Detections, iou: float = 0.5) -> Detections:
    """Class-aware greedy NMS (used to merge tiles)."""
    keep = []
    for c in np.unique(det.cls):
        idx = np.nonzero(det.cls == c)[0]
        idx = idx[np.argsort(-det.score[idx])]
        while len(idx):
            i = idx[0]
            keep.append(i)
            if len(idx) == 1:
                break
            ious = box_iou(det.xyxy[i:i + 1], det.xyxy[idx[1:]])[0]
            idx = idx[1:][ious < iou]
    keep = np.array(sorted(keep), dtype=int)
    return det.select(keep) if len(keep) else Detections.empty()


def tiles(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    step = int(size * (1 - overlap))
    xs = list(range(0, max(width - size, 0) + 1, step)) or [0]
    ys = list(range(0, max(height - size, 0) + 1, step)) or [0]
    if xs[-1] + size < width:
        xs.append(width - size)
    if ys[-1] + size < height:
        ys.append(height - size)
    return [(x, y, min(x + size, width), min(y + size, height)) for y in ys for x in xs]


def detect_tiled(detector: Detector, frame: np.ndarray, conf: float, size: int, overlap: float,
                 iou: float = 0.5, full_frame: bool = True) -> Detections:
    """Sliced inference for 4K panoramic frames: tiles + (optionally) the full frame, merged by NMS."""
    H, W = frame.shape[:2]
    parts = []
    for x1, y1, x2, y2 in tiles(W, H, size, overlap):
        d = detector.detect(frame[y1:y2, x1:x2], conf=conf)
        if len(d):
            d.xyxy = d.xyxy + np.array([x1, y1, x1, y1], float)
            parts.append(d)
    if full_frame:
        parts.append(detector.detect(frame, conf=conf))
    return nms(Detections.concat(parts), iou)
