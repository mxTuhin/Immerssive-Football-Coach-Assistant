"""Fake models driven by a SyntheticScene. Noisy on purpose (misses, jitter, weak boxes, OCR errors) so the
tracker's second pass, linking, voting and review logic are actually exercised in CI."""

from __future__ import annotations

import zlib

import numpy as np

from fmi_pipeline.geometry import apply_h
from fmi_pipeline.models.base import CLASS_IDS, Detections
from fmi_pipeline.synthetic import SyntheticScene


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, keys)).encode()))


class FakeDetector:
    def __init__(self, scene: SyntheticScene, miss: float = 0.04, ball_miss: float = 0.25, jitter: float = 2.0):
        self.scene, self.miss, self.ball_miss, self.jitter = scene, miss, ball_miss, jitter

    def detect(self, frame: np.ndarray, conf: float = 0.1) -> Detections:
        f = getattr(frame, "index", -1)
        if f < 0:
            return Detections.empty()
        rng = _rng("det", f)
        boxes, scores, cls = [], [], []
        for o in self.scene.boxes(f):
            if rng.random() < (self.ball_miss if o["role"] == "ball" else self.miss):
                continue
            b = np.asarray(o["box"]) + rng.normal(0, self.jitter, 4)
            s = rng.uniform(0.55, 0.95) if rng.random() > 0.08 else rng.uniform(0.2, 0.5)  # some weak boxes
            boxes.append(b)
            scores.append(s)
            cls.append(CLASS_IDS[o["role"]])
        if rng.random() < 0.1:  # occasional ball false positive (crowd / line)
            u, v = rng.uniform(100, 1800), rng.uniform(300, 1000)
            boxes.append([u - 5, v - 10, u + 5, v])
            scores.append(rng.uniform(0.05, 0.3))
            cls.append(CLASS_IDS["ball"])
        if not boxes:
            return Detections.empty()
        d = Detections(np.asarray(boxes, float), np.asarray(scores, float), np.asarray(cls, int))
        return d.select(d.score >= conf)


class FakeKeypoints:
    def __init__(self, scene: SyntheticScene, noise_px: float = 1.5):
        self.scene, self.noise = scene, noise_px

    def predict(self, frame: np.ndarray):
        f = getattr(frame, "index", 0)
        rng = _rng("kp", f)
        kp = apply_h(self.scene.H_pitch2img, self.scene.pitch.keypoints()) + rng.normal(0, self.noise, (32, 2))
        vis = (kp[:, 0] > 0) & (kp[:, 0] < self.scene.width) & (kp[:, 1] > 0) & (kp[:, 1] < self.scene.height)
        conf = np.where(vis, rng.uniform(0.6, 0.99, 32), rng.uniform(0.0, 0.3, 32))
        return kp, conf


class FakeTeamEmbedder:
    dim = 16

    def __init__(self, scene: SyntheticScene):
        self.scene = scene
        base = np.zeros((3, self.dim))
        base[0, 0] = base[1, 1] = base[2, 2] = 3.0
        self.base = {"home": base[0], "away": base[1], "referee": base[2]}

    def embed(self, crops):
        out = []
        for c in crops:
            o = self.scene.nearest(getattr(c, "index", -1), getattr(c, "box", (0, 0, 0, 0))) if getattr(c, "box", None) else None
            rng = _rng("emb", getattr(c, "index", -1), getattr(c, "box", None))
            v = self.base.get(o["team"], np.zeros(self.dim)) if o else np.zeros(self.dim)
            out.append(v + rng.normal(0, 0.4, self.dim))
        return np.asarray(out).reshape(len(crops), self.dim)


class FakeReID:
    dim = 32

    def __init__(self, scene: SyntheticScene, noise: float = 0.35):
        self.scene, self.noise = scene, noise

    def embed(self, crops):
        out = []
        for c in crops:
            box = getattr(c, "box", None)
            o = self.scene.nearest(getattr(c, "index", -1), box) if box else None
            ident = _rng("id", o["id"] if o else "none").normal(0, 1, self.dim)
            v = ident / np.linalg.norm(ident) + _rng("reid", getattr(c, "index", -1), box).normal(0, self.noise / np.sqrt(self.dim), self.dim)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out).reshape(len(crops), self.dim)


class FakeOCR:
    def __init__(self, scene: SyntheticScene, p_read: float = 0.35, p_wrong: float = 0.05):
        self.scene, self.p_read, self.p_wrong = scene, p_read, p_wrong

    def read(self, crops):
        out = []
        for c in crops:
            box = getattr(c, "box", None)
            o = self.scene.nearest(getattr(c, "index", -1), box) if box else None
            rng = _rng("ocr", getattr(c, "index", -1), box)
            if o is None or o["number"] is None or rng.random() > self.p_read:
                out.append(("", 0.0))
            elif rng.random() < self.p_wrong:
                out.append((str(int(rng.integers(1, 30))), float(rng.uniform(0.3, 0.7))))
            else:
                out.append((str(o["number"]), float(rng.uniform(0.6, 0.98))))
        return out
