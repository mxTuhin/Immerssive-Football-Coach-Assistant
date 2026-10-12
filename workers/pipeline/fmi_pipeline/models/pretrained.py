"""Pretrained models used by the MVP (no training). All imports are lazy so CPU-only installs still work.

* Player/GK/referee/ball detector and ball detector: roboflow/sports YOLOv8 weights via ultralytics
* Pitch keypoints: roboflow/sports YOLOv8-pose (32 keypoints)
* Team embeddings: SigLIP vision tower (transformers)
* ReID: OSNet via torchreid
* Jersey numbers: PARSeq via torch.hub (baudm/parseq)
* Colour histogram embedder: CPU fallback for team clustering
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from fmi_pipeline.models.base import CLASS_IDS, Detections

log = logging.getLogger(__name__)


class YoloDetector:
    """Ultralytics YOLO wrapper mapping the model's class names onto our class ids."""

    def __init__(self, weights: str | Path, imgsz: int = 1280, device: str | None = None,
                 force_class: str | None = None):
        from ultralytics import YOLO  # type: ignore

        if not Path(weights).exists():
            raise FileNotFoundError(f"{weights} missing; run `python scripts/fetch_data.py --models`")
        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device
        names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.map = {k: CLASS_IDS.get(force_class or v, CLASS_IDS["player"]) for k, v in names.items()}

    def detect(self, frame: np.ndarray, conf: float = 0.1) -> Detections:
        r = self.model.predict(np.asarray(frame), imgsz=self.imgsz, conf=conf, device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections.empty()
        cls = np.array([self.map.get(int(c), CLASS_IDS["player"]) for c in b.cls.cpu().numpy()], dtype=int)
        return Detections(b.xyxy.cpu().numpy().astype(float), b.conf.cpu().numpy().astype(float), cls)


class CombinedDetector:
    """Players from the player model; ball from the dedicated ball model (better recall on small balls)."""

    def __init__(self, player: YoloDetector, ball: YoloDetector | None = None, ball_conf: float = 0.05):
        self.player, self.ball, self.ball_conf = player, ball, ball_conf

    def detect(self, frame: np.ndarray, conf: float = 0.1) -> Detections:
        d = self.player.detect(frame, conf=conf)
        if self.ball is None:
            return d
        d = d.select(d.cls != CLASS_IDS["ball"])
        return Detections.concat([d, self.ball.detect(frame, conf=self.ball_conf)])


class YoloPitchKeypoints:
    def __init__(self, weights: str | Path, imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO  # type: ignore

        if not Path(weights).exists():
            raise FileNotFoundError(f"{weights} missing; run `python scripts/fetch_data.py --models`")
        self.model = YOLO(str(weights))
        self.imgsz, self.device = imgsz, device

    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.predict(np.asarray(frame), imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return np.full((32, 2), np.nan), np.zeros(32)
        xy = r.keypoints.xy[0].cpu().numpy().astype(float)
        conf = r.keypoints.conf[0].cpu().numpy().astype(float) if r.keypoints.conf is not None else np.ones(len(xy))
        return xy, conf


class SiglipEmbedder:
    def __init__(self, name: str = "google/siglip-base-patch16-224", device: str = "cuda", batch: int = 32):
        import torch  # type: ignore
        from transformers import AutoProcessor, SiglipVisionModel  # type: ignore

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = SiglipVisionModel.from_pretrained(name).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(name)
        self.batch = batch
        self.dim = self.model.config.hidden_size

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(crops), self.batch):
                imgs = [np.ascontiguousarray(c[..., ::-1]) for c in crops[i:i + self.batch]]  # BGR -> RGB
                inp = self.proc(images=imgs, return_tensors="pt").to(self.device)
                out.append(self.model(**inp).pooler_output.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim))


class ColorHistEmbedder:
    """HSV histogram of the torso: cheap CPU team features (fallback for SigLIP)."""

    dim = 48

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        feats = []
        for c in crops:
            c = np.asarray(c)
            if c.size == 0 or c.ndim != 3:
                feats.append(np.zeros(self.dim))
                continue
            h, w = c.shape[:2]
            torso = c[int(0.15 * h):int(0.55 * h), int(0.2 * w):int(0.8 * w)].reshape(-1, 3).astype(float) / 255.0
            if not len(torso):
                feats.append(np.zeros(self.dim))
                continue
            b, g, r = torso[:, 0], torso[:, 1], torso[:, 2]
            mx, mn = torso.max(1), torso.min(1)
            v = mx
            s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
            hue = np.zeros_like(mx)
            d = np.maximum(mx - mn, 1e-9)
            hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
            grass = (hue > 0.17) & (hue < 0.45) & (s > 0.25)
            keep = ~grass
            hh = np.histogram(hue[keep], 16, (0, 1))[0]
            sh = np.histogram(s[keep], 16, (0, 1))[0]
            vh = np.histogram(v[keep], 16, (0, 1))[0]
            f = np.concatenate([hh, sh, vh]).astype(float)
            feats.append(f / max(f.sum(), 1))
        return np.asarray(feats)


class OsnetReID:
    def __init__(self, weights: str | Path | None, device: str = "cuda", model_name: str = "osnet_x1_0"):
        from torchreid.utils import FeatureExtractor  # type: ignore

        self.extractor = FeatureExtractor(model_name=model_name, model_path=str(weights) if weights else "",
                                          device=device)
        self.dim = 512

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim))
        imgs = [np.ascontiguousarray(np.asarray(c)[..., ::-1]) for c in crops]
        f = self.extractor(imgs).cpu().numpy()
        return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)


class ParseqOCR:
    """Scene-text recogniser; we keep only 1-2 digit outputs as jersey numbers."""

    def __init__(self, repo: str = "baudm/parseq", device: str = "cuda", batch: int = 64):
        import torch  # type: ignore

        self.torch = torch
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model = torch.hub.load(repo, "parseq", pretrained=True, trust_repo=True).eval().to(self.device)
        self.batch = batch

    def _prep(self, crop: np.ndarray):
        import cv2  # type: ignore

        img = cv2.resize(np.ascontiguousarray(np.asarray(crop)[..., ::-1]), (128, 32)).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return self.torch.from_numpy(img.transpose(2, 0, 1))

    def read(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        with self.torch.no_grad():
            for i in range(0, len(crops), self.batch):
                chunk = [c for c in crops[i:i + self.batch]]
                ok = [c is not None and np.asarray(c).size > 0 for c in chunk]
                imgs = [self._prep(c) for c, k in zip(chunk, ok) if k]
                res: list[tuple[str, float]] = []
                if imgs:
                    logits = self.model(self.torch.stack(imgs).to(self.device))
                    labels, confs = self.model.tokenizer.decode(logits.softmax(-1))
                    res = [(lab, float(c.prod().item()) if hasattr(c, "prod") else float(c)) for lab, c in zip(labels, confs)]
                it = iter(res)
                for k in ok:
                    text, conf = next(it) if k else ("", 0.0)
                    digits = "".join(ch for ch in text if ch.isdigit())
                    out.append((digits if 1 <= len(digits) <= 2 else "", conf if digits else 0.0))
        return out
