"""Model registry: lazily loads pretrained models (GPU worker) or fakes (tests / synthetic://)."""

from __future__ import annotations

import logging
from typing import Any, Callable

from fmi_pipeline.config import PipelineConfig
from fmi_pipeline.models.base import (
    CLASS_IDS,
    CLASS_NAMES,
    Detections,
    Detector,
    Embedder,
    JerseyOCR,
    KeypointModel,
    box_iou,
    detect_tiled,
    nms,
)

log = logging.getLogger(__name__)

__all__ = ["CLASS_IDS", "CLASS_NAMES", "Detections", "Detector", "Embedder", "JerseyOCR", "KeypointModel",
           "ModelRegistry", "box_iou", "detect_tiled", "nms"]


class ModelRegistry:
    """``registry.detector``, ``.keypoints``, ``.team_embedder``, ``.reid``, ``.ocr`` -- created on first use."""

    def __init__(self, factories: dict[str, Callable[[], Any]]):
        self._factories = factories
        self._cache: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name not in self._cache:
            if name not in self._factories:
                raise KeyError(f"no model {name!r} configured")
            self._cache[name] = self._factories[name]()
        return self._cache[name]

    def has(self, name: str) -> bool:
        if name not in self._factories:
            return False
        try:
            self.get(name)
            return True
        except Exception as exc:  # missing weights / packages: stage falls back
            log.warning("model %s unavailable: %s", name, exc)
            self._factories.pop(name, None)
            return False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name)

    @classmethod
    def pretrained(cls, cfg: PipelineConfig) -> "ModelRegistry":
        from fmi_pipeline.models import pretrained as P

        m = cfg.models

        def detector():
            player = P.YoloDetector(m.player_detector, imgsz=cfg.detect.imgsz, device=m.device)
            ball = None
            try:
                ball = P.YoloDetector(m.ball_detector, imgsz=cfg.detect.imgsz, device=m.device, force_class="ball")
            except FileNotFoundError as exc:
                log.warning("ball model missing, using player model ball class: %s", exc)
            return P.CombinedDetector(player, ball, cfg.detect.ball_conf)

        def team_embedder():
            if cfg.team.embedder == "siglip":
                try:
                    return P.SiglipEmbedder(m.siglip, device=m.device)
                except Exception as exc:  # pragma: no cover - depends on env
                    log.warning("SigLIP unavailable (%s); falling back to colour histograms", exc)
            return P.ColorHistEmbedder()

        return cls({
            "detector": detector,
            "keypoints": lambda: P.YoloPitchKeypoints(m.pitch_keypoints, device=m.device),
            "team_embedder": team_embedder,
            "reid": lambda: P.OsnetReID(m.osnet, device=m.device),
            "ocr": lambda: P.ParseqOCR(m.parseq, device=m.device),
        })

    @classmethod
    def fake(cls, scene) -> "ModelRegistry":
        from fmi_pipeline.models import fakes as F

        return cls({
            "detector": lambda: F.FakeDetector(scene),
            "keypoints": lambda: F.FakeKeypoints(scene),
            "team_embedder": lambda: F.FakeTeamEmbedder(scene),
            "reid": lambda: F.FakeReID(scene),
            "ocr": lambda: F.FakeOCR(scene),
        })

    @classmethod
    def for_video(cls, video: str, cfg: PipelineConfig) -> "ModelRegistry":
        if video.startswith("synthetic://"):
            from fmi_pipeline.video import open_video

            return cls.fake(open_video(video).scene)
        return cls.pretrained(cfg)
