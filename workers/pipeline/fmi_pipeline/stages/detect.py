"""Stage 2: detection of players, goalkeepers, referees and the ball (pretrained, tiled for 4K)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.models.base import CLASS_IDS, detect_tiled
from fmi_pipeline.stages._util import proxy, save, video_meta


class Detect(Stage):
    name = "detect"
    version = "1"
    deps = ("ingest",)
    config_sections = ("detect",)

    def run(self, ctx: StageContext, inputs) -> StageResult:
        cfg = ctx.config.detect
        vm = video_meta(inputs["ingest"])
        video = proxy(inputs["ingest"])
        detector = ctx.models.detector
        tiling = cfg.tiling == "on" or (cfg.tiling == "auto" and vm["width"] > 2500)
        rows = []
        n = vm["n_frames"]
        for i, (f, frame) in enumerate(video.iter_frames(stride=cfg.stride)):
            if tiling:
                d = detect_tiled(detector, frame, cfg.player_conf, cfg.tile_size, cfg.tile_overlap, cfg.nms_iou)
            else:
                d = detector.detect(frame, conf=min(cfg.player_conf, cfg.ball_conf))
            if len(d):
                keep = np.where(d.cls == CLASS_IDS["ball"], d.score >= cfg.ball_conf, d.score >= cfg.player_conf)
                d = d.select(keep)
                rows.append(pd.DataFrame({"frame": f, "x1": d.xyxy[:, 0], "y1": d.xyxy[:, 1], "x2": d.xyxy[:, 2],
                                          "y2": d.xyxy[:, 3], "score": d.score, "cls": d.cls}))
            if i % 250 == 0:
                ctx.report(self.name, f / max(n, 1), f"frame {f}/{n}")
        det = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["frame", "x1", "y1", "x2", "y2", "score", "cls"])
        persons = det[det["cls"] != CLASS_IDS["ball"]]
        per_frame = persons[persons["score"] >= 0.5].groupby("frame").size()
        qa = self.qa(frames=int(n), tiling=bool(tiling), detections=int(len(det)),
                     mean_players_per_frame=float(per_frame.mean()) if len(per_frame) else 0.0,
                     frames_with_ball=int(det[det["cls"] == CLASS_IDS["ball"]]["frame"].nunique()))
        if len(per_frame) and per_frame.mean() < 10:
            qa.warnings.append("low player recall (< 10 confident players per frame on average)")
        return StageResult({"detections": save(ctx, det, "detections")}, qa)
