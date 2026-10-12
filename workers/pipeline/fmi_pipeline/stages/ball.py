"""Stage 8: ball. Low-threshold candidates -> pitch metres -> one path by DP -> gap interpolation."""

from __future__ import annotations

import numpy as np

from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.models.base import CLASS_IDS
from fmi_pipeline.stages._util import homographies, load, project_points, save, video_meta
from fmi_pipeline.tracking.ball_path import ball_track


class Ball(Stage):
    name = "ball"
    version = "1"
    deps = ("ingest", "detect", "calib")
    config_sections = ("ball",)

    def run(self, ctx: StageContext, inputs) -> StageResult:
        cfg = ctx.config.ball
        vm = video_meta(inputs["ingest"])
        det = load(inputs["detect"], "detections")
        c = det[det["cls"] == CLASS_IDS["ball"]].copy()
        H = homographies(inputs["calib"])
        if len(c):
            # bottom-centre of the box = ground contact for a ball on the ground
            uv = np.stack([((c["x1"] + c["x2"]) / 2).to_numpy(), c["y2"].to_numpy()], 1)
            xy = project_points(H, c["frame"].to_numpy(), uv)
            c["px"], c["py"] = xy[:, 0], xy[:, 1]
        else:
            c["px"], c["py"] = [], []
        ball = ball_track(c[["frame", "px", "py", "score"]], vm["fps"], cfg, (0, vm["n_frames"] - 1))
        qa = self.qa(candidates=int(len(c)), observed_frac=float((~ball["estimated"]).mean()),
                     available_frac=float(np.isfinite(ball["x"]).mean()), in_air_frac=float(ball["in_air"].mean()))
        if qa.metrics["available_frac"] < 0.5:
            qa.warnings.append("ball position unavailable for more than half of the frames")
        return StageResult({"ball": save(ctx, ball, "ball")}, qa)
