"""Stage 4: online tracking of people (players, goalkeepers, referees) in pitch metres."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.models.base import CLASS_IDS
from fmi_pipeline.stages._util import homographies, load, proxy, save, video_meta
from fmi_pipeline.geometry import apply_h
from fmi_pipeline.tracking.tracker import OnlineTracker, foot_point
from fmi_pipeline.video import crop


class Track(Stage):
    name = "track"
    version = "1"
    deps = ("ingest", "detect", "calib")
    config_sections = ("track",)

    def run(self, ctx: StageContext, inputs) -> StageResult:
        cfg = ctx.config.track
        vm = video_meta(inputs["ingest"])
        det = load(inputs["detect"], "detections")
        det = det[det["cls"] != CLASS_IDS["ball"]].sort_values("frame")
        H = homographies(inputs["calib"])
        reid = ctx.models.reid if (cfg.use_appearance and ctx.models.has("reid")) else None
        tracker = OnlineTracker(cfg, vm["fps"])
        by_frame = {int(f): g for f, g in det.groupby("frame")}
        video = proxy(inputs["ingest"]) if reid is not None else None
        frame_iter = video.iter_frames() if video is not None else ((f, None) for f in range(vm["n_frames"]))
        rows = []
        emb_sum: dict[int, np.ndarray] = {}
        emb_n: dict[int, int] = {}
        n = vm["n_frames"]
        for f, img in frame_iter:
            g = by_frame.get(int(f))
            if g is None:
                tracker.update(f, np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None)
                continue
            boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(float)
            scores = g["score"].to_numpy(float)
            cls = g["cls"].to_numpy(int)
            h = H.get(int(f))
            pxy = apply_h(h, foot_point(boxes)) if h is not None else None
            embs = None
            if reid is not None and img is not None and f % cfg.appearance_every == 0:
                strong = scores >= cfg.high_thresh
                embs = np.full((len(boxes), reid.dim), np.nan)
                if strong.any():
                    e = reid.embed([crop(img, b) for b in boxes[strong]])
                    embs[strong] = e
            matches = tracker.update(f, boxes, scores, cls, pxy, embs)
            for d, tid in matches:
                rows.append((int(f), f"t{tid}", *boxes[d], float(scores[d]), int(cls[d]),
                             float(pxy[d, 0]) if pxy is not None else np.nan,
                             float(pxy[d, 1]) if pxy is not None else np.nan))
                if embs is not None and np.isfinite(embs[d]).all():
                    emb_sum[tid] = emb_sum.get(tid, 0) + embs[d]
                    emb_n[tid] = emb_n.get(tid, 0) + 1
            if f % 250 == 0:
                ctx.report(self.name, f / max(n, 1), f"frame {f}/{n}")
        tracks = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2", "score", "cls", "px", "py"])
        np.savez_compressed(ctx.out_dir / "embeddings.npz",
                            **{f"t{k}": v / max(np.linalg.norm(v), 1e-9) for k, v in emb_sum.items()})
        lengths = tracks.groupby("track_id").size() if len(tracks) else pd.Series(dtype=int)
        qa = self.qa(tracklets=int(tracks["track_id"].nunique()) if len(tracks) else 0,
                     median_len_frames=float(lengths.median()) if len(lengths) else 0.0,
                     detections_tracked_frac=float(len(tracks) / max(len(det), 1)),
                     appearance=bool(reid is not None),
                     projected_frac=float(np.isfinite(tracks["px"]).mean()) if len(tracks) else 0.0)
        return StageResult({"tracklets": save(ctx, tracks, "tracklets"), "embeddings": "embeddings.npz"}, qa)
