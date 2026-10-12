"""Stage 5: offline linking of tracklets into long tracks (global assignment + football rules)."""

from __future__ import annotations

import numpy as np

from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._util import load, save, video_meta
from fmi_pipeline.tracking.linking import apply_splits, link


class Link(Stage):
    name = "link"
    version = "1"
    deps = ("ingest", "track")
    corrections = ("split", "merge")
    config_sections = ("link",)

    def run(self, ctx: StageContext, inputs) -> StageResult:
        cfg = ctx.config.link
        fps = video_meta(inputs["ingest"])["fps"]
        tracks = load(inputs["track"], "tracklets")
        tracks["track_id"] = tracks["track_id"].astype(str)
        npz = np.load(inputs["track"].path("embeddings"))
        embs = {k: npz[k] for k in npz.files}
        tracks = apply_splits(tracks, ctx.corrections.get("split"))
        for k in list(embs):  # split pieces inherit their parent's appearance
            for tid in tracks["track_id"].unique():
                if tid.startswith(k + "s"):
                    embs[tid] = embs[k]
        mapping, metrics = link(tracks, embs, cfg, fps, ctx.corrections.get("merge"))
        tracks["tracklet_id"] = tracks["track_id"]
        tracks["track_id"] = tracks["tracklet_id"].map(mapping)
        tracks = tracks.dropna(subset=["track_id"])
        # a long track can momentarily hold two boxes in one frame only if linking is wrong; keep the best
        tracks = tracks.sort_values("score", ascending=False).drop_duplicates(["frame", "track_id"]).sort_values(
            ["track_id", "frame"])
        # long-track appearance = mean of member tracklets
        long_embs = {}
        for lid, members in tracks.groupby("track_id")["tracklet_id"].unique().items():
            vs = [embs[m] for m in members if m in embs]
            if vs:
                v = np.mean(vs, axis=0)
                long_embs[lid] = v / max(np.linalg.norm(v), 1e-9)
        np.savez_compressed(ctx.out_dir / "embeddings.npz", **long_embs)
        qa = self.qa(**metrics)
        return StageResult({"tracks": save(ctx, tracks, "tracks"), "embeddings": "embeddings.npz"}, qa)
