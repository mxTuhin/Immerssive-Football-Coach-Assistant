"""Helpers shared by stages."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fmi_pipeline.contract import StageContext, StageOutput
from fmi_pipeline.geometry import apply_h, row_to_h
from fmi_pipeline.video import open_video
from fmi_schema.io import read_table, write_table


def video_meta(ingest: StageOutput) -> dict:
    return json.loads(ingest.path("video_meta").read_text())


def proxy(ingest: StageOutput):
    vm = video_meta(ingest)
    src = vm["proxy"]
    return open_video(src if src.startswith("synthetic://") else ingest.dir / src)


def save(ctx: StageContext, df: pd.DataFrame, name: str) -> str:
    p = write_table(df, ctx.out_dir / f"{name}.parquet")
    return p.name


def load(out: StageOutput, name: str) -> pd.DataFrame:
    return read_table(out.path(name))


def homographies(calib: StageOutput) -> dict[int, np.ndarray]:
    df = load(calib, "calibration")
    ok = df[np.isfinite(df["h00"])]
    return {int(r["frame"]): row_to_h(r) for _, r in ok.iterrows()}


def project_points(H: dict[int, np.ndarray], frames: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Project image points (N,2) at ``frames`` to pitch metres; NaN where no homography."""
    out = np.full((len(frames), 2), np.nan)
    frames = np.asarray(frames)
    for f in np.unique(frames):
        h = H.get(int(f))
        if h is None:
            continue
        m = frames == f
        out[m] = apply_h(h, uv[m])
    return out


def sample_evenly(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def crops_for_tracks(video, tracks: pd.DataFrame, per_track: int, region=(0.0, 1.0),
                     min_score: float = 0.4) -> dict[str, list]:
    """Read sampled crops for each track (sequential frame access, one pass over the video)."""
    from fmi_pipeline.video import crop

    want: dict[int, list[tuple[str, tuple]]] = {}
    for tid, g in tracks.groupby("track_id", sort=False):
        g = g[g["score"] >= min_score] if (g["score"] >= min_score).any() else g
        rows = sample_evenly(list(g.itertuples(index=False)), per_track)
        for r in rows:
            want.setdefault(int(r.frame), []).append((str(tid), (r.x1, r.y1, r.x2, r.y2)))
    frames = video.read(sorted(want))
    out: dict[str, list] = {}
    for f, items in want.items():
        img = frames.get(f)
        if img is None:
            continue
        for tid, box in items:
            out.setdefault(tid, []).append(crop(img, box, region=region))
    return out
