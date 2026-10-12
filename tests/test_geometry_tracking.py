import numpy as np
import pandas as pd

from fmi_pipeline.config import BallConfig, LinkConfig, TrackConfig
from fmi_pipeline.geometry import apply_h, dlt, fit_homography, leave_one_out_error
from fmi_pipeline.models.base import CLASS_IDS, Detections, box_iou, detect_tiled, nms, tiles
from fmi_pipeline.tracking.ball_path import ball_track
from fmi_pipeline.tracking.linking import link
from fmi_pipeline.tracking.tracker import OnlineTracker


def test_dlt_and_ransac_recover_homography():
    H = np.array([[2.0, 0.3, 100], [0.1, 1.5, 50], [0.0005, 0.0002, 1]])
    src = np.random.default_rng(0).uniform(-50, 50, (30, 2))
    dst = apply_h(H, src)
    assert np.allclose(dlt(src, dst), H, atol=1e-6)
    dst_noisy = dst.copy()
    dst_noisy[:5] += 300  # outliers
    H2, inl = fit_homography(src, dst_noisy, thresh=3.0)
    assert inl[5:].all() and not inl[:5].any()
    assert leave_one_out_error(src[5:], dst[5:]) < 1e-6


def test_tiling_and_nms():
    t = tiles(3840, 2160, 1280, 0.2)
    assert t[0] == (0, 0, 1280, 1280) and t[-1][2] == 3840 and t[-1][3] == 2160

    class Det:
        def detect(self, frame, conf=0.1):
            return Detections(np.array([[10.0, 10, 40, 80]]), np.array([0.9]), np.array([2]))

    d = detect_tiled(Det(), np.zeros((2160, 3840, 3), np.uint8), 0.1, 1280, 0.2)
    assert len(d) >= 1
    two = Detections(np.array([[0, 0, 10, 10], [1, 1, 11, 11.0]]), np.array([0.9, 0.8]), np.array([2, 2]))
    assert len(nms(two, 0.5)) == 1
    assert box_iou(two.xyxy, two.xyxy).diagonal().min() == 1.0


def _walkers(T=60, n=4):
    return {k: np.stack([np.linspace(-20 + 10 * k, -10 + 10 * k, T), np.full(T, 5.0 * k)], 1) for k in range(n)}


def test_tracker_keeps_ids_in_metres():
    paths = _walkers()
    tr = OnlineTracker(TrackConfig(), fps=10)
    ids = {k: set() for k in paths}
    for f in range(60):
        xy = np.stack([paths[k][f] for k in paths])
        boxes = np.stack([[100 * k + f, 100, 100 * k + f + 30, 180] for k in paths]).astype(float)
        scores = np.full(len(paths), 0.9)
        if f == 30:
            scores[1] = 0.3  # weak box: must still be matched in the second pass
        res = tr.update(f, boxes, scores, np.full(len(paths), CLASS_IDS["player"]), xy)
        for d, tid in res:
            ids[list(paths)[d]].add(tid)
    assert all(len(v) == 1 for v in ids.values())


def _tracklets_df(pieces):
    rows = []
    for tid, (f0, xy) in pieces.items():
        for i, (x, y) in enumerate(xy):
            rows.append({"frame": f0 + i, "track_id": tid, "px": x, "py": y, "cls": 2, "score": 0.9})
    return pd.DataFrame(rows)


def test_linking_joins_feasible_gaps_only():
    a = np.stack([np.linspace(0, 5, 30), np.zeros(30)], 1)
    b = np.stack([np.linspace(6, 10, 30), np.zeros(30)], 1)          # continues a after 1 s
    c = np.stack([np.linspace(40, 45, 30), np.zeros(30)], 1)         # 35 m away after 1 s: infeasible
    df = _tracklets_df({"t1": (0, a), "t2": (40, b), "t3": (40, c)})
    mapping, qa = link(df, {}, LinkConfig(), fps=10)
    assert mapping["t1"] == mapping["t2"] != mapping["t3"]
    forced, _ = link(df, {}, LinkConfig(), fps=10, merges=[{"track_ids": ["t1", "t3"]}])
    assert forced["t1"] == forced["t3"]


def test_ball_path_ignores_distractors():
    rng = np.random.default_rng(2)
    T = 100
    true = np.stack([np.linspace(-30, 30, T), np.zeros(T)], 1)
    rows = []
    for f in range(T):
        if f % 7 != 3:
            rows.append({"frame": f, "px": true[f, 0], "py": true[f, 1], "score": 0.6})
        rows.append({"frame": f, "px": rng.uniform(-50, 50), "py": rng.uniform(-30, 30), "score": 0.3})
    out = ball_track(pd.DataFrame(rows), 10.0, BallConfig(), (0, T - 1))
    err = np.hypot(out["x"] - true[:, 0], out["y"] - true[:, 1])
    assert np.nanmedian(err) < 0.5
    assert out["estimated"].sum() >= 10  # the missing frames were interpolated and flagged
