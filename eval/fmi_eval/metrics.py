"""Tracking metrics against ground truth in pitch metres (build plan section 8).

Inputs are long tables with columns ``frame, id, x, y`` (pred and truth on the same 10 Hz frame grid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_schema.geom import linear_assignment


def frame_matches(pred: pd.DataFrame, truth: pd.DataFrame, max_dist: float = 2.0) -> pd.DataFrame:
    """Per-frame Hungarian matching on distance. Returns rows: frame, gt_id, pred_id, dist."""
    rows = []
    pg = {f: g for f, g in pred.groupby("frame")}
    for f, g in truth.groupby("frame"):
        p = pg.get(f)
        if p is None or p.empty:
            continue
        d = np.linalg.norm(g[["x", "y"]].to_numpy(float)[:, None] - p[["x", "y"]].to_numpy(float)[None], axis=2)
        d = np.where(np.isfinite(d), d, np.inf)
        for i, j in linear_assignment(d, max_cost=max_dist):
            rows.append((f, g["id"].iloc[i], p["id"].iloc[j], d[i, j]))
    return pd.DataFrame(rows, columns=["frame", "gt_id", "pred_id", "dist"])


def position_error(matches: pd.DataFrame) -> dict:
    if matches.empty:
        return {"median_m": None, "mean_m": None, "p90_m": None, "n": 0}
    d = matches["dist"].to_numpy()
    return {"median_m": float(np.median(d)), "mean_m": float(d.mean()), "p90_m": float(np.percentile(d, 90)),
            "n": int(len(d))}


def id_metrics(pred: pd.DataFrame, truth: pd.DataFrame, matches: pd.DataFrame, fps: float = 10.0) -> dict:
    """IDF1 (global id mapping), ID switches per 10 minutes, recall/precision of detections."""
    n_gt, n_pred = len(truth), len(pred)
    if matches.empty:
        return {"idf1": 0.0, "id_switches": 0, "id_switches_per_10min": 0.0, "recall": 0.0, "precision": 0.0}
    co = matches.groupby(["gt_id", "pred_id"]).size().unstack(fill_value=0)
    pairs = linear_assignment(-co.to_numpy(float))
    idtp = int(sum(co.to_numpy()[i, j] for i, j in pairs))
    idf1 = 2 * idtp / max(n_gt + n_pred, 1)
    sw = 0
    for _, g in matches.sort_values("frame").groupby("gt_id"):
        ids = g["pred_id"].to_numpy()
        sw += int((ids[1:] != ids[:-1]).sum())
    minutes = (truth["frame"].max() - truth["frame"].min() + 1) / fps / 60.0
    return {"idf1": float(idf1), "id_switches": sw, "id_switches_per_10min": float(sw / max(minutes, 1e-9) * 10),
            "recall": float(len(matches) / max(n_gt, 1)), "precision": float(len(matches) / max(n_pred, 1))}


def identity_accuracy(matches: pd.DataFrame, gt_to_player: dict[str, str]) -> dict:
    """% of matched player-frames whose predicted id is the right squad player."""
    m = matches[matches["gt_id"].isin(gt_to_player)]
    if m.empty:
        return {"player_frames_correct": None, "n": 0}
    ok = m["pred_id"].astype(str).to_numpy() == m["gt_id"].map(gt_to_player).astype(str).to_numpy()
    return {"player_frames_correct": float(ok.mean()), "n": int(len(m))}


def ball_within(pred_ball: pd.DataFrame, truth_ball: pd.DataFrame, radius: float = 2.0) -> dict:
    j = truth_ball.merge(pred_ball, on="frame", suffixes=("_gt", "_p"))
    if j.empty:
        return {"within_2m": None, "coverage": 0.0}
    d = np.hypot(j["x_gt"] - j["x_p"], j["y_gt"] - j["y_p"])
    return {"within_2m": float(np.nanmean(d <= radius)), "coverage": float(np.isfinite(d).sum() / len(truth_ball)),
            "median_m": float(np.nanmedian(d))}


def evaluate_tracking(pred: pd.DataFrame, truth: pd.DataFrame, gt_to_player: dict[str, str] | None = None,
                      fps: float = 10.0) -> dict:
    pp = pred[pred["id"] != "ball"]
    tp = truth[truth["id"] != "ball"]
    common = sorted(set(pp["frame"]) & set(tp["frame"]))
    pp, tp = pp[pp["frame"].isin(common)], tp[tp["frame"].isin(common)]
    m = frame_matches(pp, tp)
    out = {"frames": len(common), "position": position_error(m), "ids": id_metrics(pp, tp, m, fps)}
    if gt_to_player:
        out["identity"] = identity_accuracy(m, gt_to_player)
    tb, pb = truth[truth["id"] == "ball"], pred[pred["id"] == "ball"]
    if len(tb):
        out["ball"] = ball_within(pb[["frame", "x", "y"]], tb[["frame", "x", "y"]])
    return out
