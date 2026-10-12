"""Stage 3: calibration. Pretrained 32-point pitch keypoints -> RANSAC homography per keyframe.

* static camera: solve on a sample of frames, take the element-wise median, verify on all samples
* moving camera: keyframes every N frames, median-smoothed and interpolated within each shot segment
* every solution is checked against painted lines it was not fitted to (line score) and by held-out
  keypoint reprojection; a segment failing the check produces ONE review item (not one per frame)
* manual 4-point corrections from the review UI override the automatic solution around their frame
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.geometry import dlt, fit_homography, h_to_row, leave_one_out_error, line_score
from fmi_pipeline.stages._util import load, proxy, save, video_meta
from fmi_schema.models import ReviewItem
from fmi_schema.pitch import PitchDims


def _gray(frame) -> np.ndarray | None:
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = a.astype(np.uint8)
    return a if a.std() > 0 else None


def solve_frame(frame, kp_model, pitch: PitchDims, cfg) -> dict | None:
    kp, conf = kp_model.predict(frame)
    ok = (conf >= cfg.kp_conf) & np.isfinite(kp).all(1)
    if ok.sum() < cfg.min_points:
        return None
    pitch_pts = pitch.keypoints()[ok]
    img_pts = kp[ok]
    H_p2i, inl = fit_homography(pitch_pts, img_pts, cfg.ransac_px)
    if H_p2i is None or inl.sum() < cfg.min_points:
        return None
    try:
        H = np.linalg.inv(H_p2i)
    except np.linalg.LinAlgError:
        return None
    H /= H[2, 2]
    loo = leave_one_out_error(pitch_pts[inl], img_pts[inl])
    ls = line_score(_gray(frame), H, pitch)
    # without texture (synthetic frames) score from held-out reprojection instead
    score = ls if np.isfinite(ls) else float(np.exp(-(loo if np.isfinite(loo) else 20.0) / 10.0))
    return {"H": H, "score": score, "reproj_px": loo, "inliers": int(inl.sum())}


def _median_h(Hs: list[np.ndarray]) -> np.ndarray:
    A = np.stack([h / h[2, 2] for h in Hs])
    return np.median(A, axis=0)


class Calibrate(Stage):
    name = "calib"
    version = "1"
    deps = ("ingest",)
    corrections = ("calibration_points",)
    config_sections = ("calib",)
    uses_source = True  # pitch size + camera hint from the match input

    def run(self, ctx: StageContext, inputs) -> StageResult:
        cfg = ctx.config.calib
        mi = ctx.match_input
        pitch = PitchDims(mi.pitch_length, mi.pitch_width)
        vm = video_meta(inputs["ingest"])
        segments = load(inputs["ingest"], "segments")
        n = vm["n_frames"]
        camera = cfg.camera
        if camera == "auto":
            camera = "static" if mi.video_tier in ("panoramic", "static") else "moving"
        video = proxy(inputs["ingest"])
        kp_model = ctx.models.keypoints
        qa = self.qa(camera=camera)

        if camera == "static":
            sample = sorted(set(np.linspace(0, n - 1, min(cfg.static_samples, n)).round().astype(int)))
        else:
            sample = list(range(0, n, cfg.keyframe_every))
        frames = video.read(sample)
        sols: dict[int, dict] = {}
        for i, f in enumerate(sample):
            if f in frames:
                s = solve_frame(frames[f], kp_model, pitch, cfg)
                if s is not None:
                    sols[f] = s
            if i % 50 == 0:
                ctx.report(self.name, i / max(len(sample), 1), f"keyframe {i}/{len(sample)}")

        # manual corrections become authoritative keyframes
        manual: dict[int, dict] = {}
        for c in ctx.corrections.get("calibration_points", []) or []:
            H = dlt(np.asarray(c["image"], float), np.asarray(c["pitch"], float))
            if H is not None:
                manual[int(c["frame"])] = {"H": H, "score": 1.0, "reproj_px": 0.0, "inliers": len(c["image"])}

        rows = []
        review = []
        Hcols = [f"h{i}{j}" for i in range(3) for j in range(3)]
        if camera == "static":
            good = [s for s in sols.values() if s["score"] >= cfg.line_score_ok] or list(sols.values())
            if manual:
                H, src, score = _median_h([m["H"] for m in manual.values()]), "manual", 1.0
                reproj = 0.0
            elif good:
                H, src = _median_h([s["H"] for s in good]), "static"
                score = float(np.median([s["score"] for s in good]))
                reproj = float(np.nanmedian([s["reproj_px"] for s in good]))
            else:
                H, src, score, reproj = None, "none", 0.0, np.nan
            row = h_to_row(H) if H is not None else {c: np.nan for c in Hcols}
            calib = pd.DataFrame({"frame": np.arange(n)})
            for c in Hcols:
                calib[c] = row[c]
            calib["score"], calib["reproj_px"], calib["source"] = score, reproj, src
            if score < cfg.line_score_ok:
                review.append(ReviewItem(item_id="calib-static", stage=self.name, kind="calibration", priority=0.1,
                                         frame=int(sample[len(sample) // 2]) if sample else 0,
                                         message=f"static calibration score {score:.2f} < {cfg.line_score_ok}; "
                                                 "click 4 pitch points"))
        else:
            for seg in segments.itertuples(index=False):
                fr = np.arange(seg.start_frame, seg.end_frame + 1)
                keys = sorted(k for k in list(sols) + list(manual) if seg.start_frame <= k <= seg.end_frame)
                if not keys:
                    part = pd.DataFrame({"frame": fr, **{c: np.nan for c in Hcols}, "score": 0.0,
                                         "reproj_px": np.nan, "source": "none"})
                    rows.append(part)
                    review.append(ReviewItem(item_id=f"calib-seg{seg.segment}", stage=self.name, kind="calibration",
                                             priority=0.1, frame=int(seg.start_frame),
                                             message="no calibration for this shot; click 4 pitch points"))
                    continue
                K = np.stack([(manual.get(k) or sols[k])["H"] / (manual.get(k) or sols[k])["H"][2, 2] for k in keys])
                sc = np.array([(manual.get(k) or sols[k])["score"] for k in keys])
                rp = np.array([(manual.get(k) or sols[k])["reproj_px"] for k in keys], float)
                is_manual = np.array([k in manual for k in keys])
                # running median over keyframes (manual ones are kept exact)
                w = cfg.smooth_window // 2
                Ks = K.copy()
                for i in range(len(keys)):
                    if is_manual[i]:
                        continue
                    lo, hi = max(0, i - w), min(len(keys), i + w + 1)
                    sel = [j for j in range(lo, hi) if sc[j] >= cfg.line_score_ok * 0.8 or is_manual[j]] or list(range(lo, hi))
                    Ks[i] = np.median(K[sel], axis=0)
                flat = Ks.reshape(len(keys), 9)
                interp = np.stack([np.interp(fr, keys, flat[:, c]) for c in range(9)], 1)
                part = pd.DataFrame(interp, columns=Hcols)
                part.insert(0, "frame", fr)
                part["score"] = np.interp(fr, keys, sc)
                part["reproj_px"] = np.interp(fr, keys, np.nan_to_num(rp, nan=20.0))
                near_manual = np.zeros(len(fr), bool)
                for k in np.array(keys)[is_manual]:
                    near_manual |= np.abs(fr - k) <= cfg.keyframe_every * w
                part["source"] = np.where(np.isin(fr, keys), "keyframe", "interp")
                part.loc[near_manual, "source"] = "manual"
                rows.append(part)
                bad = (sc < cfg.line_score_ok) & ~is_manual
                if bad.mean() > 0.5:
                    review.append(ReviewItem(item_id=f"calib-seg{seg.segment}", stage=self.name, kind="calibration",
                                             priority=0.15, frame=int(keys[int(np.argmin(sc))]),
                                             message=f"calibration line score {np.median(sc):.2f} in this shot; "
                                                     "click 4 pitch points"))
            calib = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["frame", *Hcols, "score"])
        solved = np.isfinite(calib["h00"]) if len(calib) else pd.Series(dtype=bool)
        qa.metrics.update(keyframes=len(sample), keyframes_solved=len(sols), manual_points=len(manual),
                          frames_calibrated_frac=float(solved.mean()) if len(calib) else 0.0,
                          median_score=float(calib["score"].median()) if len(calib) else 0.0,
                          frac_below_ok=float((calib["score"] < cfg.line_score_ok).mean()) if len(calib) else 1.0)
        qa.review = review
        qa.ok = qa.metrics["frames_calibrated_frac"] > 0.5
        return StageResult({"calibration": save(ctx, calib, "calibration")}, qa)
