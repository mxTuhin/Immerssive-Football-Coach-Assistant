"""Stage 4: online tracker, BoT-SORT style, with gates in pitch metres (not pixels).

Per frame:
  1. predict every track with a constant-velocity Kalman filter in metres
  2. pass 1: confirmed + lost tracks vs strong boxes (score >= high), cost = IoU + metric distance + appearance
  3. pass 2: remaining active tracks vs weak boxes (low <= score < high), IoU + distance only
  4. unmatched strong boxes start new tracks only above ``new_track_thresh`` (~0.6)
  5. tracks unseen for ``max_lost_s`` are closed (offline linking joins them later)
The ball box is inflated for matching only, so a small fast ball still links between frames.
Without a homography for a frame, the distance term falls back to pixels normalised by box height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fmi_pipeline.config import TrackConfig
from fmi_pipeline.models.base import CLASS_IDS, box_iou
from fmi_schema.geom import linear_assignment


@dataclass
class _KF:
    """Constant-velocity Kalman filter on (x, y) in metres."""

    x: np.ndarray                    # [x, y, vx, vy]
    P: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 9.0, 9.0]))

    def predict(self, dt: float, q: float = 8.0) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q

    def update(self, z: np.ndarray, r: float = 0.25) -> None:
        Hm = np.zeros((2, 4))
        Hm[0, 0] = Hm[1, 1] = 1
        S = Hm @ self.P @ Hm.T + np.eye(2) * r
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P


@dataclass
class Track:
    tid: int
    box: np.ndarray
    score: float
    cls_votes: dict
    kf: _KF | None
    emb: np.ndarray | None = None
    hits: int = 1
    last_frame: int = 0
    start_frame: int = 0

    @property
    def pos(self) -> np.ndarray | None:
        return None if self.kf is None else self.kf.x[:2]


def foot_point(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)


def inflate(boxes: np.ndarray, factor: float) -> np.ndarray:
    c = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
    wh = np.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1) * factor / 2
    return np.concatenate([c - wh, c + wh], 1)


class OnlineTracker:
    def __init__(self, cfg: TrackConfig, fps: float):
        self.cfg, self.fps = cfg, fps
        self.dt = 1.0 / fps
        self.tracks: list[Track] = []
        self.next_id = 1
        self.max_lost = int(round(cfg.max_lost_s * fps))

    # ------------------------------------------------------------------ cost
    def _cost(self, frame: int, tracks: list[Track], boxes: np.ndarray, pitch_xy: np.ndarray | None,
              embs: np.ndarray | None, cls: np.ndarray, use_app: bool) -> np.ndarray:
        cfg = self.cfg
        if not tracks or not len(boxes):
            return np.zeros((len(tracks), len(boxes)))
        tb = np.stack([t.box for t in tracks])
        is_ball = cls == CLASS_IDS["ball"]
        db = boxes.copy()
        db[is_ball] = inflate(boxes[is_ball], cfg.ball_inflate)
        t_ball = np.array([max(t.cls_votes, key=t.cls_votes.get) == CLASS_IDS["ball"] for t in tracks])
        tb = tb.copy()
        tb[t_ball] = inflate(tb[t_ball], cfg.ball_inflate)
        iou = box_iou(tb, db)
        cost = cfg.w_iou * (1 - iou)
        if pitch_xy is not None and all(t.pos is not None for t in tracks):
            tp = np.stack([t.pos for t in tracks])
            d = np.linalg.norm(tp[:, None, :] - pitch_xy[None, :, :], axis=2)
            # the gate widens with time unseen: a player can cover <= 10 m/s
            lost_s = np.array([(frame - t.last_frame - 1) * self.dt for t in tracks])
            gate = (cfg.gate_m + 10.0 * np.clip(lost_s, 0, None))[:, None]
            dist_term = np.clip(d / gate, 0, 1.5)
            cost = cost + cfg.w_dist * dist_term
            cost[d > gate * 1.5] = np.inf
        else:
            fp_t, fp_d = foot_point(tb), foot_point(boxes)
            h = np.maximum(boxes[:, 3] - boxes[:, 1], 10)
            d = np.linalg.norm(fp_t[:, None, :] - fp_d[None, :, :], axis=2) / h[None, :]
            cost = cost + cfg.w_dist * np.clip(d, 0, 1.5)
            cost[d > 1.5] = np.inf
        if use_app and embs is not None:
            te = [t.emb for t in tracks]
            if all(e is not None for e in te):
                sim = np.stack(te) @ embs.T
                cost = cost + cfg.w_app * (1 - np.clip(sim, -1, 1))
        # never match ball <-> person
        cost[t_ball[:, None] != is_ball[None, :]] = np.inf
        return cost

    # ------------------------------------------------------------------ step
    def update(self, frame: int, boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray,
               pitch_xy: np.ndarray | None, embs: np.ndarray | None = None) -> list[tuple[int, int]]:
        """Associate this frame's detections. Returns [(det_index, track_id)] for matched/new detections."""
        cfg = self.cfg
        for t in self.tracks:
            if t.kf is not None:
                t.kf.predict(self.dt)
        active = [t for t in self.tracks if frame - t.last_frame <= self.max_lost]
        high = np.nonzero(scores >= cfg.high_thresh)[0]
        low = np.nonzero((scores >= cfg.low_thresh) & (scores < cfg.high_thresh))[0]
        results: list[tuple[int, int]] = []

        def assign(tracks, det_idx, use_app, thresh):
            if not tracks or not len(det_idx):
                return [], list(range(len(tracks))), list(det_idx)
            pxy = pitch_xy[det_idx] if pitch_xy is not None else None
            e = embs[det_idx] if embs is not None else None
            c = self._cost(frame, tracks, boxes[det_idx], pxy, e, cls[det_idx], use_app)
            pairs = linear_assignment(c, max_cost=thresh)
            mt = {r for r, _ in pairs}
            md = {k for _, k in pairs}
            return ([(tracks[r], det_idx[k]) for r, k in pairs],
                    [i for i in range(len(tracks)) if i not in mt], [det_idx[k] for k in range(len(det_idx)) if k not in md])

        m1, un_t, un_d = assign(active, high, cfg.use_appearance, cfg.match_thresh)
        remaining = [active[i] for i in un_t if frame - active[i].last_frame <= 1]
        m2, _, _ = assign(remaining, low, False, cfg.second_match_thresh)
        for t, d in m1 + m2:
            self._update_track(t, frame, d, boxes, scores, cls, pitch_xy, embs)
            results.append((int(d), t.tid))
        for d in un_d:
            if scores[d] >= cfg.new_track_thresh:
                t = self._new_track(frame, d, boxes, scores, cls, pitch_xy, embs)
                results.append((int(d), t.tid))
        # drop long-dead tracks from memory
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_lost]
        return results

    def _new_track(self, frame, d, boxes, scores, cls, pitch_xy, embs) -> Track:
        kf = None
        if pitch_xy is not None and np.isfinite(pitch_xy[d]).all():
            kf = _KF(np.array([pitch_xy[d, 0], pitch_xy[d, 1], 0.0, 0.0]))
        t = Track(tid=self.next_id, box=boxes[d].copy(), score=float(scores[d]), cls_votes={int(cls[d]): 1},
                  kf=kf, emb=None if embs is None else embs[d].copy(), last_frame=frame, start_frame=frame)
        self.next_id += 1
        self.tracks.append(t)
        return t

    def _update_track(self, t: Track, frame, d, boxes, scores, cls, pitch_xy, embs) -> None:
        t.box = boxes[d].copy()
        t.score = 0.8 * t.score + 0.2 * float(scores[d])
        t.cls_votes[int(cls[d])] = t.cls_votes.get(int(cls[d]), 0) + 1
        t.hits += 1
        t.last_frame = frame
        if pitch_xy is not None and np.isfinite(pitch_xy[d]).all():
            if t.kf is None:
                t.kf = _KF(np.array([pitch_xy[d, 0], pitch_xy[d, 1], 0.0, 0.0]))
            else:
                t.kf.update(pitch_xy[d])
        if embs is not None and np.isfinite(embs[d]).all():
            e = embs[d] if t.emb is None else 0.9 * t.emb + 0.1 * embs[d]
            t.emb = e / max(np.linalg.norm(e), 1e-9)
