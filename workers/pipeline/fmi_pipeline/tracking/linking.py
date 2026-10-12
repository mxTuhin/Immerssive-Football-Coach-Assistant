"""Stage 5: offline tracklet linking by global assignment (not greedy stitching).

Every tracklet end is a row and every tracklet start a column of one cost matrix; a minimum-cost bipartite
matching over the whole match is a minimum-cost path cover, so one bad early link cannot cascade.

Football rules in the cost:
  * feasible speed across the gap: distance <= 10 m/s * gap + slack, else forbidden
  * no temporal overlap (a link always goes from an end to a later start)
  * class conflicts (ball/referee/player) forbidden; known team conflicts forbidden
The "<= 11 per team on the pitch" rule is checked after team assignment (team stage) and sent to review.
Human ``merge`` corrections force links; ``split`` corrections cut tracklets before linking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmi_pipeline.config import LinkConfig
from fmi_schema.geom import linear_assignment


@dataclass
class Tracklet:
    tid: str
    start: int
    end: int
    p_start: np.ndarray
    p_end: np.ndarray
    v_start: np.ndarray
    v_end: np.ndarray
    emb: np.ndarray | None
    cls: int
    team: str | None = None


def apply_splits(tracks: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    """Cut ``track_id`` at ``frame``: rows from that frame on get a new id ``<id>s<frame>``."""
    tracks = tracks.copy()
    for s in splits or []:
        tid, f = str(s["track_id"]), int(s["frame"])
        m = (tracks["track_id"] == tid) & (tracks["frame"] >= f)
        tracks.loc[m, "track_id"] = f"{tid}s{f}"
    return tracks


def summarize(tracks: pd.DataFrame, embs: dict[str, np.ndarray], fps: float) -> list[Tracklet]:
    out = []
    w = max(2, int(fps))  # 1 s window for velocity
    for tid, g in tracks.groupby("track_id", sort=False):
        g = g.sort_values("frame")
        xy = g[["px", "py"]].to_numpy(float)
        fr = g["frame"].to_numpy()
        ok = np.isfinite(xy).all(1)
        if ok.sum() == 0:
            continue
        xy_ok, fr_ok = xy[ok], fr[ok]

        def vel(sl):
            if len(sl) < 2:
                return np.zeros(2)
            a, b = sl[0], sl[-1]
            dt = (fr_ok[b] - fr_ok[a]) / fps
            return (xy_ok[b] - xy_ok[a]) / dt if dt > 0 else np.zeros(2)

        idx = np.arange(len(fr_ok))
        cls = int(g["cls"].mode().iloc[0])
        out.append(Tracklet(tid=str(tid), start=int(fr[0]), end=int(fr[-1]), p_start=xy_ok[0], p_end=xy_ok[-1],
                            v_start=vel(idx[:w]), v_end=vel(idx[-w:]), emb=embs.get(str(tid)), cls=cls,
                            team=g["team"].mode().iloc[0] if "team" in g and g["team"].notna().any() else None))
    return out


def link_cost_matrix(tl: list[Tracklet], cfg: LinkConfig, fps: float) -> np.ndarray:
    """Vectorised end(i) -> start(j) costs; forbidden links are +inf."""
    n = len(tl)
    if n == 0:
        return np.zeros((0, 0))
    start = np.array([t.start for t in tl])
    end = np.array([t.end for t in tl])
    p_end = np.stack([t.p_end for t in tl])
    v_end = np.stack([t.v_end for t in tl])
    p_start = np.stack([t.p_start for t in tl])
    cls = np.array([t.cls for t in tl])
    team = np.array([t.team or "" for t in tl], dtype=object)
    gap = start[None, :] - end[:, None]
    gap_s = gap / fps
    ok = (gap > 0) & (gap <= cfg.max_gap_s * fps) & (cls[:, None] == cls[None, :])
    ok &= ~((team[:, None] != "") & (team[None, :] != "") & (team[:, None] != team[None, :]))
    pred = p_end[:, None, :] + v_end[:, None, :] * np.clip(gap_s, 0, 1.0)[:, :, None]
    d = np.linalg.norm(p_start[None, :, :] - pred, axis=2)
    d_raw = np.linalg.norm(p_start[None, :, :] - p_end[:, None, :], axis=2)
    feasible = cfg.max_speed * np.clip(gap_s, 0, None) + cfg.slack_m
    ok &= d_raw <= feasible
    cost = cfg.w_dist * np.minimum(d, d_raw) / feasible + cfg.w_gap * gap_s / cfg.max_gap_s
    has_emb = np.array([t.emb is not None for t in tl])
    app = np.full((n, n), 0.5)
    if has_emb.any():
        dim = next(t.emb for t in tl if t.emb is not None).shape[0]
        E = np.stack([t.emb if t.emb is not None else np.zeros(dim) for t in tl])
        sim = np.clip(E @ E.T, -1, 1)
        both = has_emb[:, None] & has_emb[None, :]
        app = np.where(both, 1 - sim, 0.5)
    cost = cost + cfg.w_app * app
    return np.where(ok, cost, np.inf)


def link(tracks: pd.DataFrame, embs: dict[str, np.ndarray], cfg: LinkConfig, fps: float,
         merges: list[dict] | None = None) -> tuple[dict[str, str], dict]:
    """Return ({tracklet_id: linked_id}, qa metrics)."""
    tl = summarize(tracks, embs, fps)
    min_len = cfg.min_len_s * fps
    keep = [t for t in tl if t.end - t.start + 1 >= min_len]
    dropped = {t.tid for t in tl} - {t.tid for t in keep}
    idx = {t.tid: i for i, t in enumerate(keep)}
    C = link_cost_matrix(keep, cfg, fps)
    # forced merges: give the requested consecutive pairs a zero cost (ordered by start time)
    forced = 0
    for m in merges or []:
        ids = sorted([str(t) for t in m.get("track_ids", []) if str(t) in idx], key=lambda t: keep[idx[t]].start)
        for a, b in zip(ids, ids[1:]):
            C[idx[a], :] = np.where(np.arange(len(keep)) == idx[b], 0.0, np.inf)
            C[:, idx[b]] = np.where(np.arange(len(keep)) == idx[a], 0.0, np.inf)
            forced += 1
    pairs = linear_assignment(C, max_cost=cfg.link_thresh)
    nxt = {keep[i].tid: keep[j].tid for i, j in pairs}
    prev = {v: k for k, v in nxt.items()}
    mapping: dict[str, str] = {}
    n_chains = 0
    for t in sorted(keep, key=lambda t: t.start):
        if t.tid in prev:
            continue
        n_chains += 1
        chain_id = f"L{n_chains}"
        cur = t.tid
        while cur is not None:
            mapping[cur] = chain_id
            cur = nxt.get(cur)
    qa = {"tracklets": len(tl), "tracklets_dropped_short": len(dropped), "links": len(pairs),
          "forced_merges": forced, "long_tracks": n_chains}
    return mapping, qa
