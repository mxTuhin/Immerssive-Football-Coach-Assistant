"""Stage 11: rule-based events from tracking + possession (SPADL-compatible vocabulary).

* possession A -> B, same team, ball travelled           -> pass (success)
* possession A -> B, other team, ball travelled >= 5 m   -> pass (fail) + interception
* possession A -> B, other team, short / close           -> possession_loss + recovery
* release with ball speed > 12 m/s toward the goal, within 30 m of it -> shot (goal if the ball crosses
  the goal line between the posts)
* pass from a wide final-third zone into the box         -> cross
* ball dead (still or out) >= 1.5 s then played          -> set piece typed by location
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign
from fmi_schema.models import EVENT_COLUMNS
from fmi_schema.package import conform
from fmi_schema.pitch import GOAL_WIDTH, PENALTY_BOX_LENGTH, PENALTY_BOX_WIDTH, PitchDims

SPADL_MAP = {
    "pass": "pass", "cross": "cross", "shot": "shot", "interception": "interception", "tackle": "tackle",
    "clearance": "clearance", "dribble": "dribble", "throw_in": "throw_in", "corner": "corner_short",
    "free_kick": "freekick_short", "goal_kick": "goalkick", "foul": "foul", "recovery": "non_action",
    "possession_loss": "bad_touch", "kick_off": "pass", "goal": "non_action",
}


@dataclass
class EventConfig:
    fps: float = 10.0
    pass_min_dist: float = 5.0
    shot_min_speed: float = 12.0
    shot_max_dist: float = 30.0
    dead_ball_s: float = 1.5
    dead_ball_speed: float = 0.5


def _segments(owner: np.ndarray) -> list[tuple[int, int, str]]:
    """Runs of consecutive identical non-null owners: (start_idx, end_idx_inclusive, owner)."""
    segs = []
    start, cur = None, None
    for i, o in enumerate(owner):
        if o != cur:
            if cur is not None:
                segs.append((start, i - 1, cur))
            start, cur = i, o
    if cur is not None:
        segs.append((start, len(owner) - 1, cur))
    return [s for s in segs if s[2] is not None]


def detect_events(tracking: pd.DataFrame, possession: pd.DataFrame, meta, cfg: EventConfig | None = None) -> pd.DataFrame:
    cfg = cfg or EventConfig()
    pitch = PitchDims(meta.pitch_length, meta.pitch_width)
    team_of = tracking.drop_duplicates("id").set_index("id")["team"].astype(str).to_dict()
    pos = tracking.set_index(["frame", "id"])[["x", "y"]]
    out: list[dict] = []

    def player_xy(frame, pid):
        try:
            r = pos.loc[(frame, pid)]
            return float(r["x"]), float(r["y"])
        except KeyError:
            return np.nan, np.nan

    for period, pp in possession.groupby("period", sort=True):
        pp = pp.sort_values("frame").reset_index(drop=True)
        frames = pp["frame"].to_numpy()
        bx, by = pp["ball_x"].to_numpy(dtype=float), pp["ball_y"].to_numpy(dtype=float)
        tclock = pp["t"].to_numpy(dtype=float)
        bvx = np.gradient(bx) * cfg.fps if len(bx) > 1 else np.zeros_like(bx)
        bvy = np.gradient(by) * cfg.fps if len(by) > 1 else np.zeros_like(by)
        bspeed = np.hypot(bvx, bvy)
        segs = _segments(np.array([None if pd.isna(o) else o for o in pp["owner"]], dtype=object))

        def base(i, typ, team, pid, conf):
            return {"period": int(period), "t": float(tclock[i]), "frame": int(frames[i]), "type": typ,
                    "team": team, "player_id": pid, "source": "auto", "confidence": conf}

        # --- passes, interceptions, recoveries, shots
        shot_until = -1
        for k, (s0, e0, a) in enumerate(segs):
            ta = team_of.get(a, "unknown")
            sign = attack_sign(meta, int(period), ta)
            # shot check over the ball flight after A's release
            nxt_start = segs[k + 1][0] if k + 1 < len(segs) else len(frames)
            flight = range(e0, min(nxt_start, e0 + int(3 * cfg.fps)))
            if e0 > shot_until and len(flight) > 1:
                rel_x, rel_y = bx[e0], by[e0]
                gx, gy = pitch.goal_centre(sign)
                dist_goal = np.hypot(gx - rel_x, gy - rel_y)
                f = list(flight)
                vmax_i = f[int(np.nanargmax(bspeed[f]))] if np.isfinite(bspeed[f]).any() else e0
                vx, vy = bvx[vmax_i], bvy[vmax_i]
                toward = (vx * (gx - rel_x) + vy * (gy - rel_y)) / max(np.hypot(vx, vy) * dist_goal, 1e-9)
                if dist_goal <= cfg.shot_max_dist and bspeed[vmax_i] >= cfg.shot_min_speed and toward > 0.9:
                    crossed = [i for i in f if np.isfinite(bx[i]) and sign * bx[i] >= pitch.half_length]
                    goal = bool(crossed) and abs(by[crossed[0]]) <= GOAL_WIDTH / 2 + 0.3
                    ev = base(e0, "shot", ta, a, 0.7 if not crossed else 0.8)
                    ev.update(start_x=rel_x, start_y=rel_y, outcome="success" if goal else "fail",
                              end_x=bx[f[-1]], end_y=by[f[-1]])
                    out.append(ev)
                    if goal:
                        g = base(crossed[0], "goal", ta, a, 0.8)
                        g.update(start_x=bx[crossed[0]], start_y=by[crossed[0]], outcome="success")
                        out.append(g)
                    shot_until = nxt_start
                    continue
            if k + 1 >= len(segs):
                continue
            s1, e1, b = segs[k + 1]
            tb = team_of.get(b, "unknown")
            travel = np.hypot(bx[s1] - bx[e0], by[s1] - by[e0])
            sx, sy = player_xy(frames[e0], a)
            ex, ey = player_xy(frames[s1], b)
            if tb == ta:
                if travel < 2.0:
                    continue  # tight exchange / tracking noise, not a pass
                typ = "pass"
                nsx, nsy = sign * sx, sign * sy
                nex, ney = sign * ex, sign * ey
                wide = abs(nsy) > PENALTY_BOX_WIDTH / 2 and nsx > pitch.half_length / 3
                into_box = nex > pitch.half_length - PENALTY_BOX_LENGTH and abs(ney) < PENALTY_BOX_WIDTH / 2
                if wide and into_box:
                    typ = "cross"
                ev = base(e0, typ, ta, a, 0.75)
                ev.update(receiver_id=b, start_x=sx, start_y=sy, end_x=ex, end_y=ey, outcome="success",
                          t_end=float(tclock[s1]))
                out.append(ev)
            else:
                if travel >= cfg.pass_min_dist:
                    ev = base(e0, "pass", ta, a, 0.6)
                    ev.update(start_x=sx, start_y=sy, end_x=ex, end_y=ey, outcome="fail", t_end=float(tclock[s1]))
                    out.append(ev)
                    it = base(s1, "interception", tb, b, 0.6)
                    it.update(start_x=ex, start_y=ey, outcome="success")
                    out.append(it)
                else:
                    lo = base(e0, "possession_loss", ta, a, 0.55)
                    lo.update(start_x=sx, start_y=sy, outcome="fail")
                    out.append(lo)
                    rc = base(s1, "recovery", tb, b, 0.55)
                    rc.update(start_x=ex, start_y=ey, outcome="success")
                    out.append(rc)

        # --- set pieces: ball dead for >= dead_ball_s, then the next owner plays it
        dead = (bspeed < cfg.dead_ball_speed) | ~pitch.contains(bx, by, margin=0.5)
        min_run = int(cfg.dead_ball_s * cfg.fps)
        i = 0
        n = len(dead)
        seg_starts = {s: o for s, _, o in segs}
        while i < n:
            if dead[i]:
                j = i
                while j < n and dead[j]:
                    j += 1
                if j - i >= min_run and j < n:
                    # first owner at or just before the restart
                    k = next((s for s in range(max(i, j - 5), min(n, j + 10)) if s in seg_starts), None)
                    if k is not None:
                        pid = seg_starts[k]
                        team = team_of.get(pid, "unknown")
                        x, y = bx[min(j, n - 1)], by[min(j, n - 1)]
                        typ = _set_piece_type(x, y, team, attack_sign(meta, int(period), team), pitch)
                        ev = base(k, typ, team, pid, 0.5)
                        ev.update(start_x=x, start_y=y, outcome="unknown")
                        out.append(ev)
                i = j
            else:
                i += 1

    ev = pd.DataFrame(out)
    if ev.empty:
        return conform(ev, EVENT_COLUMNS)
    ev = ev.sort_values(["period", "t"]).reset_index(drop=True)
    ev["event_id"] = [f"a{i}" for i in range(len(ev))]
    if "t_end" not in ev:
        ev["t_end"] = ev["t"]
    ev["t_end"] = ev["t_end"].fillna(ev["t"])
    return conform(ev, EVENT_COLUMNS)


def _set_piece_type(x, y, team, sign, pitch: PitchDims) -> str:
    if not (np.isfinite(x) and np.isfinite(y)):
        return "free_kick"
    if abs(x) < 1.5 and abs(y) < 1.5:
        return "kick_off"
    if abs(y) > pitch.half_width - 1.5:
        return "throw_in" if abs(x) < pitch.half_length - 2 else "corner"
    if abs(x) > pitch.half_length - 2.5 and abs(y) > pitch.half_width - 3:
        return "corner"
    own_goal_x = -sign * pitch.half_length
    if abs(x - own_goal_x) < 6.5 and abs(y) < 10:
        return "goal_kick"
    return "free_kick"


def to_spadl(events: pd.DataFrame) -> pd.DataFrame:
    """Minimal SPADL view (type_name, result_name, start/end coords) for socceraction tooling."""
    out = events.copy()
    out["type_name"] = out["type"].map(SPADL_MAP).fillna("non_action")
    out["result_name"] = out["outcome"].map({"success": "success", "fail": "fail"}).fillna("success")
    return out[["event_id", "period", "t", "team", "player_id", "type_name", "result_name",
                "start_x", "start_y", "end_x", "end_y"]]


def event_f1(pred: pd.DataFrame, truth: pd.DataFrame, types=("pass", "shot"), tol_s: float = 1.0) -> dict:
    """Event F1 at +-tol_s per type (build plan section 8). Greedy one-to-one matching in time."""
    res = {}
    for typ in types:
        p = pred[pred["type"] == typ].sort_values(["period", "t"])
        g = truth[truth["type"] == typ].sort_values(["period", "t"])
        tp = 0
        for period in set(p["period"]).union(set(g["period"])):
            pt = p.loc[p["period"] == period, "t"].to_numpy()
            gt = g.loc[g["period"] == period, "t"].to_numpy()
            used = np.zeros(len(gt), dtype=bool)
            for t in pt:
                if len(gt) == 0:
                    break
                d = np.abs(gt - t)
                d[used] = np.inf
                j = int(np.argmin(d))
                if d[j] <= tol_s:
                    used[j] = True
                    tp += 1
        prec = tp / len(p) if len(p) else float("nan")
        rec = tp / len(g) if len(g) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec > 0 else 0.0
        res[typ] = {"tp": tp, "n_pred": int(len(p)), "n_true": int(len(g)), "precision": prec, "recall": rec, "f1": f1}
    return res
