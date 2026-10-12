"""Label-free QA and plausibility gates (build plan sections 6 and 8).

Checks: speed > 36 km/h, acceleration > 8 m/s^2, > 11 players per team on the pitch, players off the pitch
for > 2 s, mid-pitch ID births/deaths, impossible jumps between frames. Produces a ``StageQA`` with
review items, so wrong data is flagged instead of shown as fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmi_analytics.common import MAX_PLAYER_ACCEL, MAX_PLAYER_SPEED
from fmi_schema.models import ReviewItem, StageQA
from fmi_schema.pitch import PitchDims


def plausibility(tracking: pd.DataFrame, meta, stage: str = "qa", version: str = "1",
                 fps: float = 10.0, max_review: int = 200) -> StageQA:
    pitch = PitchDims(meta.pitch_length, meta.pitch_width)
    players = tracking[tracking["team"].isin(["home", "away"])].sort_values(["id", "frame"])
    qa = StageQA(stage=stage, version=version)
    review: list[ReviewItem] = []
    if players.empty:
        qa.ok = False
        qa.warnings.append("no player tracking")
        return qa

    sp = players["speed"].to_numpy(dtype=float)
    qa.metrics["speed_over_cap_samples"] = int(np.nansum(sp > MAX_PLAYER_SPEED + 1e-6))
    same = (players["id"].to_numpy()[1:] == players["id"].to_numpy()[:-1]) & (np.diff(players["frame"].to_numpy()) == 1)
    dv = np.hypot(np.diff(players["vx"].to_numpy(float)), np.diff(players["vy"].to_numpy(float))) * fps
    qa.metrics["accel_over_cap_samples"] = int(np.nansum((dv > MAX_PLAYER_ACCEL) & same))
    jump = np.hypot(np.diff(players["x"].to_numpy(float)), np.diff(players["y"].to_numpy(float)))
    jumps = same & (jump > MAX_PLAYER_SPEED / fps + 1.0)
    qa.metrics["impossible_jumps"] = int(jumps.sum())
    for i in np.nonzero(jumps)[0][:50]:
        r = players.iloc[i + 1]
        review.append(ReviewItem(item_id=f"jump-{r['id']}-{int(r['frame'])}", stage=stage, kind="plausibility",
                                 priority=0.2, frame=int(r["frame"]), period=int(r["period"]), t=float(r["t"]),
                                 track_ids=[str(r["id"])], message=f"jump of {jump[i]:.1f} m in one frame"))

    # players per team on the pitch
    on = players[pitch.contains(players["x"].to_numpy(float), players["y"].to_numpy(float), margin=1.0)]
    counts = on.groupby(["frame", "team"]).size()
    over = counts[counts > 11]
    qa.metrics["frames_over_11_per_team"] = int(over.index.get_level_values(0).nunique())
    for (frame, team), n in over.head(30).items():
        review.append(ReviewItem(item_id=f"count-{team}-{frame}", stage=stage, kind="identity", priority=0.3,
                                 frame=int(frame), message=f"{n} {team} players on the pitch"))

    # off the pitch for > 2 s
    off = ~pitch.contains(players["x"].to_numpy(float), players["y"].to_numpy(float), margin=3.0)
    players = players.assign(_off=off)
    long_off = 0
    for pid, g in players.groupby("id"):
        o = g["_off"].to_numpy()
        if not o.any():
            continue
        d = np.diff(np.concatenate([[0], o.astype(int), [0]]))
        runs = list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))
        long_off += sum(1 for a, b in runs if (b - a) / fps > 2.0)
    qa.metrics["off_pitch_over_2s"] = long_off

    # births / deaths away from the touchlines and outside period starts/ends
    period_bounds = {p.period: (p.start_frame, p.end_frame) for p in meta.periods}
    mid_births = 0
    for pid, g in players.groupby("id"):
        first, last = g.iloc[0], g.iloc[-1]
        lo, hi = period_bounds.get(int(first["period"]), (first["frame"], last["frame"]))
        for row, edge in ((first, lo), (last, hi)):
            if abs(int(row["frame"]) - edge) <= fps * 2:
                continue
            inside = abs(row["x"]) < pitch.half_length - 5 and abs(row["y"]) < pitch.half_width - 5
            if inside:
                mid_births += 1
                if len(review) < max_review:
                    review.append(ReviewItem(item_id=f"birth-{pid}-{int(row['frame'])}", stage=stage, kind="merge",
                                             priority=0.4, frame=int(row["frame"]), period=int(row["period"]),
                                             t=float(row["t"]), track_ids=[str(pid)],
                                             message="track starts or ends in mid-pitch (possible ID break)"))
    qa.metrics["mid_pitch_births_deaths"] = mid_births
    qa.metrics["players_tracked"] = int(players["id"].nunique())
    qa.metrics["estimated_fraction"] = float(players["estimated"].mean())

    qa.ok = qa.metrics["frames_over_11_per_team"] == 0 and qa.metrics["impossible_jumps"] < 50
    if qa.metrics["frames_over_11_per_team"]:
        qa.warnings.append("more than 11 players of one team on the pitch in some frames")
    qa.review = sorted(review, key=lambda r: r.priority)[:max_review]
    return qa
