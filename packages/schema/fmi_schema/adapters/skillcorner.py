"""SkillCorner open data (github.com/SkillCorner/opendata) -> Match Package.

Files per match (``data/opendata/skillcorner/matches/<id>/``):
    <id>_match.json                      metadata, roster, periods, pitch size
    <id>_tracking_extrapolated.jsonl(.gz) 10 fps player + ball positions in centred metres
    <id>_dynamic_events.csv(.gz)         player possessions, passing options, engagements, runs
    <id>_phases_of_play.csv(.gz)         phases of play

SkillCorner tracking already uses our coordinate convention (centred metres, y up) and 10 fps.
``home_team_side[i] == "left_to_right"`` means the home team attacks +x in period i+1.
Dynamic-event coordinates are normalised so the acting team attacks left-to-right; we flip them back to the
tracking frame using the ``attacking_side`` column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fmi_schema.io import iter_jsonl, load_json, resolve
from fmi_schema.models import EVENT_COLUMNS, TRACKING_COLUMNS, MatchMeta, PeriodInfo, TeamInfo
from fmi_schema.package import MatchPackage, conform


def match_dir(root: str | Path, match_id: str | int) -> Path:
    root = Path(root)
    d = root / "matches" / str(match_id)
    return d if d.exists() else root / str(match_id)


def list_matches(root: str | Path) -> list[dict]:
    root = Path(root)
    path = root / "matches.json"
    if not path.exists():
        return [{"id": int(p.name)} for p in sorted((root / "matches").iterdir()) if p.is_dir()]
    return load_json(path)


def load_meta(match: dict) -> tuple[MatchMeta, pd.DataFrame, dict[int, str]]:
    home_id = match["home_team"]["id"]

    def team(side: str) -> TeamInfo:
        t = match[f"{side}_team"]
        kit = match.get(f"{side}_team_kit") or {}
        return TeamInfo(side=side, name=t["name"], short_name=t.get("short_name"), provider_id=str(t["id"]),
                        jersey_color=kit.get("jersey_color"), number_color=kit.get("number_color"))

    sides = match.get("home_team_side") or ["left_to_right", "right_to_left"]
    periods = [
        PeriodInfo(period=p["period"], start_frame=p["start_frame"], end_frame=p["end_frame"],
                   home_attack_sign=1 if sides[min(i, len(sides) - 1)] == "left_to_right" else -1)
        for i, p in enumerate(match["match_periods"])
    ]
    comp = match.get("competition_edition") or {}
    meta = MatchMeta(
        match_id=f"sc-{match['id']}",
        source="skillcorner",
        competition=(comp.get("competition") or {}).get("name"),
        season=(comp.get("season") or {}).get("name"),
        date=match.get("date_time"),
        venue=(match.get("stadium") or {}).get("name"),
        home=team("home"),
        away=team("away"),
        score={"home": match.get("home_team_score"), "away": match.get("away_team_score")},
        pitch_length=float(match.get("pitch_length") or 105),
        pitch_width=float(match.get("pitch_width") or 68),
        fps=10.0,
        video_tier="broadcast",
        periods=periods,
        provider_ids={"skillcorner_match_id": match["id"]},
    )
    rows, side_of = [], {}
    for p in match["players"]:
        side = "home" if p["team_id"] == home_id else "away"
        side_of[p["id"]] = side
        if p.get("trackable_object") is not None:
            side_of.setdefault(p["trackable_object"], side)
        rows.append({
            "player_id": str(p["id"]),
            "team": side,
            "team_name": meta.team_info(side).name,
            "number": p.get("number"),
            "name": p.get("short_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "position": ((p.get("player_role") or {}).get("acronym")),
            "starter": p.get("start_time") == "00:00:00",
        })
    roster = pd.DataFrame(rows)
    return meta, roster, side_of


def load_tracking(path: str | Path, meta: MatchMeta, side_of: dict[int, str]) -> pd.DataFrame:
    starts = {p.period: p.start_frame for p in meta.periods}
    ends = {p.period: p.end_frame for p in meta.periods}
    cols: dict[str, list] = {k: [] for k in ("frame", "period", "t", "id", "team", "x", "y", "z", "confidence", "estimated")}

    def add(frame, period, oid, team, x, y, z, detected):
        cols["frame"].append(frame)
        cols["period"].append(period)
        cols["t"].append((frame - starts[period]) / meta.fps)
        cols["id"].append(oid)
        cols["team"].append(team)
        cols["x"].append(x)
        cols["y"].append(y)
        cols["z"].append(z)
        cols["confidence"].append(1.0 if detected else 0.6)
        cols["estimated"].append(not detected)

    for rec in iter_jsonl(path):
        period = rec.get("period")
        if period is None or period not in starts:
            continue
        frame = rec["frame"]
        if frame < starts[period] or frame > ends[period]:
            continue
        for p in rec.get("player_data") or []:
            if p.get("x") is None:
                continue
            add(frame, period, str(p["player_id"]), side_of.get(p["player_id"], "unknown"),
                p["x"], p["y"], np.nan, bool(p.get("is_detected")))
        b = rec.get("ball_data") or {}
        if b.get("x") is not None:
            add(frame, period, "ball", "ball", b["x"], b["y"], b.get("z") if b.get("z") is not None else np.nan,
                bool(b.get("is_detected")))
    df = pd.DataFrame(cols)
    return conform(df, TRACKING_COLUMNS)


_SET_PIECE = {
    "throw_in": "throw_in", "corner": "corner", "free_kick": "free_kick", "goal_kick": "goal_kick",
}


def load_events(path: str | Path, meta: MatchMeta, side_by_team_id: dict[str, str]) -> pd.DataFrame:
    """Map SkillCorner ``player_possession`` rows to MVP events (source = provider)."""
    df = pd.read_csv(path, low_memory=False)
    pp = df[df["event_type"] == "player_possession"].copy()
    starts = {p.period: p.start_frame for p in meta.periods}
    out = []

    def t_of(frame, period):
        return (frame - starts.get(period, 0)) / meta.fps

    flip_cols = [c for c in pp.columns if c.endswith(("x_start", "y_start", "x_end", "y_end", "_x_reception",
                                                       "_y_reception", "_x_pass", "_y_pass"))]
    sign = np.where(pp["attacking_side"].astype(str) == "right_to_left", -1.0, 1.0)
    for c in flip_cols:
        pp[c] = pd.to_numeric(pp[c], errors="coerce") * sign
    for r in pp.itertuples(index=False):
        period = int(r.period)
        team = side_by_team_id.get(str(r.team_id), "unknown")
        pid = str(int(r.player_id)) if pd.notna(r.player_id) else None
        base = {"period": period, "team": team, "player_id": pid, "source": "provider", "confidence": 1.0}
        # start-of-possession events
        st = str(getattr(r, "start_type", "") or "")
        if st in ("recovery",):
            out.append({**base, "type": "recovery", "frame": r.frame_start, "t": t_of(r.frame_start, period),
                        "start_x": r.x_start, "start_y": r.y_start, "outcome": "success"})
        elif st.endswith("interception"):
            out.append({**base, "type": "interception", "frame": r.frame_start, "t": t_of(r.frame_start, period),
                        "start_x": r.x_start, "start_y": r.y_start, "outcome": "success"})
        gib = str(getattr(r, "game_interruption_before", "") or "")
        for key, etype in _SET_PIECE.items():
            if gib.startswith(key) and gib.endswith("_for"):
                out.append({**base, "type": etype, "frame": r.frame_start, "t": t_of(r.frame_start, period),
                            "start_x": r.x_start, "start_y": r.y_start, "outcome": "unknown"})
        # end-of-possession events
        et = str(getattr(r, "end_type", "") or "")
        ev = {**base, "frame": r.frame_end, "t": t_of(r.frame_end, period), "start_x": r.x_end, "start_y": r.y_end}
        if et == "pass":
            outcome = {"successful": "success", "unsuccessful": "fail", "offside": "fail"}.get(
                str(getattr(r, "pass_outcome", "")), "unknown")
            recv = getattr(r, "player_targeted_id", np.nan)
            out.append({**ev, "type": "pass", "outcome": outcome,
                        "receiver_id": str(int(recv)) if pd.notna(recv) else None,
                        "end_x": getattr(r, "player_targeted_x_reception", np.nan),
                        "end_y": getattr(r, "player_targeted_y_reception", np.nan)})
        elif et == "shot":
            goal = bool(getattr(r, "lead_to_goal", False)) and str(getattr(r, "game_interruption_after", "")) == "goal_for"
            out.append({**ev, "type": "shot", "outcome": "success" if goal else "fail"})
        elif et == "clearance":
            out.append({**ev, "type": "clearance", "outcome": "unknown"})
        elif et == "possession_loss":
            out.append({**ev, "type": "possession_loss", "outcome": "fail"})
    events = pd.DataFrame(out)
    if events.empty:
        return conform(events, EVENT_COLUMNS)
    events = events.sort_values(["period", "t"]).reset_index(drop=True)
    events["event_id"] = [f"sc{i}" for i in range(len(events))]
    events["t_end"] = events["t"]
    return conform(events, EVENT_COLUMNS)


def load_match(root: str | Path, match_id: str | int, with_tracking: bool = True) -> MatchPackage:
    d = match_dir(root, match_id)
    match = load_json(resolve(d / f"{match_id}_match.json"))
    meta, roster, side_of = load_meta(match)
    pkg = MatchPackage(meta=meta, roster=roster)
    if with_tracking:
        pkg.tracking = load_tracking(resolve(d / f"{match_id}_tracking_extrapolated.jsonl"), meta, side_of)
    side_by_team_id = {meta.home.provider_id: "home", meta.away.provider_id: "away"}
    try:
        pkg.events = load_events(resolve(d / f"{match_id}_dynamic_events.csv"), meta, side_by_team_id)
    except FileNotFoundError:
        pass
    try:
        phases = pd.read_csv(resolve(d / f"{match_id}_phases_of_play.csv"))
        pkg.derived["provider_phases"] = phases
    except FileNotFoundError:
        pass
    return pkg
