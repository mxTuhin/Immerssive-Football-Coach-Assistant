"""Deterministic tools over a Match Package. The AI analyst may only state numbers these tools return.

Every tool returns JSON-serialisable dicts with a ``notes`` field describing data quality where relevant
(e.g. events detected automatically vs provided, estimated positions), so the model can hedge correctly.
"""

from __future__ import annotations

import difflib
from typing import Any, Callable

import numpy as np
import pandas as pd

from fmi_analytics.common import attack_sign
from fmi_schema.io import to_jsonable
from fmi_schema.package import MatchPackage

ToolFn = Callable[..., dict]


def _round(df: pd.DataFrame, nd: int = 2) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    return to_jsonable(df.round(nd).replace({np.nan: None}).to_dict(orient="records"))


class MatchTools:
    def __init__(self, pkg: MatchPackage):
        self.pkg = pkg
        self.ev = pkg.events
        self.d = pkg.derived

    # ------------------------------------------------------------------ helpers
    def _team(self, team: str | None) -> str | None:
        if team is None:
            return None
        t = team.strip().lower()
        if t in ("home", "away"):
            return t
        for side in ("home", "away"):
            info = self.pkg.meta.team_info(side)
            names = [info.name.lower(), (info.short_name or "").lower()]
            if any(t and (t in n or n in t) for n in names if n):
                return side
        return None

    def _player(self, player: str | None) -> str | None:
        if not player:
            return None
        r = self.pkg.roster
        if str(player) in set(r["player_id"].astype(str)):
            return str(player)
        names = r["name"].astype(str).tolist()
        hit = difflib.get_close_matches(player, names, n=1, cutoff=0.5)
        if not hit:
            low = player.lower()
            hit = [n for n in names if low in n.lower()][:1]
        return str(r.loc[r["name"] == hit[0], "player_id"].iloc[0]) if hit else None

    def _notes(self) -> list[str]:
        notes = []
        src = set(self.ev["source"].dropna().unique()) if len(self.ev) else set()
        if "auto" in src:
            notes.append("events detected automatically from tracking (pass F1 ~0.77, shot F1 ~0.35 on open-data "
                         "validation); treat shot counts as approximate")
        if "provider" in src:
            notes.append("events supplied by the data provider")
        est = float(self.pkg.tracking["estimated"].mean()) if len(self.pkg.tracking) else 0.0
        if est > 0.1:
            notes.append(f"{est:.0%} of positions are interpolated/estimated")
        for name, q in self.pkg.qa.items():
            notes += [f"{name}: {w}" for w in q.warnings[:3]]
        return notes

    # ------------------------------------------------------------------ tools
    def get_match_overview(self) -> dict:
        s = self.d.get("summary", {})
        meta = self.pkg.meta
        return {"teams": {k: meta.team_info(k).name for k in ("home", "away")}, "score": meta.score,
                "competition": meta.competition, "date": meta.date, "pitch_m": [meta.pitch_length, meta.pitch_width],
                "periods": [p.model_dump() for p in meta.periods],
                "possession_share": s.get("possession_share"), "pass_completion": s.get("pass_completion"),
                "event_counts": s.get("event_counts"), "ppda": s.get("ppda"),
                "team_distance_km": s.get("team_distance_km"), "xt_by_team": s.get("xt_by_team"),
                "notes": self._notes()}

    def list_key_moments(self) -> dict:
        return {"moments": to_jsonable(self.d.get("key_moments", [])),
                "how_selected": "deterministic scoring of goals, shots, high turnovers, set-piece chances, "
                                "high-xT passes and long sprints; >= 45 s apart"}

    def get_events(self, type: str | None = None, team: str | None = None, player: str | None = None,
                   period: int | None = None, t_from: float | None = None, t_to: float | None = None,
                   limit: int = 50) -> dict:
        ev = self.ev
        if type:
            ev = ev[ev["type"].isin([x.strip() for x in type.split(",")])]
        side = self._team(team)
        if side:
            ev = ev[ev["team"] == side]
        pid = self._player(player)
        if player and pid:
            ev = ev[(ev["player_id"] == pid) | (ev["receiver_id"] == pid)]
        if period:
            ev = ev[ev["period"] == period]
        if t_from is not None:
            ev = ev[ev["t"] >= t_from]
        if t_to is not None:
            ev = ev[ev["t"] <= t_to]
        return self._format_events(ev, limit)

    def _format_events(self, ev: pd.DataFrame, limit: int) -> dict:
        cols = ["event_id", "period", "t", "type", "team", "player_id", "receiver_id", "start_x", "start_y",
                "end_x", "end_y", "outcome", "source", "confidence"]
        out = ev[cols].head(min(limit, 200)).copy()
        out["player"] = out["player_id"].map(lambda p: self.pkg.player_name(p) if pd.notna(p) else None)
        out["clock"] = [f"P{int(p)} {int(t // 60):02d}:{int(t % 60):02d}" for p, t in zip(out["period"], out["t"])]
        return {"count": int(len(ev)), "events": _round(out), "notes": self._notes()}

    def get_player_stats(self, player: str | None = None, team: str | None = None, sort_by: str = "distance_m",
                         top: int = 10) -> dict:
        phys = self.d.get("physical")
        if not isinstance(phys, pd.DataFrame) or phys.empty:
            return {"players": [], "notes": ["no physical data"]}
        df = phys.copy()
        if len(self.ev):
            counts = self.ev.groupby(["player_id", "type"]).size().unstack(fill_value=0)
            df = df.merge(counts, left_on="player_id", right_index=True, how="left")
        pid = self._player(player)
        if player:
            if not pid:
                return {"players": [], "notes": [f"no player matching {player!r}"]}
            df = df[df["player_id"] == pid]
        side = self._team(team)
        if side:
            df = df[df["team"] == side]
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
        df["name"] = df["player_id"].map(self.pkg.player_name)
        return {"players": _round(df.head(top)), "units": {"distance": "m", "top_speed": "km/h"},
                "notes": ["top speed = 99.5th percentile of smoothed speed"] + self._notes()}

    def get_team_shape(self, team: str, possession: str = "all") -> dict:
        side = self._team(team)
        shape = (self.d.get("summary", {}).get("shape") or {}).get(side or "", {})
        key = {"all": "all", "in": "in_possession", "out": "out_possession"}.get(possession, "all")
        return {"team": side, "possession": possession, "shape": shape.get(key),
                "definitions": {"width": "m between widest outfield players", "length": "m deepest to highest",
                                "hull_area": "m^2 convex hull of outfield players",
                                "line_height": "m from own goal line to the deepest outfield line",
                                "centroid_x": "m, + is toward the opponent goal"}}

    def get_phases(self, team: str) -> dict:
        side = self._team(team)
        return {"team": side, "share_of_time": (self.d.get("summary", {}).get("phases") or {}).get(side or ""),
                "notes": ["shares are fractions of frames with a known team in possession"]}

    def get_pressing(self, team: str) -> dict:
        side = self._team(team)
        s = self.d.get("summary", {})
        return {"team": side, "pressing": (s.get("pressing") or {}).get(side or ""), "ppda": (s.get("ppda") or {}).get(side or ""),
                "definitions": {"pressed_share": "share of opponent ball-carrier frames under pressure",
                                "ppda": "opponent passes in their own 60% per defensive action there (lower = more pressing)"}}

    def get_possession_timeline(self, bin_minutes: float = 5.0) -> dict:
        poss = self.d.get("possession")
        if not isinstance(poss, pd.DataFrame) or poss.empty:
            return {"bins": []}
        p = poss.dropna(subset=["team_in_possession"]).copy()
        p["bin"] = (p["t"] // (bin_minutes * 60)).astype(int)
        g = p.groupby(["period", "bin"])["team_in_possession"].value_counts(normalize=True).unstack(fill_value=0)
        rows = [{"period": int(k[0]), "from_min": float(k[1] * bin_minutes), "to_min": float((k[1] + 1) * bin_minutes),
                 "home": round(float(v.get("home", 0)), 3), "away": round(float(v.get("away", 0)), 3)} for k, v in g.iterrows()]
        return {"bins": rows}

    def get_moment_context(self, period: int, t: float, window_s: float = 10.0) -> dict:
        ev = self.get_events(period=period, t_from=t - window_s, t_to=t + window_s, limit=30)
        tr = self.pkg.tracking
        try:
            frame = self.pkg.frame_at(period, t)
        except KeyError:
            return {"error": f"no period {period}"}
        snap = tr[tr["frame"] == frame]
        ball = snap[snap["id"] == "ball"]
        near = []
        if not ball.empty:
            bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
            pl = snap[snap["team"].isin(["home", "away"])].copy()
            pl["dist_to_ball"] = np.hypot(pl["x"] - bx, pl["y"] - by)
            pl = pl.nsmallest(6, "dist_to_ball")
            near = [{"player": self.pkg.player_name(r.id), "team": r.team, "dist_to_ball_m": round(r.dist_to_ball, 1),
                     "speed_ms": round(r.speed, 1) if pd.notna(r.speed) else None,
                     "facing_deg": round(r.body_yaw, 0) if pd.notna(r.body_yaw) else None} for r in pl.itertuples()]
        return {"period": period, "t": t, "events": ev["events"], "players_near_ball": near,
                "scene": {"period": period, "t_start": max(0.0, t - window_s), "t_end": t + window_s},
                "notes": ev["notes"]}

    def compare_periods(self) -> dict:
        ev = self.ev
        out = {}
        for p in sorted(ev["period"].dropna().unique()) if len(ev) else []:
            e = ev[ev["period"] == p]
            out[f"period_{int(p)}"] = {t: e[e["team"] == t]["type"].value_counts().to_dict() for t in ("home", "away")}
        poss = self.d.get("possession")
        if isinstance(poss, pd.DataFrame) and len(poss):
            for p, g in poss.groupby("period"):
                vc = g["team_in_possession"].value_counts(normalize=True)
                out.setdefault(f"period_{int(p)}", {})["possession_share"] = {k: round(float(v), 3) for k, v in vc.items()}
        return {"periods": out, "notes": self._notes()}

    def search_moments(self, event_type: str | None = None, zone: str | None = None, team: str | None = None,
                       player: str | None = None, outcome: str | None = None, limit: int = 20) -> dict:
        """Structured search. zone: final_third | middle_third | defensive_third | box | wide."""
        ev = self.ev
        if event_type:
            ev = ev[ev["type"].isin([x.strip() for x in event_type.split(",")])]
        side = self._team(team)
        if side:
            ev = ev[ev["team"] == side]
        pid = self._player(player)
        if player and pid:
            ev = ev[ev["player_id"] == pid]
        if outcome:
            ev = ev[ev["outcome"] == outcome]
        if zone and len(ev):
            L, W = self.pkg.meta.pitch_length, self.pkg.meta.pitch_width
            sign = np.array([attack_sign(self.pkg.meta, int(p), t) for p, t in zip(ev["period"], ev["team"])])
            x, y = ev["start_x"].to_numpy(float) * sign, ev["start_y"].to_numpy(float) * sign
            m = {"final_third": x > L / 6, "middle_third": np.abs(x) <= L / 6, "defensive_third": x < -L / 6,
                 "box": (x > L / 2 - 16.5) & (np.abs(y) < 20.16), "wide": np.abs(y) > W / 2 - 15}.get(zone)
            if m is not None:
                ev = ev[m]
        return self._format_events(ev, limit)

    # ------------------------------------------------------------------ registry for the LLM
    def specs(self) -> list[dict]:
        s = lambda **p: {"type": "object", "properties": p, "additionalProperties": False}  # noqa: E731
        string, integer, number = {"type": "string"}, {"type": "integer"}, {"type": "number"}
        team_d = {"type": "string", "description": "home, away, or a team name"}
        return [
            {"name": "get_match_overview", "description": "Teams, score, possession, pass completion, event counts, "
             "PPDA, distance, xT by team, and data-quality notes. Call first.", "input_schema": s()},
            {"name": "list_key_moments", "description": "Deterministic key-moment candidates with time windows "
             "(for clips and 3D scenes) and the reason each was selected.", "input_schema": s()},
            {"name": "get_events", "description": "Filter events. type: comma list of pass,cross,shot,goal,recovery,"
             "interception,tackle,clearance,possession_loss,throw_in,corner,free_kick,goal_kick,kick_off. "
             "t_from/t_to are seconds of the period clock.",
             "input_schema": s(type=string, team=team_d, player=string, period=integer, t_from=number, t_to=number,
                               limit=integer)},
            {"name": "get_player_stats", "description": "Per-player physical metrics (distance, speed zones, top "
             "speed, sprints, accelerations) and event counts. Filter by player name/id or team; sort_by any column.",
             "input_schema": s(player=string, team=team_d, sort_by=string, top=integer)},
            {"name": "get_team_shape", "description": "Average width, length, compactness (hull area), defensive "
             "line height and centroid. possession: all | in | out.",
             "input_schema": {**s(team=team_d, possession={"type": "string", "enum": ["all", "in", "out"]}),
                              "required": ["team"]}},
            {"name": "get_phases", "description": "Share of time in build-up/progression/final third and high/mid/"
             "low block and transitions.", "input_schema": {**s(team=team_d), "required": ["team"]}},
            {"name": "get_pressing", "description": "Pressing intensity on the opponent ball carrier and PPDA.",
             "input_schema": {**s(team=team_d), "required": ["team"]}},
            {"name": "get_possession_timeline", "description": "Possession share per time bin.",
             "input_schema": s(bin_minutes=number)},
            {"name": "get_moment_context", "description": "Events around a time plus the players nearest the ball "
             "(distance, speed, facing direction) and the 3D scene window.",
             "input_schema": {**s(period=integer, t=number, window_s=number), "required": ["period", "t"]}},
            {"name": "compare_periods", "description": "Event counts and possession per period for both teams.",
             "input_schema": s()},
            {"name": "search_moments", "description": "Structured search for situations, e.g. shots in the box by a "
             "player, recoveries in the final third. zone: final_third|middle_third|defensive_third|box|wide.",
             "input_schema": s(event_type=string, zone=string, team=team_d, player=string, outcome=string,
                               limit=integer)},
        ]

    def call(self, name: str, args: dict[str, Any]) -> dict:
        allowed = {sp["name"] for sp in self.specs()}
        if name not in allowed:
            raise ValueError(f"unknown tool {name!r}")
        return getattr(self, name)(**(args or {}))
