"""Run the full tracking-level chain on a Match Package.

    smooth (stage 9) -> facing (stage 10) -> possession -> events (stage 11) -> analytics (stage 12) -> moments

Used directly for open data (SkillCorner/Metrica) and by the pipeline's ``project``/``facing``/``events``/
``analytics`` stages for our own video.
"""

from __future__ import annotations

import logging

import pandas as pd

from fmi_analytics import events as ev_mod
from fmi_analytics.common import goalkeepers
from fmi_analytics.facing import compute_facing
from fmi_analytics.kinematics import SmoothConfig, smooth_tracking
from fmi_analytics.moments import candidate_moments, select_moments
from fmi_analytics.phases import phase_segments, phase_share, phases_timeseries
from fmi_analytics.physical import player_physical, sprint_list
from fmi_analytics.possession import PossessionConfig, compute_possession, possession_share
from fmi_analytics.pressing import ppda, press_summary, pressure_timeseries
from fmi_analytics.qa import plausibility
from fmi_analytics.shape import shape_summary, team_shape_timeseries
from fmi_analytics.xt import xt_added
from fmi_schema.package import MatchPackage

log = logging.getLogger(__name__)


def smooth_and_face(pkg: MatchPackage, smooth: bool = True) -> dict:
    qa = {}
    if smooth:
        pkg.tracking, qa = smooth_tracking(pkg.tracking, SmoothConfig(fps=pkg.meta.fps))
    gks = set(goalkeepers(pkg).values())
    pkg.tracking = compute_facing(pkg.tracking, pkg.meta, gks)
    return qa


def run_events(pkg: MatchPackage, possession: pd.DataFrame) -> pd.DataFrame:
    return ev_mod.detect_events(pkg.tracking, possession, pkg.meta, ev_mod.EventConfig(fps=pkg.meta.fps))


def run_analytics(pkg: MatchPackage, events: pd.DataFrame, possession: pd.DataFrame) -> dict:
    meta = pkg.meta
    gks = set(goalkeepers(pkg).values())
    phys = player_physical(pkg.tracking, meta.fps)
    if not phys.empty:
        phys["name"] = phys["player_id"].map(pkg.player_name)
    sprints = sprint_list(pkg.tracking, meta.fps)
    shape_ts = team_shape_timeseries(pkg.tracking, meta, gks, possession)
    ph = phases_timeseries(possession, shape_ts, meta, meta.fps)
    pressure = pressure_timeseries(pkg.tracking, possession)
    xt_moves = xt_added(events, meta) if not events.empty else pd.DataFrame()
    cands = candidate_moments(pkg, events, xt_moves, sprints)
    moments = select_moments(cands)

    counts = events.groupby(["team", "type"]).size().unstack(fill_value=0) if not events.empty else pd.DataFrame()
    summary = {
        "match_id": pkg.match_id,
        "teams": {s: pkg.meta.team_info(s).name for s in ("home", "away")},
        "score": meta.score,
        "possession_share": possession_share(possession),
        "event_counts": {t: counts.loc[t].to_dict() for t in counts.index} if not counts.empty else {},
        "pass_completion": _pass_completion(events),
        "shape": shape_summary(shape_ts),
        "phases": phase_share(ph),
        "pressing": press_summary(pressure),
        "ppda": ppda(events, meta),
        "team_distance_km": (phys.groupby("team")["distance_m"].sum() / 1000).round(2).to_dict() if not phys.empty else {},
        "xt_by_team": xt_moves.groupby("team")["xt_added"].sum().round(3).to_dict() if not xt_moves.empty else {},
        "events_source": sorted(events["source"].dropna().unique().tolist()) if not events.empty else [],
    }
    return {
        "summary": summary,
        "physical": phys,
        "sprints": sprints,
        "shape_timeseries": shape_ts,
        "phases": phase_segments(ph, meta.fps),
        "pressure": pressure,
        "xt_moves": xt_moves,
        "possession": possession,
        "moment_candidates": cands,
        "key_moments": moments,
    }


def _pass_completion(events: pd.DataFrame) -> dict:
    p = events[events["type"].isin(["pass", "cross"])] if not events.empty else events
    if p.empty:
        return {}
    return (p.assign(ok=p["outcome"] == "success").groupby("team")["ok"].mean().round(3).to_dict())


def run_all(pkg: MatchPackage, smooth: bool = True, events_source: str = "best") -> MatchPackage:
    """Run everything and store results in ``pkg.derived`` / ``pkg.qa``.

    events_source: ``auto`` (ours from tracking), ``provider`` (bundled with open data), ``best`` (provider when
    present, else auto). Auto events are always stored in ``derived['auto_events']`` for evaluation.
    """
    sq = smooth_and_face(pkg, smooth=smooth)
    possession = compute_possession(pkg.tracking, PossessionConfig(fps=pkg.meta.fps))
    auto_events = run_events(pkg, possession)
    provider = pkg.events[pkg.events["source"].isin(["provider", "human"])] if not pkg.events.empty else pkg.events
    if events_source == "auto" or (events_source == "best" and provider.empty):
        events = auto_events
    else:
        events = provider
    pkg.events = events
    pkg.derived["auto_events"] = auto_events
    if not provider.empty:
        pkg.derived["auto_vs_provider_f1"] = ev_mod.event_f1(auto_events, provider, types=("pass", "shot"))
    pkg.derived.update(run_analytics(pkg, events, possession))
    qa = plausibility(pkg.tracking, pkg.meta, stage="analytics")
    qa.metrics.update({k: v for k, v in sq.items()})
    pkg.qa["analytics"] = qa
    return pkg
