"""Stage 9: projection & smoothing -> the clean 10 Hz tracking table in pitch metres.

Foot points were projected during tracking; here tracks get their identity and team, periods are applied,
everything is resampled to 10 Hz and passed through the RTS smoother with speed/acceleration caps.
Interpolated points are flagged ``estimated``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fmi_analytics.kinematics import SmoothConfig, smooth_tracking
from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._util import load, save, video_meta
from fmi_schema.adapters.common import assign_frames, infer_home_attack_sign, resample_long
from fmi_schema.io import dump_json
from fmi_schema.models import TRACKING_COLUMNS, MatchMeta, PeriodInfo, TeamInfo
from fmi_schema.package import conform


def build_meta(ctx: StageContext, vm: dict, fps_out: float) -> MatchMeta:
    mi = ctx.match_input
    off = vm.get("source_offset_s", 0.0)
    periods_s = [(a - off, b - off) for a, b in mi.periods] if mi.periods else [(0.0, vm["duration"])]
    periods = [PeriodInfo(period=i + 1, start_frame=int(round(a * fps_out)), end_frame=int(round(b * fps_out)),
                          home_attack_sign=1) for i, (a, b) in enumerate(periods_s)]
    tier = mi.video_tier
    return MatchMeta(match_id=mi.match_id, source="pipeline", competition=mi.competition, date=mi.date,
                     home=TeamInfo(side="home", name=mi.home_name, jersey_color=mi.home_color),
                     away=TeamInfo(side="away", name=mi.away_name, jersey_color=mi.away_color),
                     pitch_length=mi.pitch_length, pitch_width=mi.pitch_width, fps=fps_out,
                     video_tier=tier if tier in ("broadcast", "panoramic", "follow_cam", "static") else "broadcast",
                     video_fps=vm["fps"], periods=periods,
                     provider_ids={"source_offset_s": off, "periods_video_s": periods_s})


class Project(Stage):
    name = "project"
    version = "1"
    deps = ("ingest", "link", "team", "identity", "ball")
    config_sections = ("output",)
    uses_source = True

    def run(self, ctx: StageContext, inputs) -> StageResult:
        fps_out = ctx.config.output.fps
        vm = video_meta(inputs["ingest"])
        fps = vm["fps"]
        meta = build_meta(ctx, vm, fps_out)
        tracks = load(inputs["link"], "tracks")
        ids = load(inputs["identity"], "identities")
        ball = load(inputs["ball"], "ball")
        roster = load(inputs["ingest"], "roster")

        t = tracks.merge(ids[["track_id", "team", "role", "player_id", "id_conf"]], on="track_id", how="left")
        t = t[t["team"].isin(["home", "away", "referee"])]
        t["id"] = np.where(t["player_id"].notna(), t["player_id"].astype(object), t["track_id"].astype(object))
        t = t.sort_values("score", ascending=False).drop_duplicates(["frame", "id"])
        long = pd.DataFrame({"vframe": t["frame"], "id": t["id"], "team": t["team"], "x": t["px"], "y": t["py"],
                             "confidence": t["score"] * np.where(t["player_id"].notna(), 1.0, 0.8)})
        b = ball[~ball["estimated"].astype(bool) & np.isfinite(ball["x"])]
        long = pd.concat([long, pd.DataFrame({"vframe": b["frame"], "id": "ball", "team": "ball", "x": b["x"],
                                              "y": b["y"], "confidence": b["confidence"]})], ignore_index=True)
        long = long[np.isfinite(long["x"])]
        # periods
        s = long["vframe"] / fps
        long["period"] = 0
        long["t"] = np.nan
        for p, (a, bnd) in zip(meta.periods, meta.provider_ids["periods_video_s"]):
            m = (s >= a) & (s <= bnd)
            long.loc[m, "period"] = p.period
            long.loc[m, "t"] = s[m] - a
        long = long[long["period"] > 0].drop(columns=["vframe"])
        long10 = resample_long(long, fps, fps_out, value_cols=("x", "y", "confidence"))
        long10["z"] = np.nan
        long10["estimated"] = False
        if ctx.match_input.home_attack_sign_p1 is not None:
            s1 = int(ctx.match_input.home_attack_sign_p1)
        else:
            s1 = infer_home_attack_sign(long10, 1)
        for p in meta.periods:
            p.home_attack_sign = s1 if p.period % 2 == 1 else -s1
        long10 = assign_frames(long10, meta)
        tracking, sqa = smooth_tracking(conform(long10, TRACKING_COLUMNS), SmoothConfig(fps=fps_out))
        tracking = conform(tracking, TRACKING_COLUMNS)
        dump_json(meta.model_dump(mode="json"), ctx.out_dir / "meta.json")
        qa = self.qa(rows=len(tracking), objects=int(tracking["id"].nunique()) if len(tracking) else 0,
                     home_attack_sign_p1=s1, **sqa)
        return StageResult({"tracking": save(ctx, tracking, "tracking"), "meta": "meta.json",
                            "roster": save(ctx, roster, "roster")}, qa)


def load_meta(out) -> MatchMeta:
    return MatchMeta.model_validate(json.loads(out.path("meta").read_text()))
