"""Stage 1: ingest. Probe the upload, make a constant-fps proxy (optionally a segment), find shot cuts,
parse the squad CSV."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

from fmi_pipeline import video as V
from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._util import save
from fmi_schema.io import dump_json

SQUAD_COLUMNS = ["team", "number", "name", "position", "player_id"]


def parse_squad(text: str, home_name: str = "Home", away_name: str = "Away") -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "team" not in df or "number" not in df:
        raise ValueError("squad CSV needs at least columns: team, number, name")
    df["team"] = df["team"].astype(str).str.strip().str.lower().map(
        lambda t: "home" if t in ("home", "h", home_name.lower()) else ("away" if t in ("away", "a", away_name.lower()) else t))
    bad = set(df["team"]) - {"home", "away"}
    if bad:
        raise ValueError(f"squad CSV: team must be home/away (or the team names); got {sorted(bad)}")
    df["number"] = pd.to_numeric(df["number"], errors="coerce").astype("Int64")
    if "name" not in df:
        df["name"] = df["team"] + " #" + df["number"].astype(str)
    if "position" not in df:
        df["position"] = None
    if "player_id" not in df or df["player_id"].isna().all():
        df["player_id"] = df["team"] + "_" + df["number"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    df["team_name"] = df["team"].map({"home": home_name, "away": away_name})
    df["starter"] = df.get("starter", pd.Series([True] * len(df))).fillna(True).astype(bool)
    return df[["player_id", "team", "team_name", "number", "name", "position", "starter"]]


class Ingest(Stage):
    name = "ingest"
    version = "1"
    config_sections = ("ingest",)
    uses_source = True

    def run(self, ctx: StageContext, inputs) -> StageResult:
        mi, cfg = ctx.match_input, ctx.config.ingest
        qa = self.qa()
        if mi.video.startswith("synthetic://"):
            src = V.open_video(mi.video)
            info, proxy_ref, cuts = src.info, mi.video, []
            squad_text = src.scene.squad_csv() if not mi.squad_csv else Path(mi.squad_csv).read_text()
        else:
            if not V.have_ffmpeg():
                raise RuntimeError("ffmpeg/ffprobe not found on PATH (needed for ingest)")
            V.make_proxy(mi.video, ctx.out_dir / "proxy.mp4", cfg.proxy_fps, cfg.proxy_max_width, mi.segment)
            info = V.probe(ctx.out_dir / "proxy.mp4")
            proxy_ref = "proxy.mp4"
            cuts = V.detect_scene_cuts(ctx.out_dir / "proxy.mp4", cfg.scene_cut_threshold)
            squad_text = Path(mi.squad_csv).read_text() if mi.squad_csv else ""
        # shot segments in proxy frames
        bounds = [0] + sorted({int(round(c * info.fps)) for c in cuts if 0 < c * info.fps < info.n_frames}) + [info.n_frames]
        segments = pd.DataFrame({"segment": np.arange(len(bounds) - 1), "start_frame": bounds[:-1],
                                 "end_frame": [b - 1 for b in bounds[1:]]})
        roster = parse_squad(squad_text, mi.home_name, mi.away_name) if squad_text.strip() else pd.DataFrame(
            columns=["player_id", "team", "team_name", "number", "name", "position", "starter"])
        if roster.empty:
            qa.warnings.append("no squad list: identities will stay as track ids until assigned in review")
        vm = {"proxy": proxy_ref, "fps": info.fps, "width": info.width, "height": info.height,
              "n_frames": info.n_frames, "duration": info.duration,
              "source_offset_s": mi.segment[0] if mi.segment else 0.0, "scene_cuts_s": cuts}
        dump_json(vm, ctx.out_dir / "video_meta.json")
        qa.metrics.update(fps=info.fps, width=info.width, height=info.height, n_frames=info.n_frames,
                          shot_segments=len(segments), roster_players=len(roster))
        return StageResult({"video_meta": "video_meta.json", "segments": save(ctx, segments, "segments"),
                            "roster": save(ctx, roster, "roster"),
                            **({"proxy": "proxy.mp4"} if proxy_ref == "proxy.mp4" else {})}, qa)
