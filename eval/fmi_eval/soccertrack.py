"""SoccerTrack v2 label loader (V2: the labelled half-match).

SoccerTrack v2 (huggingface.co/datasets/atomscott/soccertrack-v2, gated) ships game-state labels: pitch
positions in metres, track ids, jersey numbers, roles, team, and ball-action events.

**Verify after download**: the exact file layout of the release you download decides which reader applies.
Two readers are provided; ``load_labels`` auto-detects:

1. SoccerNet-GSR style JSON (``Labels-GameState.json``): ``images`` + ``annotations`` with ``track_id``,
   ``attributes.{role, team, jersey}`` and ``bbox_pitch.{x_bottom_middle, y_bottom_middle}``.
2. A long CSV with a column mapping (defaults below; override with ``columns=``).

Output: long DataFrame ``frame, id, x, y, team, role, jersey`` in centred metres, resampled to 10 Hz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fmi_schema.io import open_text

DEFAULT_COLUMNS = {"frame": "frame", "id": "id", "x": "x", "y": "y", "team": "team", "role": "role",
                   "jersey": "jersey"}


def _from_gsr_json(path: Path) -> tuple[pd.DataFrame, float]:
    with open_text(path) as fh:
        d = json.load(fh)
    img_frame = {}
    for i, im in enumerate(sorted(d["images"], key=lambda im: str(im.get("file_name", im["image_id"])))):
        img_frame[im["image_id"]] = int(im.get("frame", i))
    rows = []
    for a in d["annotations"]:
        bp = a.get("bbox_pitch") or {}
        if bp.get("x_bottom_middle") is None:
            continue
        att = a.get("attributes") or {}
        role = att.get("role") or ("ball" if a.get("category_id") == 4 else None)
        rows.append({"frame": img_frame.get(a["image_id"]), "id": "ball" if role == "ball" else str(a.get("track_id")),
                     "x": float(bp["x_bottom_middle"]), "y": float(bp["y_bottom_middle"]),
                     "team": att.get("team"), "role": role, "jersey": att.get("jersey")})
    fps = float((d.get("info") or {}).get("frame_rate", 25))
    return pd.DataFrame(rows), fps


def load_labels(path: str | Path, fps_in: float | None = None, fps_out: float = 10.0,
                columns: dict[str, str] | None = None, flip_y: bool = False,
                origin: str = "centre", pitch: tuple[float, float] = (105.0, 68.0)) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".json", ".gz") and "json" in path.name:
        df, fps_guess = _from_gsr_json(path)
    else:
        cols = {**DEFAULT_COLUMNS, **(columns or {})}
        raw = pd.read_csv(path)
        df = pd.DataFrame({k: raw[v] if v in raw else None for k, v in cols.items()})
        fps_guess = 25.0
    fps_in = fps_in or fps_guess
    if origin == "corner":  # labels with (0, 0) at a corner flag
        df["x"] = df["x"] - pitch[0] / 2
        df["y"] = df["y"] - pitch[1] / 2
    if flip_y:
        df["y"] = -df["y"]
    df = df.dropna(subset=["frame", "x", "y"])
    df["frame"] = df["frame"].astype(int)
    # resample to the 10 Hz grid used everywhere (nearest source frame)
    step = fps_in / fps_out
    k = np.round(df["frame"] / step)
    df = df.assign(_dt=np.abs(df["frame"] / step - k), frame=k.astype(int), id=df["id"].astype(str))
    df = df.sort_values("_dt").drop_duplicates(["frame", "id"]).drop(columns="_dt")
    return df.sort_values(["frame", "id"]).reset_index(drop=True)


def squad_from_labels(labels: pd.DataFrame, home_label: str = "left") -> pd.DataFrame:
    """Squad CSV rows (team, number, name, player_id) derived from labelled jerseys, for the V2 run."""
    p = labels[labels["role"].isin(["player", "goalkeeper"]) & labels["jersey"].notna()]
    g = p.groupby("id").agg(team=("team", lambda s: s.mode().iloc[0]), jersey=("jersey", lambda s: s.mode().iloc[0]))
    g["team"] = np.where(g["team"] == home_label, "home", "away")
    g["number"] = pd.to_numeric(g["jersey"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["number"]).drop_duplicates(["team", "number"])
    g["player_id"] = g["team"] + "_" + g["number"].astype(str)
    g["name"] = g["player_id"]
    return g.reset_index()[["team", "number", "name", "player_id", "id"]].rename(columns={"id": "gt_id"})
