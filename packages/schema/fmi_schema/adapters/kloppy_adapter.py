"""Any kloppy ``TrackingDataset`` -> Match Package (e.g. Metrica Sample Game 3 in EPTS format).

Optional dependency: ``pip install kloppy``. Kloppy's default coordinate system is normalised 0..1 with
the origin top-left (y down); we convert with the dataset's pitch dimensions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fmi_schema.adapters.common import assign_frames, build_periods, resample_long
from fmi_schema.models import TRACKING_COLUMNS, MatchMeta, TeamInfo
from fmi_schema.package import MatchPackage, conform


def from_tracking_dataset(dataset, match_id: str, source: str = "kloppy",
                          pitch: tuple[float, float] = (105.0, 68.0)) -> MatchPackage:
    md = dataset.metadata
    fps_in = float(md.frame_rate)
    df = dataset.to_df()
    home_team, away_team = md.teams[0], md.teams[1]
    side = {}
    rows = []
    for team, label in ((home_team, "home"), (away_team, "away")):
        for p in team.players:
            side[str(p.player_id)] = label
            rows.append({"player_id": str(p.player_id), "team": label, "team_name": team.name,
                         "number": p.jersey_no, "name": p.name or str(p.player_id),
                         "position": str(p.starting_position) if getattr(p, "starting_position", None) else None,
                         "starter": bool(getattr(p, "starting", False))})
    L, W = pitch
    parts = []
    t = df["timestamp"].map(lambda v: v.total_seconds() if hasattr(v, "total_seconds") else float(v)).to_numpy()
    period = df["period_id"].to_numpy()
    ids = sorted({c[:-2] for c in df.columns if c.endswith("_x")})
    for oid in ids:
        x = df[f"{oid}_x"].to_numpy(dtype=float)
        y = df[f"{oid}_y"].to_numpy(dtype=float)
        team = "ball" if oid == "ball" else side.get(oid, "unknown")
        parts.append(pd.DataFrame({"period": period, "t": t, "id": oid, "team": team,
                                   "x": (x - 0.5) * L, "y": -(y - 0.5) * W}))
    long = pd.concat(parts, ignore_index=True)
    long = long[np.isfinite(long["x"])]
    long10 = resample_long(long, fps_in, 10.0, value_cols=("x", "y"))
    long10["z"] = np.nan
    long10["confidence"] = 1.0
    long10["estimated"] = False
    meta = MatchMeta(match_id=match_id, source=source if source in ("metrica", "kloppy") else "kloppy",
                     home=TeamInfo(side="home", name=home_team.name), away=TeamInfo(side="away", name=away_team.name),
                     pitch_length=L, pitch_width=W, fps=10.0, periods=build_periods(long10, 10.0))
    tracking = conform(assign_frames(long10, meta), TRACKING_COLUMNS)
    return MatchPackage(meta=meta, roster=pd.DataFrame(rows), tracking=tracking)


def load_metrica_epts(root: str | Path, game: int = 3) -> MatchPackage:
    from kloppy import metrica  # type: ignore

    d = Path(root) / f"Sample_Game_{game}"
    raw = d / f"Sample_Game_{game}_tracking.txt"
    if not raw.exists():
        import gzip
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / raw.name
        with gzip.open(str(raw) + ".gz", "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        raw = tmp
    ds = metrica.load_tracking_epts(meta_data=str(d / f"Sample_Game_{game}_metadata.xml"), raw_data=str(raw))
    return from_tracking_dataset(ds, match_id=f"metrica-{game}", source="metrica")
