"""The Match Package: Parquet tables + JSON, one directory per match.

    <dir>/
      meta.json
      roster.parquet
      calibration.parquet      (pipeline matches only)
      tracking.parquet         10 Hz, long format, see TRACKING_COLUMNS
      events.parquet
      qa.json                  {stage: StageQA}
      derived/*.json|parquet   metrics, key moments, scenes, AI outputs
      body.parquet             reserved for body keypoints (parked, build plan section 10)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fmi_schema.io import dump_json, load_json, read_table, table_exists, write_table
from fmi_schema.models import (
    CALIBRATION_COLUMNS,
    EVENT_COLUMNS,
    ROSTER_COLUMNS,
    TRACKING_COLUMNS,
    MatchMeta,
    StageQA,
)
from fmi_schema.pitch import PitchDims


def empty_frame(columns: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in columns.items()})


def conform(df: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame:
    """Add missing columns with defaults, order columns, coerce dtypes. Extra columns are kept at the end."""
    df = df.copy()
    for col, dtype in columns.items():
        if col not in df.columns:
            if dtype == "bool":
                df[col] = False
            elif dtype.startswith("float"):
                df[col] = np.nan
            else:
                df[col] = pd.Series([pd.NA] * len(df), dtype=dtype if dtype != "int8" else "Int64")
    for col, dtype in columns.items():
        try:
            if dtype in ("int64", "int8") and df[col].isna().any():
                df[col] = df[col].astype("Int64")
            elif dtype == "string":
                df[col] = df[col].astype("string")
            else:
                df[col] = df[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    extra = [c for c in df.columns if c not in columns]
    return df[list(columns) + extra]


@dataclass
class MatchPackage:
    meta: MatchMeta
    roster: pd.DataFrame = field(default_factory=lambda: empty_frame(ROSTER_COLUMNS))
    tracking: pd.DataFrame = field(default_factory=lambda: empty_frame(TRACKING_COLUMNS))
    events: pd.DataFrame = field(default_factory=lambda: empty_frame(EVENT_COLUMNS))
    calibration: pd.DataFrame = field(default_factory=lambda: empty_frame(CALIBRATION_COLUMNS))
    qa: dict[str, StageQA] = field(default_factory=dict)
    derived: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def match_id(self) -> str:
        return self.meta.match_id

    @property
    def pitch(self) -> PitchDims:
        return PitchDims(self.meta.pitch_length, self.meta.pitch_width)

    def players(self) -> pd.DataFrame:
        return self.tracking[~self.tracking["id"].isin(["ball"]) & self.tracking["team"].isin(["home", "away"])]

    def ball(self) -> pd.DataFrame:
        return self.tracking[self.tracking["id"] == "ball"]

    def player_name(self, player_id: str) -> str:
        row = self.roster[self.roster["player_id"] == str(player_id)]
        if row.empty:
            return str(player_id)
        r = row.iloc[0]
        num = r["number"]
        return f"{r['name']} (#{int(num)})" if pd.notna(num) else str(r["name"])

    def frame_at(self, period: int, t: float) -> int:
        """Nearest 10 Hz frame index for a period clock time."""
        p = self.meta.period(period)
        if p is None:
            raise KeyError(f"period {period} not in match")
        return int(round(p.start_frame + t * self.meta.fps))

    # ------------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Return a list of schema problems (empty means valid)."""
        problems = []
        for name, cols in (("tracking", TRACKING_COLUMNS), ("events", EVENT_COLUMNS), ("roster", ROSTER_COLUMNS)):
            df = getattr(self, name)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                problems.append(f"{name}: missing columns {missing}")
        tr = self.tracking
        if not tr.empty:
            if tr.duplicated(["frame", "id"]).any():
                problems.append("tracking: duplicate (frame, id) rows")
            bad_team = set(tr["team"].dropna().unique()) - {"home", "away", "referee", "ball", "unknown"}
            if bad_team:
                problems.append(f"tracking: unknown team labels {sorted(bad_team)}")
        if not self.meta.periods:
            problems.append("meta: no periods")
        return problems

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        dump_json(self.meta.model_dump(mode="json"), d / "meta.json")
        write_table(conform(self.roster, ROSTER_COLUMNS), d / "roster.parquet")
        write_table(conform(self.tracking, TRACKING_COLUMNS), d / "tracking.parquet")
        write_table(conform(self.events, EVENT_COLUMNS), d / "events.parquet")
        if not self.calibration.empty:
            write_table(conform(self.calibration, CALIBRATION_COLUMNS), d / "calibration.parquet")
        dump_json({k: v.model_dump(mode="json") for k, v in self.qa.items()}, d / "qa.json")
        dd = d / "derived"
        for key, value in self.derived.items():
            if isinstance(value, pd.DataFrame):
                write_table(value, dd / f"{key}.parquet")
            else:
                dump_json(value, dd / f"{key}.json")
        return d

    @classmethod
    def load(cls, directory: str | Path, with_derived: bool = True) -> "MatchPackage":
        d = Path(directory)
        meta = MatchMeta.model_validate(load_json(d / "meta.json"))
        pkg = cls(meta=meta)
        for name in ("roster", "tracking", "events", "calibration"):
            p = d / f"{name}.parquet"
            if table_exists(p):
                setattr(pkg, name, read_table(p))
        if (d / "qa.json").exists():
            pkg.qa = {k: StageQA.model_validate(v) for k, v in load_json(d / "qa.json").items()}
        dd = d / "derived"
        if with_derived and dd.exists():
            for f in sorted(dd.iterdir()):
                if f.name.endswith(".json"):
                    pkg.derived[f.name[:-5]] = load_json(f)
                elif f.name.endswith(".parquet") or f.name.endswith(".csv.gz"):
                    key = f.name.split(".")[0]
                    pkg.derived[key] = read_table(f)
        pkg.tracking = conform(pkg.tracking, TRACKING_COLUMNS)
        pkg.events = conform(pkg.events, EVENT_COLUMNS)
        pkg.roster = conform(pkg.roster, ROSTER_COLUMNS)
        return pkg
