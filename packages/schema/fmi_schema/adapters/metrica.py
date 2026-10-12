"""Metrica Sports sample data (github.com/metrica-sports/sample-data) -> Match Package.

Sample Games 1 and 2 use Metrica's CSV format and are parsed natively here.
Sample Game 3 uses EPTS/FIFA format: load it through ``fmi_schema.adapters.kloppy_adapter``.

Metrica coordinates are normalised 0..1 with (0, 0) at the top-left; pitch 105 x 68 m; 25 fps.
We convert to centred metres with y pointing up and resample to 10 Hz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fmi_schema.adapters.common import assign_frames, build_periods, resample_long
from fmi_schema.io import resolve
from fmi_schema.models import EVENT_COLUMNS, TRACKING_COLUMNS, MatchMeta, TeamInfo
from fmi_schema.package import MatchPackage, conform

PITCH = (105.0, 68.0)
FPS_IN = 25.0


def _to_m(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (x - 0.5) * PITCH[0], -(y - 0.5) * PITCH[1]


def read_tracking_csv(path: str | Path, team: str) -> pd.DataFrame:
    """Parse one Metrica tracking CSV (3 header rows) into long format at 25 fps."""
    raw = pd.read_csv(path, header=None, skiprows=3, low_memory=False)
    header = pd.read_csv(path, header=None, nrows=3)
    names = header.iloc[2].tolist()
    period = raw.iloc[:, 0].to_numpy(dtype=int)
    t_abs = raw.iloc[:, 2].to_numpy(dtype=float)
    parts = []
    col = 3
    while col < len(names):
        name = str(names[col])
        if name in ("nan", "None"):
            col += 1
            continue
        x = raw.iloc[:, col].to_numpy(dtype=float)
        y = raw.iloc[:, col + 1].to_numpy(dtype=float)
        xm, ym = _to_m(x, y)
        oid = "ball" if name == "Ball" else name
        parts.append(pd.DataFrame({"period": period, "t_abs": t_abs, "id": oid,
                                   "team": "ball" if oid == "ball" else team, "x": xm, "y": ym}))
        col += 2
    df = pd.concat(parts, ignore_index=True)
    df = df[np.isfinite(df["x"])]
    # period clock: seconds since the first frame of the period
    df["t"] = df["t_abs"] - df.groupby("period")["t_abs"].transform("min")
    return df.drop(columns=["t_abs"])


_TYPE_MAP = {
    "PASS": "pass", "SHOT": "shot", "RECOVERY": "recovery", "BALL LOST": "possession_loss",
    "CHALLENGE": "tackle", "FAULT RECEIVED": "foul",
}
_SET_PIECE_MAP = {
    "KICK OFF": "kick_off", "THROW IN": "throw_in", "CORNER KICK": "corner", "FREE KICK": "free_kick",
    "GOAL KICK": "goal_kick",
}


def read_events_csv(path: str | Path, period_t0: dict[int, float]) -> pd.DataFrame:
    ev = pd.read_csv(path)
    ev.columns = ["team", "type", "subtype", "period", "start_frame", "start_time", "end_frame", "end_time",
                  "src", "dst", "sx", "sy", "ex", "ey"][: len(ev.columns)]
    out = []
    for i, r in enumerate(ev.itertuples(index=False)):
        typ = str(r.type).upper()
        sub = str(r.subtype).upper() if pd.notna(r.subtype) else ""
        etype = _TYPE_MAP.get(typ)
        if typ == "SET PIECE":
            etype = next((v for k, v in _SET_PIECE_MAP.items() if k in sub), None)
        if typ == "PASS" and "CROSS" in sub:
            etype = "cross"
        # Metrica logs failed passes as BALL LOST (subtype INTERCEPTION / FORWARD / CROSS ...)
        failed_pass = typ == "BALL LOST" and any(k in sub for k in ("INTERCEPTION", "FORWARD", "CROSS", "GOAL KICK"))
        if failed_pass:
            etype = "cross" if "CROSS" in sub else "pass"
        if typ == "CHALLENGE" and "WON" not in sub:
            continue  # keep only the winning side of a duel
        if etype is None:
            continue
        period = int(r.period)
        t0 = period_t0.get(period, 0.0)
        sx, sy = _to_m(np.array([r.sx]), np.array([r.sy]))
        ex, ey = _to_m(np.array([r.ex]), np.array([r.ey]))
        outcome = "unknown"
        if etype in ("pass", "cross"):
            outcome = "fail" if failed_pass or pd.isna(r.dst) else "success"
        elif etype == "shot":
            outcome = "success" if "GOAL" in sub and "OFF TARGET" not in sub else "fail"
        out.append({
            "event_id": f"m{i}", "period": period,
            "t": float(r.start_time) - t0, "t_end": float(r.end_time) - t0 if pd.notna(r.end_time) else np.nan,
            "type": etype, "team": str(r.team).lower(),
            "player_id": str(r.src) if pd.notna(r.src) else None,
            "receiver_id": str(r.dst) if pd.notna(r.dst) else None,
            "start_x": float(sx[0]), "start_y": float(sy[0]), "end_x": float(ex[0]), "end_y": float(ey[0]),
            "outcome": outcome, "source": "provider", "confidence": 1.0,
        })
    return pd.DataFrame(out)


def load_match(root: str | Path, game: int) -> MatchPackage:
    """Load ``Sample_Game_{game}`` (1 or 2) from ``root`` (the folder holding ``Sample_Game_*`` dirs)."""
    d = Path(root) / f"Sample_Game_{game}"
    home = read_tracking_csv(resolve(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"), "home")
    away = read_tracking_csv(resolve(d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv"), "away")
    ball = home[home["id"] == "ball"]
    long25 = pd.concat([home[home["id"] != "ball"], away[away["id"] != "ball"], ball], ignore_index=True)
    long10 = resample_long(long25, FPS_IN, 10.0, value_cols=("x", "y"))
    long10["z"] = np.nan
    long10["confidence"] = 1.0
    long10["estimated"] = False

    periods = build_periods(long10, 10.0)
    meta = MatchMeta(
        match_id=f"metrica-{game}", source="metrica", competition="Metrica sample (anonymised)",
        home=TeamInfo(side="home", name="Home", jersey_color="#d62728"),
        away=TeamInfo(side="away", name="Away", jersey_color="#1f77b4"),
        pitch_length=PITCH[0], pitch_width=PITCH[1], fps=10.0, video_tier="none", periods=periods,
        provider_ids={"metrica_game": game},
    )
    tracking = conform(assign_frames(long10, meta), TRACKING_COLUMNS)
    roster = (tracking[tracking["team"].isin(["home", "away"])][["id", "team"]].drop_duplicates()
              .rename(columns={"id": "player_id"}))
    roster["team_name"] = roster["team"].str.title()
    roster["number"] = roster["player_id"].str.extract(r"(\d+)").astype(float).iloc[:, 0].astype("Int64")
    roster["name"] = roster["player_id"]
    roster["position"] = None
    first = tracking[tracking["frame"] == tracking["frame"].min()]["id"]
    roster["starter"] = roster["player_id"].isin(set(first))

    # event times are absolute seconds; align them to the same period clock as tracking
    raw_home = pd.read_csv(resolve(d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"),
                           header=None, skiprows=3, usecols=[0, 2], low_memory=False)
    t0 = raw_home.groupby(0)[2].min().to_dict()
    events = read_events_csv(resolve(d / f"Sample_Game_{game}_RawEventsData.csv"), {int(k): float(v) for k, v in t0.items()})
    if not events.empty:
        events["frame"] = [meta.period(int(p)).start_frame + int(round(t * 10)) if meta.period(int(p)) else pd.NA
                           for p, t in zip(events["period"], events["t"])]
    return MatchPackage(meta=meta, roster=roster, tracking=tracking, events=conform(events, EVENT_COLUMNS))
