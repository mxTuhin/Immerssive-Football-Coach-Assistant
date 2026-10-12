"""Typed records and table column specs of the Match Package."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Table column specs (long format DataFrames, stored as Parquet)
# --------------------------------------------------------------------------------------

#: tracking at 10 Hz. One row per object per frame. ``id`` is a player id, a track id
#: (``"t17"``) before identity is resolved, or ``"ball"``.
TRACKING_COLUMNS: dict[str, str] = {
    "frame": "int64",        # 10 Hz frame index, global over the match
    "period": "int8",
    "t": "float64",          # seconds since the start of the period
    "id": "string",
    "team": "string",        # home | away | referee | ball | unknown
    "x": "float64",
    "y": "float64",
    "z": "float64",          # ball height when known, else NaN
    "vx": "float64",
    "vy": "float64",
    "speed": "float64",      # m/s
    "body_yaw": "float64",   # degrees on the pitch, 0 = +x
    "yaw_conf": "float64",   # 0..1
    "yaw_source": "string",  # motion | ball | blend | hold | play | none
    "confidence": "float64", # 0..1 position confidence
    "estimated": "bool",     # interpolated / extrapolated, not observed
}

ROSTER_COLUMNS: dict[str, str] = {
    "player_id": "string",
    "team": "string",        # home | away
    "team_name": "string",
    "number": "Int64",
    "name": "string",
    "position": "string",
    "starter": "boolean",
}

EVENT_COLUMNS: dict[str, str] = {
    "event_id": "string",
    "period": "int8",
    "t": "float64",
    "t_end": "float64",
    "frame": "Int64",
    "type": "string",        # see EventType
    "team": "string",
    "player_id": "string",
    "receiver_id": "string",
    "start_x": "float64",
    "start_y": "float64",
    "end_x": "float64",
    "end_y": "float64",
    "outcome": "string",     # success | fail | unknown
    "source": "string",      # auto | human | provider
    "confidence": "float64",
}

CALIBRATION_COLUMNS: dict[str, str] = {
    "frame": "int64",        # video frame index (source fps)
    **{f"h{i}{j}": "float64" for i in range(3) for j in range(3)},  # image px -> pitch metres
    "score": "float64",      # line-alignment score 0..1
    "reproj_px": "float64",  # held-out keypoint reprojection error
    "source": "string",      # auto | keyframe | interp | manual | static
}


class EventType(str, Enum):
    """MVP event vocabulary. ``to_spadl`` maps these onto SPADL action types."""

    PASS = "pass"
    CROSS = "cross"
    SHOT = "shot"
    RECOVERY = "recovery"
    INTERCEPTION = "interception"
    TACKLE = "tackle"
    CLEARANCE = "clearance"
    DRIBBLE = "dribble"
    KICK_OFF = "kick_off"
    THROW_IN = "throw_in"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    GOAL_KICK = "goal_kick"
    GOAL = "goal"
    FOUL = "foul"
    POSSESSION_LOSS = "possession_loss"


SET_PIECES = {EventType.KICK_OFF, EventType.THROW_IN, EventType.CORNER, EventType.FREE_KICK, EventType.GOAL_KICK}


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


class TeamInfo(BaseModel):
    side: Literal["home", "away"]
    name: str
    short_name: str | None = None
    provider_id: str | None = None
    jersey_color: str | None = None  # "#rrggbb"
    number_color: str | None = None


class PeriodInfo(BaseModel):
    period: int
    start_frame: int  # 10 Hz frame index
    end_frame: int
    #: +1 if the home team attacks towards +x in this period, -1 otherwise
    home_attack_sign: int = 1

    def attack_sign(self, team: str) -> int:
        if team == "home":
            return self.home_attack_sign
        if team == "away":
            return -self.home_attack_sign
        return 0


class MatchMeta(BaseModel):
    match_id: str
    source: Literal["pipeline", "skillcorner", "metrica", "kloppy", "synthetic"] = "pipeline"
    competition: str | None = None
    season: str | None = None
    date: str | None = None
    venue: str | None = None
    home: TeamInfo
    away: TeamInfo
    score: dict[str, int] | None = None
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    fps: float = 10.0
    video_tier: Literal["broadcast", "panoramic", "follow_cam", "static", "none"] = "none"
    video_fps: float | None = None
    periods: list[PeriodInfo] = Field(default_factory=list)
    provider_ids: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "1.0"

    def period(self, period: int) -> PeriodInfo | None:
        return next((p for p in self.periods if p.period == period), None)

    def team_info(self, side: str) -> TeamInfo:
        return self.home if side == "home" else self.away


class ReviewItem(BaseModel):
    """A thing a human should look at, sorted by priority (low confidence first)."""

    item_id: str
    stage: str
    kind: Literal[
        "identity", "team", "merge", "split", "calibration", "event", "ball", "plausibility", "facing"
    ]
    priority: float = 0.5  # 0 = urgent, 1 = cosmetic
    frame: int | None = None
    t: float | None = None
    period: int | None = None
    track_ids: list[str] = Field(default_factory=list)
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class StageQA(BaseModel):
    """Every stage writes one of these: metrics, warnings, review items."""

    stage: str
    version: str
    ok: bool = True
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    review: list[ReviewItem] = Field(default_factory=list)
