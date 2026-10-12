"""Match Package schema: the single data contract shared by pipeline, analytics, agent, API and viewers.

Coordinate convention (everywhere in the codebase):
    * metres, origin at the centre spot
    * x along the pitch length, in [-L/2, +L/2]
    * y along the pitch width,  in [-W/2, +W/2]
    * angles in degrees, 0 = +x, counter-clockwise positive, wrapped to (-180, 180]
Attacking direction per team and period is stored in ``meta.periods``.
"""

from fmi_schema.models import (
    EVENT_COLUMNS,
    TRACKING_COLUMNS,
    CALIBRATION_COLUMNS,
    ROSTER_COLUMNS,
    EventType,
    MatchMeta,
    PeriodInfo,
    ReviewItem,
    StageQA,
    TeamInfo,
)
from fmi_schema.package import MatchPackage
from fmi_schema.pitch import PitchDims

__all__ = [
    "CALIBRATION_COLUMNS",
    "EVENT_COLUMNS",
    "ROSTER_COLUMNS",
    "TRACKING_COLUMNS",
    "EventType",
    "MatchMeta",
    "MatchPackage",
    "PeriodInfo",
    "PitchDims",
    "ReviewItem",
    "StageQA",
    "TeamInfo",
]
