"""Stage registry (DAG order).

    ingest -> detect ----------------------------> ball --------------\\
          \\-> calib -> track -> link -> team -> identity -> project -> facing -> events -> analytics -> package
"""

from fmi_pipeline.stages.ball import Ball
from fmi_pipeline.stages.calib import Calibrate
from fmi_pipeline.stages.detect import Detect
from fmi_pipeline.stages.analytics import Analytics
from fmi_pipeline.stages.events import Events
from fmi_pipeline.stages.facing import Facing
from fmi_pipeline.stages.package import Package
from fmi_pipeline.stages.identity import Identity
from fmi_pipeline.stages.ingest import Ingest
from fmi_pipeline.stages.link import Link
from fmi_pipeline.stages.project import Project
from fmi_pipeline.stages.team import Team
from fmi_pipeline.stages.track import Track

DEFAULT_STAGES = [Ingest, Detect, Calibrate, Track, Link, Team, Identity, Ball, Project, Facing, Events, Analytics,
                  Package]

__all__ = ["DEFAULT_STAGES"]
