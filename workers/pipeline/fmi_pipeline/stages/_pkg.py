"""Helpers for the downstream stages (facing, events, analytics, package)."""

from __future__ import annotations

import pandas as pd

from fmi_pipeline.stages._util import load
from fmi_pipeline.stages.project import load_meta
from fmi_schema.package import MatchPackage


def _pkg(project_out, tracking: pd.DataFrame | None = None, events: pd.DataFrame | None = None) -> MatchPackage:
    pkg = MatchPackage(meta=load_meta(project_out), roster=load(project_out, "roster"))
    pkg.tracking = tracking if tracking is not None else load(project_out, "tracking")
    if events is not None:
        pkg.events = events
    return pkg
