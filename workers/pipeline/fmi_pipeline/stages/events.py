"""Stage 11: events from tracking (rules) + human event corrections."""

from __future__ import annotations

import pandas as pd

from fmi_analytics.events import EventConfig, detect_events
from fmi_analytics.possession import PossessionConfig, compute_possession
from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._pkg import _pkg
from fmi_pipeline.stages._util import load, save
from fmi_schema.models import EVENT_COLUMNS
from fmi_schema.package import conform


def apply_event_corrections(events: pd.DataFrame, corr: list[dict]) -> pd.DataFrame:
    ev = events.copy()
    for c in corr or []:
        op = c.get("op")
        if op == "delete":
            ev = ev[ev["event_id"] != c["event_id"]]
        elif op == "edit":
            m = ev["event_id"] == c["event_id"]
            for k, v in (c.get("event") or {}).items():
                if k in ev.columns:
                    ev.loc[m, k] = v
            ev.loc[m, "source"] = "human"
        elif op == "add":
            row = {**(c.get("event") or {}), "source": "human", "confidence": 1.0}
            row.setdefault("event_id", f"h{len(ev)}")
            ev = pd.concat([ev, pd.DataFrame([row])], ignore_index=True)
    return conform(ev.sort_values(["period", "t"]).reset_index(drop=True), EVENT_COLUMNS)


class Events(Stage):
    name = "events"
    version = "1"
    deps = ("project", "facing")
    corrections = ("event",)

    def run(self, ctx: StageContext, inputs) -> StageResult:
        pkg = _pkg(inputs["project"], load(inputs["facing"], "tracking"))
        poss = compute_possession(pkg.tracking, PossessionConfig(fps=pkg.meta.fps))
        ev = detect_events(pkg.tracking, poss, pkg.meta, EventConfig(fps=pkg.meta.fps))
        ev = apply_event_corrections(ev, ctx.corrections.get("event"))
        qa = self.qa(events=len(ev), **{f"n_{k}": int(v) for k, v in ev["type"].value_counts().items()},
                     possession_frames_owned=float(poss["owner"].notna().mean()) if len(poss) else 0.0)
        return StageResult({"events": save(ctx, ev, "events"), "possession": save(ctx, poss, "possession")}, qa)
