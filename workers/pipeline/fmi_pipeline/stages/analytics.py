"""Stage 12: analytics (physical, shape, possession, phases, pressing, xT, key moments) + plausibility QA."""

from __future__ import annotations

import pandas as pd

from fmi_analytics.qa import plausibility
from fmi_analytics.run import run_analytics
from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._pkg import _pkg
from fmi_pipeline.stages._util import load, save
from fmi_schema.io import dump_json


class Analytics(Stage):
    name = "analytics"
    version = "1"
    deps = ("project", "facing", "events")

    def run(self, ctx: StageContext, inputs) -> StageResult:
        ev = load(inputs["events"], "events")
        pkg = _pkg(inputs["project"], load(inputs["facing"], "tracking"), ev)
        poss = load(inputs["events"], "possession")
        derived = run_analytics(pkg, ev, poss)
        arts = {}
        for k, v in derived.items():
            if isinstance(v, pd.DataFrame):
                arts[k] = save(ctx, v, k)
            else:
                dump_json(v, ctx.out_dir / f"{k}.json")
                arts[k] = f"{k}.json"
        qa = plausibility(pkg.tracking, pkg.meta, stage=self.name, version=self.version)
        qa.metrics["key_moments"] = len(derived.get("key_moments", []))
        return StageResult(arts, qa)
