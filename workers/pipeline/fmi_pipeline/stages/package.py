"""Final stage: assemble the Match Package (+ scenes for key moments) and the cross-stage review queue."""

from __future__ import annotations

import json

from fmi_analytics.scenes import build_scene
from fmi_pipeline.contract import Stage, StageContext, StageResult
from fmi_pipeline.stages._pkg import _pkg
from fmi_pipeline.stages._util import load
from fmi_schema.io import read_table, to_jsonable


class Package(Stage):
    """Assemble the Match Package (+ scenes for key moments) and collect every stage's QA."""

    name = "package"
    version = "1"
    deps = ("ingest", "detect", "calib", "track", "link", "team", "identity", "ball", "project", "facing",
            "events", "analytics")

    def run(self, ctx: StageContext, inputs) -> StageResult:
        ev = load(inputs["events"], "events")
        pkg = _pkg(inputs["project"], load(inputs["facing"], "tracking"), ev)
        pkg.calibration = load(inputs["calib"], "calibration")
        an = inputs["analytics"]
        for k, rel in an.artifacts.items():
            p = an.dir / rel
            pkg.derived[k] = json.loads(p.read_text()) if rel.endswith(".json") else read_table(p)
        pkg.derived["identities"] = load(inputs["identity"], "identities")
        pkg.derived["teams"] = load(inputs["team"], "teams")
        pkg.qa = {name: out.qa for name, out in inputs.items() if out.qa is not None}
        # review queue across stages, most urgent first
        review = [r for q in pkg.qa.values() for r in q.review]
        pkg.derived["review_queue"] = to_jsonable(sorted(review, key=lambda r: r.priority))
        scenes = {}
        for m in pkg.derived.get("key_moments", []):
            scenes[m["moment_id"]] = build_scene(pkg, m["period"], m["t_start"], m["t_end"], label=m.get("reason"))
        pkg.derived["scenes"] = scenes
        pkg.save(ctx.out_dir / "match_package")
        pkg.save(ctx.workspace / "package")  # stable location for the API
        qa = self.qa(review_items=len(review), key_moments=len(scenes),
                     schema_problems=len(pkg.validate()))
        qa.warnings += pkg.validate()
        return StageResult({"match_package": "match_package"}, qa)
