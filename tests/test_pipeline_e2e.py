import numpy as np

from fmi_pipeline import corrections
from fmi_pipeline.synthetic import get_scene
from fmi_schema import MatchPackage


def test_full_pipeline_on_synthetic_match(synthetic_run):
    runner, outs = synthetic_run
    assert set(outs) >= {"ingest", "detect", "calib", "track", "link", "team", "identity", "ball", "project",
                         "facing", "events", "analytics", "package"}
    assert outs["calib"].qa.metrics["frames_calibrated_frac"] == 1.0
    assert outs["team"].qa.metrics["home"] == 11 and outs["team"].qa.metrics["away"] == 11
    assert outs["identity"].qa.metrics["player_frames_named_frac"] > 0.85
    pkg = MatchPackage.load(runner.workspace / "package")
    assert pkg.validate() == []
    scene = get_scene(3, 12.0, 25.0)
    pl = pkg.tracking[pkg.tracking["team"].isin(["home", "away"])]
    errs = []
    for r in pl.itertuples():
        vf = int(round(r.t * 25))
        if vf < scene.n_frames and r.id in scene.ids:
            errs.append(np.hypot(*(scene.xy[vf, scene.ids.index(r.id)] - [r.x, r.y])))
    assert np.median(errs) < 0.5  # MVP bar is 1.5 m on real footage
    assert pl["body_yaw"].notna().mean() > 0.95


def test_rerun_after_correction_only_touches_affected_stages(synthetic_run):
    runner, _ = synthetic_run
    assert all(p["cached"] for p in runner.plan())
    corrections.add(runner.workspace, "assign", {"L1": "home_1"})
    plan = {p["stage"]: p["cached"] for p in runner.plan()}
    assert plan["team"] and plan["ball"] and plan["detect"]
    assert not plan["identity"] and not plan["project"] and not plan["package"]
    outs = runner.run()
    assert outs["detect"].cached and not outs["identity"].cached
    assert outs["identity"].qa.metrics["human"] == 1
