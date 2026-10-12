import numpy as np
import pandas as pd

from conftest import make_meta, tracking_from_paths
from fmi_analytics.events import detect_events, event_f1
from fmi_analytics.facing import compute_facing, facing_accuracy
from fmi_analytics.kinematics import SmoothConfig, smooth_tracking
from fmi_analytics.possession import compute_possession
from fmi_analytics.qa import plausibility
from fmi_analytics.run import run_all


def test_smoother_recovers_velocity_and_flags_gaps():
    meta = make_meta(seconds=20)
    T = 200
    rng = np.random.default_rng(1)
    xy = np.stack([np.linspace(-20, 20, T), np.zeros(T)], 1) + rng.normal(0, 0.2, (T, 2))  # 2 m/s
    xy[50:60] = np.nan    # 1 s gap -> filled, estimated
    xy[120:160] = np.nan  # 4 s gap -> not filled
    tr = tracking_from_paths({"p": ("home", xy)}, meta)
    tr = tr[np.isfinite(tr["x"])]
    out, qa = smooth_tracking(tr, SmoothConfig())
    out = out.set_index("frame")
    assert abs(np.median(out["speed"]) - 2.0) < 0.15
    assert out.loc[55, "estimated"] and not out.loc[40, "estimated"]
    assert 140 not in out.index
    assert qa["speed_capped_samples"] == 0


def test_facing_rules():
    meta = make_meta(seconds=10)
    T = 100
    runner = np.stack([np.linspace(-10, 10, T), np.zeros(T)], 1)  # 2 m/s... make it 4 m/s
    runner = np.stack([np.linspace(-20, 20, T), np.zeros(T)], 1)
    stander = np.tile([0.0, -10.0], (T, 1))
    ball = np.tile([0.0, 10.0], (T, 1))  # ball straight "north" of the stander
    tr = tracking_from_paths({"r": ("home", runner), "s": ("away", stander), "ball": ("ball", ball)}, meta)
    tr, _ = smooth_tracking(tr)
    out = compute_facing(tr, meta)
    r = out[(out["id"] == "r") & out["frame"].between(20, 80)]
    s = out[(out["id"] == "s") & out["frame"].between(20, 80)]
    assert np.all(np.abs(r["body_yaw"]) < 10) and (r["yaw_source"] == "motion").all() and r["yaw_conf"].min() >= 0.85
    assert np.all(np.abs(s["body_yaw"] - 90) < 10) and (s["yaw_source"] == "ball").all()
    acc = facing_accuracy(r["body_yaw"], np.zeros(len(r)), r["speed"])
    assert acc["moving_within_tol"] == 1.0


def test_facing_turn_rate_is_capped():
    meta = make_meta(seconds=4)
    T = 40
    xy = np.concatenate([np.stack([np.linspace(0, 10, 20), np.zeros(20)], 1),
                         np.stack([np.linspace(10, 0, 20), np.zeros(20)], 1)])  # instant reversal
    tr, _ = smooth_tracking(tracking_from_paths({"p": ("home", xy)}, meta))
    yaw = compute_facing(tr, meta).sort_values("frame")["body_yaw"].to_numpy()
    step = np.abs((np.diff(yaw) + 180) % 360 - 180)
    assert step.max() <= 72.0 + 1e-6  # 720 deg/s at 10 Hz


def test_possession_and_pass_detection(passing_package):
    pkg = passing_package
    pkg.tracking, _ = smooth_tracking(pkg.tracking)
    poss = compute_possession(pkg.tracking)
    assert set(poss["owner"].dropna()) <= {"a", "b"}
    ev = detect_events(pkg.tracking, poss, pkg.meta)
    passes = ev[ev["type"] == "pass"]
    assert len(passes) >= 4
    assert (passes["outcome"] == "success").all()
    assert set(passes["receiver_id"]) <= {"a", "b"}
    truth = pd.DataFrame({"period": 1, "t": passes["t"] + 0.3, "type": "pass"})
    assert event_f1(ev, truth)["pass"]["f1"] == 1.0


def test_run_all_and_qa(passing_package):
    pkg = run_all(passing_package)
    s = pkg.derived["summary"]
    assert s["possession_share"]["home"] > 0.9
    assert "physical" in pkg.derived and len(pkg.derived["physical"]) == 3
    assert pkg.tracking["body_yaw"].notna().any()
    qa = plausibility(pkg.tracking, pkg.meta)
    assert qa.metrics["impossible_jumps"] == 0
