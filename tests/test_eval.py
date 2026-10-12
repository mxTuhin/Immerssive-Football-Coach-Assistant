import numpy as np
import pandas as pd

from fmi_eval.metrics import evaluate_tracking


def _truth(T=50):
    rows = []
    for f in range(T):
        rows += [{"frame": f, "id": "A", "x": f * 0.1, "y": 0.0}, {"frame": f, "id": "B", "x": 0.0, "y": 10.0}]
    return pd.DataFrame(rows)


def test_perfect_tracking():
    t = _truth()
    p = t.replace({"id": {"A": "p1", "B": "p2"}})
    r = evaluate_tracking(p, t, {"A": "p1", "B": "p2"})
    assert r["position"]["median_m"] == 0.0
    assert r["ids"]["idf1"] == 1.0 and r["ids"]["id_switches"] == 0
    assert r["identity"]["player_frames_correct"] == 1.0


def test_id_switch_counted():
    t = _truth()
    p = t.copy()
    p["id"] = np.where(p["id"] == "A", np.where(p["frame"] < 25, "p1", "p3"), "p2")
    r = evaluate_tracking(p, t)
    assert r["ids"]["id_switches"] == 1
    assert r["ids"]["idf1"] < 1.0
