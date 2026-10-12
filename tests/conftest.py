"""Shared fixtures. Paths come from ``[tool.pytest.ini_options] pythonpath`` in pyproject.toml."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("packages/schema", "packages/analytics", "packages/agent", "workers/pipeline", "apps/api", "eval"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

from fmi_schema import MatchMeta, MatchPackage, PeriodInfo, TeamInfo  # noqa: E402
from fmi_schema.models import TRACKING_COLUMNS  # noqa: E402
from fmi_schema.package import conform  # noqa: E402

DATA = ROOT / "data" / "opendata"


def make_meta(match_id="test", periods=1, seconds=60) -> MatchMeta:
    ps = [PeriodInfo(period=i + 1, start_frame=i * (seconds * 10 + 600), end_frame=i * (seconds * 10 + 600) + seconds * 10,
                     home_attack_sign=1 if i % 2 == 0 else -1) for i in range(periods)]
    return MatchMeta(match_id=match_id, source="synthetic", home=TeamInfo(side="home", name="Home FC"),
                     away=TeamInfo(side="away", name="Away FC"), periods=ps)


def tracking_from_paths(paths: dict[str, tuple[str, np.ndarray]], meta: MatchMeta) -> pd.DataFrame:
    """paths: id -> (team, (T, 2) xy at 10 Hz) for period 1."""
    rows = []
    for oid, (team, xy) in paths.items():
        T = len(xy)
        rows.append(pd.DataFrame({"frame": np.arange(T), "period": 1, "t": np.arange(T) / 10.0, "id": oid,
                                  "team": team, "x": xy[:, 0], "y": xy[:, 1], "confidence": 1.0, "estimated": False}))
    return conform(pd.concat(rows, ignore_index=True), TRACKING_COLUMNS)


@pytest.fixture
def passing_package() -> MatchPackage:
    """Two home players passing to each other, one away player, a ball; 30 s at 10 Hz."""
    meta = make_meta(seconds=30)
    T = 300
    t = np.arange(T) / 10.0
    a = np.stack([np.full(T, -10.0), np.full(T, 0.0)], 1)
    b = np.stack([np.full(T, 10.0), np.full(T, 5.0)], 1)
    c = np.stack([np.full(T, 0.0), np.full(T, -20.0)], 1)
    ball = np.zeros((T, 2))
    holder = [a, b]
    seg = 50  # every 5 s: 3 s hold, 2 s flight
    for k in range(T // seg):
        src, dst = holder[k % 2], holder[(k + 1) % 2]
        s0 = k * seg
        ball[s0:s0 + 30] = src[s0:s0 + 30] + [0.3, 0]
        for i in range(20):
            if s0 + 30 + i < T:
                w = (i + 1) / 20
                ball[s0 + 30 + i] = (1 - w) * (src[s0] + [0.3, 0]) + w * (dst[s0] + [0.3, 0])
    pkg = MatchPackage(meta=meta, roster=pd.DataFrame({"player_id": ["a", "b", "c"], "team": ["home", "home", "away"],
                                                        "team_name": ["Home FC", "Home FC", "Away FC"],
                                                        "number": [7, 9, 4], "name": ["Ann", "Bea", "Cid"],
                                                        "position": [None, None, "GK"], "starter": [True] * 3}))
    pkg.tracking = tracking_from_paths({"a": ("home", a), "b": ("home", b), "c": ("away", c), "ball": ("ball", ball)}, meta)
    return pkg


@pytest.fixture
def synthetic_run(tmp_path):
    """Run the whole video pipeline on a short synthetic match with fake models."""
    from fmi_pipeline.config import MatchInput, PipelineConfig
    from fmi_pipeline.models import ModelRegistry
    from fmi_pipeline.runner import PipelineRunner
    from fmi_pipeline.storage import LocalStorage

    video = "synthetic://seed=3&seconds=12"
    mi = MatchInput(match_id="synth", video=video, home_name="Reds", away_name="Blues")
    cfg = PipelineConfig()
    runner = PipelineRunner(LocalStorage(tmp_path), mi, cfg, ModelRegistry.for_video(video, cfg))
    outs = runner.run()
    return runner, outs
