#!/usr/bin/env python3
"""Fit the default xT grid from all bundled open-data events (SkillCorner provider events + Metrica).

    python scripts/fit_xt.py            # writes packages/analytics/fmi_analytics/data/xt_grid.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in ("packages/schema", "packages/analytics"):
    sys.path.insert(0, str(ROOT / p))

from fmi_analytics.xt import DEFAULT_PATH, GRID, fit_xt, normalized_events  # noqa: E402
from fmi_schema.adapters import metrica, skillcorner  # noqa: E402


def main() -> int:
    frames = []
    sc_root = ROOT / "data/opendata/skillcorner"
    for m in skillcorner.list_matches(sc_root):
        pkg = skillcorner.load_match(sc_root, m["id"], with_tracking=False)
        frames.append(normalized_events(pkg.events, pkg.meta))
        print("skillcorner", m["id"], len(pkg.events))
    for g in (1, 2):
        try:
            pkg = metrica.load_match(ROOT / "data/opendata/metrica", g)
            frames.append(normalized_events(pkg.events, pkg.meta))
            print("metrica", g, len(pkg.events))
        except FileNotFoundError:
            pass
    ev = pd.concat(frames, ignore_index=True)
    grid = fit_xt(ev, 105.0, 68.0, GRID)
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PATH.write_text(json.dumps({"grid_shape": list(GRID), "pitch": [105.0, 68.0], "n_events": int(len(ev)),
                                        "sources": "skillcorner open data + metrica sample", "grid": grid.round(5).tolist()}))
    print(f"wrote {DEFAULT_PATH} from {len(ev)} events; max xT {grid.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
