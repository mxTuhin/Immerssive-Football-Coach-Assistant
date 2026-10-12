"""Evaluation entry points.

    # V2 (labelled): position error, IDF1, ID switches/10 min, identity %, ball within 2 m, event F1
    python -m fmi_eval.run v2 --package storage/matches/v2/package --labels data/soccertrack/<labels>.json

    # V1 / V3 (no labels): label-free QA + a frame sample list for the 200-frame hand check
    python -m fmi_eval.run qa --package storage/matches/v1/package

    # Facing: hand labels (CSV: match_id,period,t,player_id,true_deg) vs body_yaw
    python -m fmi_eval.run facing --package ... --labels eval/labels/facing_v2.csv

    # Open-data regression (fast, runs in CI): our auto events vs SkillCorner provider events
    python -m fmi_eval.run opendata --matches 1886347 1899585 --min-pass-f1 0.7

Reports are written as JSON (and a short Markdown) under eval_reports/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fmi_analytics.events import event_f1
from fmi_analytics.facing import facing_accuracy
from fmi_analytics.qa import plausibility
from fmi_eval.metrics import evaluate_tracking
from fmi_schema import MatchPackage
from fmi_schema.io import dump_json

TARGETS = {"position_median_m": 1.5, "player_frames_correct": 0.85, "facing_moving_within_45": 0.70,
           "pass_f1": 0.7, "shot_f1": 0.7}


def _write(report: dict, out: str) -> None:
    dump_json(report, out)
    md = Path(out).with_suffix(".md")
    lines = [f"# Evaluation: {report.get('kind')} {report.get('match_id', '')}", ""]
    for k, v in report.items():
        if k in ("kind", "match_id"):
            continue
        lines.append(f"- **{k}**: `{json.dumps(v, default=str)[:400]}`")
    md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, default=str))


def cmd_v2(args) -> int:
    from fmi_eval.soccertrack import load_labels

    pkg = MatchPackage.load(args.package)
    labels = load_labels(args.labels, flip_y=args.flip_y, origin=args.origin)
    # align label frames to the package frame grid
    labels["frame"] = labels["frame"] + args.frame_offset
    gt_to_player = None
    if args.squad_map:
        sm = pd.read_csv(args.squad_map)
        gt_to_player = dict(zip(sm["gt_id"].astype(str), sm["player_id"].astype(str)))
    rep = {"kind": "v2", "match_id": pkg.match_id,
           "tracking": evaluate_tracking(pkg.tracking[["frame", "id", "x", "y"]], labels, gt_to_player, pkg.meta.fps)}
    if args.events:
        truth_ev = pd.read_csv(args.events)
        rep["events"] = event_f1(pkg.events, truth_ev, types=("pass", "shot"), tol_s=1.0)
    rep["targets"] = TARGETS
    _write(rep, args.out)
    return 0


def cmd_qa(args) -> int:
    pkg = MatchPackage.load(args.package)
    qa = plausibility(pkg.tracking, pkg.meta)
    rng = np.random.default_rng(0)
    frames = np.sort(rng.choice(pkg.tracking["frame"].unique(), size=min(200, pkg.tracking["frame"].nunique()), replace=False))
    rep = {"kind": "label_free_qa", "match_id": pkg.match_id, "metrics": qa.metrics, "warnings": qa.warnings,
           "stage_qa": {k: v.metrics for k, v in pkg.qa.items()},
           "hand_check_frames": frames.tolist()}
    _write(rep, args.out)
    return 0


def cmd_facing(args) -> int:
    pkg = MatchPackage.load(args.package)
    lab = pd.read_csv(args.labels, comment="#")
    dirs = {"E": 0, "NE": 45, "N": 90, "NW": 135, "W": 180, "SW": -135, "S": -90, "SE": -45}
    if "true_dir8" in lab:
        lab["true_deg"] = lab["true_deg"].fillna(lab["true_dir8"].map(dirs))
    lab["frame"] = [pkg.frame_at(int(p), float(t)) for p, t in zip(lab["period"], lab["t"])]
    j = lab.merge(pkg.tracking[["frame", "id", "body_yaw", "speed"]], left_on=["frame", "player_id"],
                  right_on=["frame", "id"], how="left")
    rep = {"kind": "facing", "match_id": pkg.match_id,
           "result": facing_accuracy(j["body_yaw"], j["true_deg"], j["speed"]), "target": TARGETS["facing_moving_within_45"]}
    _write(rep, args.out)
    return 0


def cmd_opendata(args) -> int:
    from fmi_analytics.run import run_all
    from fmi_schema.adapters import skillcorner

    rows = {}
    for mid in args.matches:
        pkg = skillcorner.load_match(args.root, mid)
        run_all(pkg, events_source="provider")
        f1 = pkg.derived.get("auto_vs_provider_f1", {})
        rows[str(mid)] = {"pass_f1": f1.get("pass", {}).get("f1"), "shot_f1": f1.get("shot", {}).get("f1"),
                          "qa": pkg.qa["analytics"].metrics}
        print(mid, rows[str(mid)]["pass_f1"], rows[str(mid)]["shot_f1"], file=sys.stderr)
    mean_pass = float(np.mean([r["pass_f1"] for r in rows.values()]))
    rep = {"kind": "opendata_events", "matches": rows, "mean_pass_f1": mean_pass,
           "mean_shot_f1": float(np.mean([r["shot_f1"] for r in rows.values()]))}
    _write(rep, args.out)
    return 0 if mean_pass >= args.min_pass_f1 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fmi_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v2 = sub.add_parser("v2")
    v2.add_argument("--package", required=True)
    v2.add_argument("--labels", required=True)
    v2.add_argument("--squad-map", help="CSV gt_id,player_id (see soccertrack.squad_from_labels)")
    v2.add_argument("--events", help="CSV of true events (period,t,type)")
    v2.add_argument("--frame-offset", type=int, default=0)
    v2.add_argument("--flip-y", action="store_true")
    v2.add_argument("--origin", default="centre", choices=["centre", "corner"])
    v2.add_argument("--out", default="eval_reports/v2.json")
    qa = sub.add_parser("qa")
    qa.add_argument("--package", required=True)
    qa.add_argument("--out", default="eval_reports/qa.json")
    fc = sub.add_parser("facing")
    fc.add_argument("--package", required=True)
    fc.add_argument("--labels", required=True)
    fc.add_argument("--out", default="eval_reports/facing.json")
    od = sub.add_parser("opendata")
    od.add_argument("--root", default="data/opendata/skillcorner")
    od.add_argument("--matches", nargs="+", default=["1886347"])
    od.add_argument("--min-pass-f1", type=float, default=0.7)
    od.add_argument("--out", default="eval_reports/opendata_events.json")
    args = ap.parse_args(argv)
    return {"v2": cmd_v2, "qa": cmd_qa, "facing": cmd_facing, "opendata": cmd_opendata}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
