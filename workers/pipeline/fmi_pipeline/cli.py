"""Command line for developers (coaches never need it: the API + worker run the same code).

    fmi run --match-id demo --video data/videos/v1_121364_0.mp4 --squad squad.csv
    fmi run --match-id synth --video "synthetic://seed=1&seconds=20"
    fmi plan --match-id demo                       # which stages would run vs hit cache
    fmi correct --match-id demo --kind team --value '{"L3": "away"}'
    fmi opendata skillcorner 1886347 --out storage/opendata
    fmi opendata metrica 1 --out storage/opendata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from fmi_pipeline import corrections
from fmi_pipeline.config import MatchInput, PipelineConfig
from fmi_pipeline.models import ModelRegistry
from fmi_pipeline.runner import PipelineRunner
from fmi_pipeline.storage import LocalStorage, storage_from_env


def _progress(stage: str, frac: float, msg: str) -> None:
    print(f"[{frac * 100:5.1f}%] {stage:10s} {msg}", file=sys.stderr)


def _load_input(storage, args) -> MatchInput:
    if args.video:
        return MatchInput(match_id=args.match_id, video=args.video, squad_csv=args.squad,
                          home_name=args.home, away_name=args.away, home_color=args.home_color,
                          away_color=args.away_color, video_tier=args.tier,
                          segment=tuple(args.segment) if args.segment else None)
    p = storage.match_dir(args.match_id) / "source" / "match_input.json"
    if not p.exists():
        sys.exit(f"no match input for {args.match_id}; pass --video")
    return MatchInput.model_validate_json(p.read_text())


def _config(path: str | None) -> PipelineConfig:
    if not path:
        return PipelineConfig()
    import yaml

    return PipelineConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fmi")
    ap.add_argument("--storage", default=None, help="storage root (default $FMI_STORAGE_ROOT or ./storage)")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("run", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--match-id", required=True)
        p.add_argument("--video")
        p.add_argument("--squad")
        p.add_argument("--home", default="Home")
        p.add_argument("--away", default="Away")
        p.add_argument("--home-color")
        p.add_argument("--away-color")
        p.add_argument("--tier", default="broadcast", choices=["broadcast", "panoramic", "follow_cam", "static"])
        p.add_argument("--segment", nargs=2, type=float, metavar=("START_S", "END_S"))
        p.add_argument("--config", help="YAML overriding PipelineConfig")
        p.add_argument("--targets", nargs="*")
        p.add_argument("--force", nargs="*", default=[])

    c = sub.add_parser("correct")
    c.add_argument("--match-id", required=True)
    c.add_argument("--kind", required=True, choices=sorted(corrections.KIND_TO_STAGE))
    c.add_argument("--value", required=True, help="JSON")

    o = sub.add_parser("opendata")
    o.add_argument("provider", choices=["skillcorner", "metrica"])
    o.add_argument("match")
    o.add_argument("--root", default=None)
    o.add_argument("--out", default="storage/opendata")
    o.add_argument("--events", default="best", choices=["best", "auto", "provider"])

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    storage = LocalStorage(args.storage) if args.storage else storage_from_env()

    if args.cmd in ("run", "plan"):
        mi = _load_input(storage, args)
        cfg = _config(args.config)
        models = ModelRegistry.for_video(mi.video, cfg)
        runner = PipelineRunner(storage, mi, cfg, models, progress=_progress)
        if args.cmd == "plan":
            print(json.dumps(runner.plan(args.targets), indent=2))
            return 0
        outs = runner.run(targets=args.targets, force=set(args.force))
        summary = {k: {"hash": v.hash, "cached": v.cached, "ok": v.qa.ok if v.qa else None,
                       "metrics": v.qa.metrics if v.qa else {}} for k, v in outs.items()}
        print(json.dumps(summary, indent=2, default=str))
        return 0
    if args.cmd == "correct":
        ws = storage.match_dir(args.match_id)
        corrections.add(ws, args.kind, json.loads(args.value), author="cli")
        print(f"saved; affected stage: {corrections.KIND_TO_STAGE[args.kind]} (re-run `fmi run --match-id {args.match_id}`)")
        return 0
    if args.cmd == "opendata":
        from fmi_analytics.run import run_all

        if args.provider == "skillcorner":
            from fmi_schema.adapters import skillcorner

            pkg = skillcorner.load_match(args.root or "data/opendata/skillcorner", args.match)
        else:
            from fmi_schema.adapters import metrica

            pkg = metrica.load_match(args.root or "data/opendata/metrica", int(args.match))
        run_all(pkg, events_source=args.events)
        out = Path(args.out) / pkg.match_id
        pkg.save(out)
        print(json.dumps({"saved": str(out), "summary": pkg.derived.get("summary")}, indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
