"""Run the analyst test set on a Match Package and score the automatic checks.

    python -m fmi_agent.evaluate storage/matches/sc-1886347/package --out eval_reports/agent.json

Human rubric scores (correct/useful/actionable, 1-5) are added by a coach in the output JSON afterwards.
Running this calls the Claude API (costs money): it needs ANTHROPIC_API_KEY or an `ant auth login` profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fmi_agent.analyst import MatchAnalyst
from fmi_schema import MatchPackage


def fill(q: str, pkg: MatchPackage) -> str:
    phys = pkg.derived.get("physical")
    player = pkg.player_name(phys.sort_values("distance_m").iloc[-1]["player_id"]) if phys is not None and len(phys) else "the captain"
    return q.format(home=pkg.meta.home.name, away=pkg.meta.away.name, player=player)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--out", default="eval_reports/agent_eval.json")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", default="claude-opus-5-5")
    args = ap.parse_args(argv)
    pkg = MatchPackage.load(args.package)
    qs = yaml.safe_load((Path(__file__).parent / "eval" / "questions.yaml").read_text())["questions"][: args.limit]
    analyst = MatchAnalyst(pkg, model=args.model)
    rows = []
    for q in qs:
        question = fill(q, pkg)
        res = analyst.ask(question)
        g = res.get("grounding") or {}
        rows.append({"question": question, "answer": res.get("answer"), "tool_calls": res.get("tool_calls"),
                     "grounded": bool(g.get("ok")), "ungrounded_numbers": g.get("ungrounded"),
                     "rubric": {"correct": None, "useful": None, "actionable": None}})
        print(f"[{'ok ' if g.get('ok') else 'FAIL'}] {question}")
    summary = {"n": len(rows), "grounded_rate": sum(r["grounded"] for r in rows) / max(len(rows), 1),
               "with_tool_calls": sum(bool(r["tool_calls"]) for r in rows)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "answers": rows}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    return 0 if summary["grounded_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
