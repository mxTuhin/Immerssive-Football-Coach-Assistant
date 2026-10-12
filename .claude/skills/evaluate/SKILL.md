---
name: evaluate
description: Measure FMI quality against the build-plan targets - SoccerTrack v2 ground truth (position error, IDF1, ID switches, identity %, ball, event F1), label-free QA for unlabelled videos, facing accuracy from hand labels, open-data event regression, and the AI analyst test set.
---

# Evaluation (build plan §8)

| Target (plan §2) | Command |
|---|---|
| Median position error ≤ 1.5 m, IDF1, ID switches/10 min, ≥ 85% player-minutes identified, ball within 2 m | `python -m fmi_eval.run v2 --package storage/matches/v2/package --labels <labels> [--squad-map map.csv] [--frame-offset N]` |
| Pass/shot F1 ≥ 0.7 at ±1 s | `v2 --events truth_events.csv`; open data: `python -m fmi_eval.run opendata --matches 1886347 1899585` |
| Facing within ±45° on ≥ 70% of moving players | `python -m fmi_eval.run facing --package ... --labels eval/labels/facing_v2.csv` |
| V1/V3 label-free QA + 200-frame hand check list | `python -m fmi_eval.run qa --package ...` |
| AI analyst: 30 questions, no number without a tool call, coach rubric ≥ 4/5 | `python -m fmi_agent.evaluate <package>` (calls the API) |

## SoccerTrack v2 specifics
- Verify the label format of the release you downloaded; `fmi_eval/soccertrack.py` auto-detects SoccerNet-GSR JSON
  or a long CSV (pass a column mapping). Check origin (`--origin corner`) and y direction (`--flip-y`) by plotting a
  frame: the GT and our tracking must overlap.
- Build the squad CSV from labels with `soccertrack.squad_from_labels(labels)`; save `gt_id,player_id` as the
  `--squad-map` for the identity metric.
- Align time: if the pipeline used `--segment a b`, labels start at video time a → use `--frame-offset`.

## Facing labels
Use `eval/labels/facing_template.csv`; 300 samples, ~half moving (> 2 m/s), 8 directions allowed (`true_dir8`).

## CI
`.github/workflows/ci.yml` blocks merges if open-data pass F1 < 0.7. Add the V2 job on a self-hosted GPU runner
once the half-match is downloaded; fail on regression vs the last report in `eval_reports/`.
