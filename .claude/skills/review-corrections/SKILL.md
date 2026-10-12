---
name: review-corrections
description: Work with the FMI review queue and human corrections (merge/split tracks, assign players, fix teams, add calibration points, edit events) and the partial re-run logic that recomputes only affected stages.
---

# Review queue and corrections

- Every stage's `StageQA.review` items are merged into `derived/review_queue.json` (sorted by `priority`,
  0 = urgent). Kinds: identity, team, merge, split, calibration, event, ball, plausibility, facing.
- Corrections are stored per match in `storage/matches/<id>/corrections.json` (`fmi_pipeline/corrections.py`):

| kind | value | stage re-run from |
|---|---|---|
| `calibration_points` | `[{"frame", "image": [[u,v]x4+], "pitch": [[x,y]x4+]}]` | calib |
| `split` | `[{"track_id", "frame"}]` | link |
| `merge` | `[{"track_ids": [...]}]` | link |
| `team` | `{"<track_id>": "home"|"away"|"referee"}` | team |
| `assign` | `{"<track_id>": "<player_id>"}` | identity |
| `event` | `[{"op": "add"|"delete"|"edit", "event_id"?, "event"?}]` | events |

- Re-run: `POST /matches/{id}/rerun` or `fmi run --match-id <id>`; the runner recomputes the consuming stage and its
  descendants only (hash-based). `fmi plan --match-id <id>` / the corrections response lists what will re-run.
- Track ids in corrections are the long-track ids after linking (`L<n>`); split ids become `<id>s<frame>`.
- Target (plan §2 #2): ≥ 95% player-minutes correct after ≤ 20 min review — log review time per match.
- Corrections are the future training data (plan §4 "corrections feed retraining"): never delete the log.
