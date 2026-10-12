# SkillCorner open data (bundled)

From https://github.com/SkillCorner/opendata (MIT licence, `LICENSE`). Layout kept as upstream except that large
text files are gzipped: `matches/<id>/<id>_tracking_extrapolated.jsonl.gz`, `<id>_dynamic_events.csv.gz`,
`<id>_phases_of_play.csv.gz`. Load with `fmi_schema.adapters.skillcorner.load_match(root, match_id)`.

Conventions verified on this data: tracking is centred metres with y up at 10 fps; `home_team_side[i] ==
"left_to_right"` means the home team attacks +x in period i+1; dynamic-event coordinates are normalised to the
acting team attacking left-to-right (the adapter flips them back using `attacking_side`).
