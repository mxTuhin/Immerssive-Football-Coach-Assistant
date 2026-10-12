---
name: open-data
description: Load SkillCorner or Metrica open tracking data into the FMI Match Package and run analytics/facing/events on it (the product track that does not need video), including provider coordinate conventions.
---

# Open data → Match Package

```bash
fmi opendata skillcorner 1886347 --out storage/opendata            # CLI
curl -X POST localhost:8000/opendata/import -d '{"provider":"skillcorner","match":"1886347"}' -H 'content-type: application/json'
```
```python
from fmi_schema.adapters import skillcorner, metrica
from fmi_analytics.run import run_all
pkg = skillcorner.load_match("data/opendata/skillcorner", 1886347)   # or metrica.load_match(root, 1)
run_all(pkg, events_source="best")    # provider events when present; auto events kept in derived["auto_events"]
pkg.save("storage/matches/sc-1886347/package")
```

## Conventions (verified)
- SkillCorner tracking: centred metres, y up, 10 fps; `home_team_side[i] == "left_to_right"` → home attacks +x in
  period i+1. Dynamic-event coordinates are normalised to the acting team attacking left-to-right → the adapter
  flips them back with `attacking_side`. Only 22 players are tracked (no referees).
- Metrica: normalised 0..1, origin top-left, y down, 25 fps, 105×68 m → converted and resampled to 10 Hz. Failed
  passes are logged as `BALL LOST` (INTERCEPTION/FORWARD/CROSS) → mapped to failed passes.
- Metrica game 3 (EPTS): `kloppy_adapter.load_metrica_epts` (needs kloppy).

## Adding a provider
Write `fmi_schema/adapters/<provider>.py` returning a `MatchPackage` with `conform(..., TRACKING_COLUMNS)`; set
periods with the correct `home_attack_sign` (check with `common.infer_home_attack_sign`); add a test in
`tests/test_opendata.py` asserting shots happen near the attacked goal.
