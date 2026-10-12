---
name: fetch-data
description: Download or add large assets (videos, model weights, SoccerTrack v2, open data) for FMI. Explains the repo rule - files under 100 MB are committed (gzip text first), larger or gated/blocked ones get a *.download flag file fetched by scripts/fetch_data.py.
---

# Data and weights

## Rule for adding data
- **< 100 MB per file** (after gzip for text like CSV/JSONL): commit it under `data/` and document the source and
  licence in that folder's README.
- **≥ 100 MB, gated, or not downloadable in CI/cloud**: do not commit. Create `<target>.download` next to where the
  file belongs, with `name, target, method (gdown|http|huggingface|manual|auto), url, purpose, group, required`
  and comment lines explaining manual steps. `.gitignore` already excludes everything else in `data/videos`,
  `data/soccertrack`, `models`.

## Fetching
```bash
python scripts/fetch_data.py --list
python scripts/fetch_data.py --models          # roboflow/sports detector, ball, pitch keypoints
python scripts/fetch_data.py --videos [--optional]
python scripts/fetch_data.py --soccertrack --list-remote
python scripts/fetch_data.py --soccertrack --pattern "<one half + labels>"
python scripts/fetch_data.py --only v1_121364_0.mp4 --force
```
Google Drive (gdown) can hit quota limits: retry later or download in a browser to the `target` path.

## Bundled open data
`data/opendata/skillcorner` (20 matches, tracking gzipped, SHA-verified vs LFS) and `data/opendata/metrica`.
Readers open `.gz` transparently. To refresh SkillCorner: clone github.com/SkillCorner/opendata without LFS, fetch
each `*_tracking_extrapolated.jsonl` from `https://media.githubusercontent.com/media/SkillCorner/opendata/master/<path>`,
check the sha256 against the LFS pointer, gzip -9.
