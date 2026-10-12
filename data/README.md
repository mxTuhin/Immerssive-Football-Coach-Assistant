# Data

| Folder | What | In git? |
|---|---|---|
| `opendata/skillcorner/` | SkillCorner open data: 20 A-League 2024/25 matches (tracking 10 fps, dynamic events, phases), MIT licence | **Yes** – tracking `.jsonl` files gzipped (~8 MB each, SHA-verified against the LFS originals) |
| `opendata/metrica/` | Metrica Sports sample games 1–3 (tracking 25 fps + events) | **Yes** – large files gzipped |
| `videos/` | V1 smoke clip, optional extra clips, V3 lower-tier match | **No** – `*.download` flag files |
| `soccertrack/` | SoccerTrack v2 one labelled half-match (gated on Hugging Face) | **No** – `*.download` flag file |

Everything that is not committed has a `<target>.download` flag file next to where it belongs.
`python scripts/fetch_data.py --list` shows what is missing; `--models`, `--videos`, `--soccertrack` fetch it.

Readers accept the `.gz` files directly (`fmi_schema.io.open_text`, `pandas.read_csv`).

## Sources and licences
* SkillCorner open data – https://github.com/SkillCorner/opendata (MIT, see `opendata/skillcorner/LICENSE`)
* Metrica sample data – https://github.com/metrica-sports/sample-data ("please acknowledge the source")
* roboflow/sports sample clips and weights – https://github.com/roboflow/sports
* SoccerTrack v2 – https://huggingface.co/datasets/atomscott/soccertrack-v2 (accept the dataset conditions)
