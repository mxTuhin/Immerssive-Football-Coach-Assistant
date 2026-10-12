# Local setup (after `git pull`)

Nothing large is committed except gzipped open data. Models and videos are flagged with `*.download` files.

## 1. CPU dev environment (API, analytics, tests)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
pip install --no-deps -e .
pytest -q                                   # runs everything with fake models
make synth                                  # whole video pipeline on a synthetic match
make opendata                               # SkillCorner match -> Match Package (analytics, facing, events)
```

## 2. Data and weights

```bash
python scripts/fetch_data.py --list         # what is missing
python scripts/fetch_data.py --models       # roboflow/sports weights (gdown)
python scripts/fetch_data.py --videos       # V1 clip
# V2: accept conditions on https://huggingface.co/datasets/atomscott/soccertrack-v2, then
huggingface-cli login
python scripts/fetch_data.py --soccertrack --list-remote
python scripts/fetch_data.py --soccertrack --pattern "<one half + labels>"
# V3: pick a lower-tier full match (see data/videos/v3_lower_tier_match.mp4.download)
```

## 3. GPU worker

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements/vision.txt
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
fmi run --match-id v1 --video data/videos/v1_121364_0.mp4 --tier broadcast
fmi run --match-id v2 --video data/soccertrack/<half>.mp4 --tier panoramic --segment 0 600   # first 10 min
```

Colab/Kaggle work too for short clips: install `requirements/vision.txt`, mount the repo, run the same CLI.

## 4. Server (API + queue)

```bash
# quickest: everything inline, SQLite, no Redis
FMI_CELERY_EAGER=1 uvicorn fmi_api.main:app --reload        # http://localhost:8000/docs
# full stack
cp infra/.env.example infra/.env   # edit
docker compose -f infra/docker-compose.yml --env-file infra/.env up -d api worker-cpu
docker compose -f infra/docker-compose.yml --env-file infra/.env --profile gpu up -d worker-gpu
```

Typical flow:

```bash
curl -X POST localhost:8000/matches -H 'content-type: application/json' \
  -d '{"match_id":"demo","home_name":"Reds","away_name":"Blues","home_color":"#d62728","away_color":"#1f77b4"}'
curl -X POST localhost:8000/matches/demo/video -F file=@data/videos/v1_121364_0.mp4
curl -X POST localhost:8000/matches/demo/squad -F file=@squad.csv
curl -X POST localhost:8000/matches/demo/process
curl localhost:8000/jobs/<job_id>
curl localhost:8000/matches/demo/key-moments
curl -X POST localhost:8000/opendata/import -H 'content-type: application/json' -d '{"provider":"skillcorner","match":"1886347"}'
```

## 5. AI analyst

Set `ANTHROPIC_API_KEY` (or `ant auth login`). `POST /matches/{id}/report`, `POST /matches/{id}/ask`.
`python -m fmi_agent.evaluate <package_dir>` runs the 30-question test set (costs API credits).
