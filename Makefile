PY ?= python
export PYTHONPATH := packages/schema:packages/analytics:packages/agent:workers/pipeline:apps/api:eval

.PHONY: install install-gpu test lint data models api worker synth opendata eval-opendata fit-xt

install:            ## CPU dev install (API, analytics, tests)
	pip install -r requirements/dev.txt && pip install --no-deps -e .

install-gpu:        ## GPU worker install (pretrained vision models)
	pip install -r requirements/all.txt && pip install git+https://github.com/KaiyangZhou/deep-person-reid.git && pip install --no-deps -e .

test:
	pytest -q

lint:
	ruff check packages workers apps eval scripts tests

data:               ## list flagged downloads
	$(PY) scripts/fetch_data.py --list

models:             ## download pretrained weights
	$(PY) scripts/fetch_data.py --models

api:
	uvicorn fmi_api.main:app --reload

worker:
	celery -A fmi_api.tasks worker -Q gpu,cpu -c 1 --loglevel=info

synth:              ## end-to-end pipeline on a synthetic match (no GPU needed)
	$(PY) -m fmi_pipeline.cli run --match-id synth --video "synthetic://seed=1&seconds=20"

opendata:           ## SkillCorner match -> Match Package with analytics + facing
	$(PY) -m fmi_pipeline.cli opendata skillcorner 1886347

eval-opendata:
	$(PY) -m fmi_eval.run opendata --matches 1886347 1899585

fit-xt:
	$(PY) scripts/fit_xt.py
