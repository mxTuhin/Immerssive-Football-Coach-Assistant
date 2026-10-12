---
name: pipeline-stage
description: Add, change or debug a stage of the FMI video pipeline (ingest, detect, calib, track, link, team, identity, ball, project, facing, events, analytics, package) while keeping the stage contract - idempotent, hash-cached, QA record, re-runnable after corrections, testable with fake models.
---

# Working on a pipeline stage

Code: `workers/pipeline/fmi_pipeline/` (contract in `contract.py`, runner in `runner.py`, stages in `stages/`).

## Contract checklist (build plan §5)
1. Subclass `Stage`; set `name`, `version`, `deps`, `config_sections`, `corrections`, `uses_source`.
2. `run(ctx, inputs) -> StageResult(artifacts, qa)`:
   - read inputs only through `inputs[dep].path(name)` / `stages._util.load()`;
   - write artifacts into `ctx.out_dir` (Parquet via `stages._util.save`, JSON via `fmi_schema.io.dump_json`);
   - never pass frames between stages: stages that need pixels read the proxy (`_util.proxy(inputs["ingest"])`);
   - always return a `StageQA` with metrics, warnings and `ReviewItem`s (one per problem *segment*, not per frame).
3. Anything that changes outputs must be in the hash: config → a `PipelineConfig` section listed in
   `config_sections`; human fixes → a correction kind in `corrections` (map it in `corrections.KIND_TO_STAGE`);
   logic change → bump `version`.
4. Register the class in `stages/__init__.py` (`DEFAULT_STAGES`, keep DAG order) and add it to `Package.deps`.
5. Use models only through `ctx.models.<name>` (interfaces in `models/base.py`). Add a fake in `models/fakes.py`
   driven by `SyntheticScene` so CI covers it, and a real wrapper in `models/pretrained.py` with lazy imports.
6. Units: pitch metres, centred origin, angles in degrees CCW from +x (see `docs/ARCHITECTURE.md`).

## Test it
```bash
make synth                                   # full pipeline, synthetic video, fake models
fmi plan --match-id synth                    # which stages re-run vs cache
pytest tests/test_pipeline_e2e.py -q
```
Add a focused unit test (see `tests/test_geometry_tracking.py`) plus an assertion in the e2e test.

## Gotchas
- `frame` before `project` = proxy-video frame (25 fps); after `project` = global 10 Hz frame.
- Shared tracking algorithms belong in `fmi_analytics` (they must also run on open data), not in the stage.
- Keep stages CPU-importable: import torch/ultralytics/cv2 inside functions or `models/pretrained.py`.
