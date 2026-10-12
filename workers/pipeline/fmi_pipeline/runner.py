"""Pipeline runner: resolves the stage DAG, skips stages whose input hash is cached, persists artifacts.

    runner = PipelineRunner(storage, match_input, config, models)
    outputs = runner.run()                      # everything
    outputs = runner.run(targets=["ball"])      # ball and its dependencies only
    outputs = runner.run(force={"team"})        # recompute team (+ everything downstream of it)

After a correction, ``run()`` recomputes exactly the stages whose hash changed: the stage consuming the
correction and its descendants. Everything else is a cache hit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Callable

from fmi_pipeline import corrections as corr
from fmi_pipeline.config import MatchInput, PipelineConfig
from fmi_pipeline.contract import Stage, StageContext, StageOutput
from fmi_pipeline.stages import DEFAULT_STAGES
from fmi_schema.io import dump_json
from fmi_schema.models import StageQA

log = logging.getLogger(__name__)


def topo_order(stages: dict[str, Stage], targets: list[str] | None = None) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()

    def visit(name: str, stack: tuple[str, ...] = ()):
        if name in stack:
            raise ValueError(f"cycle in stage graph: {' -> '.join(stack + (name,))}")
        if name in seen:
            return
        if name not in stages:
            raise KeyError(f"unknown stage {name!r}")
        for d in stages[name].deps:
            visit(d, stack + (name,))
        seen.add(name)
        order.append(name)

    for t in targets or list(stages):
        visit(t)
    return order


def descendants(stages: dict[str, Stage], names: set[str]) -> set[str]:
    out = set(names)
    changed = True
    while changed:
        changed = False
        for n, s in stages.items():
            if n not in out and any(d in out for d in s.deps):
                out.add(n)
                changed = True
    return out


class PipelineRunner:
    def __init__(self, storage, match_input: MatchInput, config: PipelineConfig | None = None, models=None,
                 stages: list[Stage] | None = None,
                 progress: Callable[[str, float, str], None] | None = None):
        self.storage = storage
        self.match_input = match_input
        self.config = config or PipelineConfig()
        self.models = models
        self.stages = {s.name: s for s in (stages or [cls() for cls in DEFAULT_STAGES])}
        self.progress = progress
        self.workspace = storage.match_dir(match_input.match_id)
        (self.workspace / "source").mkdir(parents=True, exist_ok=True)
        dump_json(match_input.model_dump(mode="json"), self.workspace / "source" / "match_input.json")

    # ------------------------------------------------------------------
    def source_digest(self) -> str:
        """Identity of the source: match input + video file size/mtime (cheap, avoids hashing GBs)."""
        h = hashlib.sha256(json.dumps(self.match_input.model_dump(mode="json"), sort_keys=True).encode())
        for p in (self.match_input.video, self.match_input.squad_csv):
            if p and Path(p).exists():
                st = Path(p).stat()
                h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
        return h.hexdigest()[:16]

    def stage_root(self, name: str) -> Path:
        return self.workspace / "stages" / name

    def load_output(self, name: str, h: str) -> StageOutput | None:
        d = self.stage_root(name) / h
        m = d / "manifest.json"
        if not m.exists():
            return None
        man = json.loads(m.read_text())
        qa = StageQA.model_validate(man["qa"]) if man.get("qa") else None
        return StageOutput(stage=name, hash=h, dir=d, artifacts=man["artifacts"], qa=qa, cached=True)

    def latest(self, name: str) -> StageOutput | None:
        p = self.stage_root(name) / "latest.json"
        if not p.exists():
            return None
        return self.load_output(name, json.loads(p.read_text())["hash"])

    # ------------------------------------------------------------------
    def run(self, targets: list[str] | None = None, force: set[str] | None = None) -> dict[str, StageOutput]:
        force = descendants(self.stages, set(force or ()))
        order = topo_order(self.stages, targets)
        corrections = corr.load(self.workspace)
        digest = self.source_digest()
        outputs: dict[str, StageOutput] = {}
        for i, name in enumerate(order):
            stage = self.stages[name]
            inputs = {d: outputs[d] for d in stage.deps}
            out_dir_tmp = self.stage_root(name) / "_tmp"
            ctx = StageContext(match_id=self.match_input.match_id, match_input=self.match_input,
                               config=self.config, models=self.models, workspace=self.workspace,
                               out_dir=out_dir_tmp, corrections=corrections, progress=self.progress)
            h = stage.input_hash(ctx, inputs, digest)
            cached = None if name in force else self.load_output(name, h)
            if cached is not None:
                log.info("stage %s: cache hit (%s)", name, h)
                outputs[name] = cached
                self._report(name, 1.0, "cached", i, len(order))
                continue
            log.info("stage %s: running (%s)", name, h)
            self._report(name, 0.0, "running", i, len(order))
            if out_dir_tmp.exists():
                shutil.rmtree(out_dir_tmp)
            out_dir_tmp.mkdir(parents=True)
            t0 = time.time()
            result = stage.run(ctx, inputs)
            final = self.stage_root(name) / h
            if final.exists():
                shutil.rmtree(final)
            out_dir_tmp.rename(final)
            manifest = {"stage": name, "version": stage.version, "hash": h, "artifacts": result.artifacts,
                        "qa": result.qa.model_dump(mode="json"), "duration_s": round(time.time() - t0, 2),
                        "deps": {k: v.hash for k, v in inputs.items()}}
            dump_json(manifest, final / "manifest.json")
            dump_json({"hash": h}, self.stage_root(name) / "latest.json")
            self.storage.persist(final)
            outputs[name] = StageOutput(stage=name, hash=h, dir=final, artifacts=result.artifacts, qa=result.qa)
            self._report(name, 1.0, "done", i, len(order))
        return outputs

    def _report(self, name: str, frac: float, msg: str, i: int, n: int) -> None:
        if self.progress:
            self.progress(name, (i + frac) / max(n, 1), msg)

    def plan(self, targets: list[str] | None = None) -> list[dict]:
        """Which stages would run vs hit cache, without running anything (used by the review UI)."""
        order = topo_order(self.stages, targets)
        corrections = corr.load(self.workspace)
        digest = self.source_digest()
        fake: dict[str, StageOutput] = {}
        res = []
        for name in order:
            stage = self.stages[name]
            inputs = {d: fake[d] for d in stage.deps}
            ctx = StageContext(match_id=self.match_input.match_id, match_input=self.match_input, config=self.config,
                               models=self.models, workspace=self.workspace, out_dir=Path("."), corrections=corrections)
            h = stage.input_hash(ctx, inputs, digest)
            cached = (self.stage_root(name) / h / "manifest.json").exists()
            fake[name] = StageOutput(stage=name, hash=h, dir=Path("."), artifacts={})
            res.append({"stage": name, "hash": h, "cached": cached})
        return res
