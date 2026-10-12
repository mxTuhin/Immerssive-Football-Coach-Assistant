"""Stage contract (build plan section 5).

* ``run(ctx, inputs) -> StageResult``; idempotent; cached by input hash.
* Parquet/JSON artifacts in storage; no raw frames passed between stages (stages that need pixels read the
  proxy video themselves).
* Every stage writes confidence + a QA record.
* Any stage can be re-run alone after a correction: corrections a stage consumes are part of its hash.
* Unit tests use fake models, so CI runs without a GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

from fmi_pipeline.config import MatchInput, PipelineConfig
from fmi_schema.models import StageQA


@dataclass
class StageOutput:
    """A completed stage: where its artifacts live and the hash that identifies them."""

    stage: str
    hash: str
    dir: Path
    artifacts: dict[str, str]
    qa: StageQA | None = None
    cached: bool = False

    def path(self, name: str) -> Path:
        if name not in self.artifacts:
            raise KeyError(f"stage {self.stage} has no artifact {name!r} (has {sorted(self.artifacts)})")
        return self.dir / self.artifacts[name]


@dataclass
class StageResult:
    artifacts: dict[str, str]
    qa: StageQA


@dataclass
class StageContext:
    match_id: str
    match_input: MatchInput
    config: PipelineConfig
    models: Any                      # fmi_pipeline.models.ModelRegistry
    workspace: Path                  # storage/matches/<id>
    out_dir: Path                    # this stage's output dir
    corrections: dict[str, Any] = field(default_factory=dict)
    progress: Callable[[str, float, str], None] | None = None

    def report(self, stage: str, frac: float, msg: str = "") -> None:
        if self.progress:
            self.progress(stage, frac, msg)

    def source_path(self, name: str) -> Path:
        return self.workspace / "source" / name


class Stage:
    name: ClassVar[str]
    version: ClassVar[str] = "1"
    deps: ClassVar[tuple[str, ...]] = ()
    #: correction kinds this stage consumes (see fmi_pipeline.corrections)
    corrections: ClassVar[tuple[str, ...]] = ()
    #: config sections included in the hash
    config_sections: ClassVar[tuple[str, ...]] = ()
    #: whether the stage reads the source video / match input (included in the hash)
    uses_source: ClassVar[bool] = False

    def run(self, ctx: StageContext, inputs: dict[str, StageOutput]) -> StageResult:  # pragma: no cover
        raise NotImplementedError

    def qa(self, **metrics) -> StageQA:
        return StageQA(stage=self.name, version=self.version, metrics=metrics)

    def input_hash(self, ctx: StageContext, inputs: dict[str, StageOutput], source_digest: str) -> str:
        payload = {
            "stage": self.name,
            "version": self.version,
            "config": {s: ctx.config.section(s) for s in self.config_sections},
            "deps": {k: v.hash for k, v in sorted(inputs.items())},
            "corrections": {k: ctx.corrections.get(k) for k in self.corrections},
            "source": source_digest if self.uses_source else None,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
