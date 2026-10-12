"""Work executed by Celery workers (or inline in eager mode). Updates Job rows as it goes."""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path

from fmi_api import db
from fmi_api.settings import get_settings

log = logging.getLogger(__name__)


def storage():
    from fmi_pipeline.storage import LocalStorage, S3Storage

    s = get_settings()
    if s.s3_bucket:
        return S3Storage(s.storage_root, bucket=s.s3_bucket, prefix=s.s3_prefix, endpoint_url=s.s3_endpoint)
    return LocalStorage(s.storage_root)


def pipeline_config():
    """PipelineConfig used by both the worker and the correction preview (so cache plans agree)."""
    from fmi_pipeline.config import PipelineConfig

    path = get_settings().pipeline_config
    if not path:
        return PipelineConfig()
    import yaml

    return PipelineConfig.model_validate(yaml.safe_load(Path(path).read_text()))


def package_dir(match_id: str) -> Path:
    return storage().match_dir(match_id) / "package"


@lru_cache(maxsize=8)
def _load_package(path: str, mtime: float):
    from fmi_schema import MatchPackage

    return MatchPackage.load(path)


def load_package(match_id: str):
    d = package_dir(match_id)
    meta = d / "meta.json"
    if not meta.exists():
        return None
    return _load_package(str(d), meta.stat().st_mtime)


def invalidate_package_cache() -> None:
    _load_package.cache_clear()


# ----------------------------------------------------------------------------- job bookkeeping

def _update(job_id: str, **fields) -> None:
    with db.session() as s:
        job = s.get(db.Job, job_id)
        if job is None:
            return
        for k, v in fields.items():
            setattr(job, k, v)
        s.commit()


def _set_match(match_id: str, **fields) -> None:
    with db.session() as s:
        m = s.get(db.Match, match_id)
        if m is None:
            return
        for k, v in fields.items():
            setattr(m, k, v)
        s.commit()


def _progress_cb(job_id: str):
    last = [0.0]

    def cb(stage: str, frac: float, msg: str):
        now = time.time()
        if now - last[0] < 1.0 and msg not in ("done", "cached"):
            return
        last[0] = now
        _update(job_id, stage=stage, progress=round(float(frac), 4), message=msg)

    return cb


# ----------------------------------------------------------------------------- jobs

def run_pipeline_job(job_id: str, match_id: str, targets: list[str] | None = None, force: list[str] | None = None):
    from fmi_pipeline.config import MatchInput
    from fmi_pipeline.models import ModelRegistry
    from fmi_pipeline.runner import PipelineRunner

    _update(job_id, status="running", progress=0.0)
    _set_match(match_id, status="processing")
    try:
        with db.session() as s:
            m = s.get(db.Match, match_id)
            mi = MatchInput.model_validate(m.input)
        if mi.video.startswith(("http://", "https://")):
            mi = mi.model_copy(update={"video": str(download_video(match_id, mi.video, job_id))})
        cfg = pipeline_config()
        runner = PipelineRunner(storage(), mi, cfg, ModelRegistry.for_video(mi.video, cfg), progress=_progress_cb(job_id))
        outs = runner.run(targets=targets, force=set(force or []))
        result = {k: {"hash": v.hash, "cached": v.cached, "ok": v.qa.ok if v.qa else None} for k, v in outs.items()}
        invalidate_package_cache()
        _update(job_id, status="done", progress=1.0, result=result, message="done")
        _set_match(match_id, status="ready")
        return result
    except Exception as exc:
        log.exception("pipeline job %s failed", job_id)
        _update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        _set_match(match_id, status="failed")
        raise


def download_video(match_id: str, url: str, job_id: str | None = None) -> Path:
    import httpx

    dest = storage().match_dir(match_id) / "source" / (Path(url.split("?")[0]).name or "video.mp4")
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if job_id and total:
                    _update(job_id, stage="download", progress=round(0.05 * done / total, 4))
    tmp.rename(dest)
    return dest


def run_opendata_job(job_id: str, match_id: str, provider: str, ref: str, events_source: str = "best"):
    from fmi_analytics.run import run_all
    from fmi_analytics.scenes import build_scene

    _update(job_id, status="running", stage="load", progress=0.05)
    try:
        root = Path(get_settings().opendata_root)
        if provider == "skillcorner":
            from fmi_schema.adapters import skillcorner

            pkg = skillcorner.load_match(root / "skillcorner", ref)
        elif provider == "metrica":
            from fmi_schema.adapters import metrica

            pkg = metrica.load_match(root / "metrica", int(ref))
        else:
            raise ValueError(f"unknown provider {provider!r}")
        pkg.meta.match_id = match_id
        _update(job_id, stage="analytics", progress=0.3)
        run_all(pkg, events_source=events_source)
        pkg.derived["scenes"] = {m["moment_id"]: build_scene(pkg, m["period"], m["t_start"], m["t_end"], label=m["reason"])
                                 for m in pkg.derived.get("key_moments", [])}
        pkg.save(package_dir(match_id))
        storage().persist(package_dir(match_id))
        invalidate_package_cache()
        _update(job_id, status="done", progress=1.0, message="done", result={"summary": pkg.derived.get("summary")})
        _set_match(match_id, status="ready")
    except Exception as exc:
        log.exception("opendata job %s failed", job_id)
        _update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        _set_match(match_id, status="failed")
        raise


def run_report_job(job_id: str, match_id: str):
    from fmi_agent.analyst import MatchAnalyst
    from fmi_schema.io import dump_json

    _update(job_id, status="running", stage="report", progress=0.1)
    try:
        pkg = load_package(match_id)
        if pkg is None:
            raise RuntimeError("match has no package yet; process it first")
        s = get_settings()
        analyst = MatchAnalyst(pkg, model=s.agent_model, effort=s.agent_effort, fallbacks=s.agent_fallbacks)
        report = analyst.report()
        dump_json(report, package_dir(match_id) / "derived" / "ai_report.json")
        invalidate_package_cache()
        _update(job_id, status="done", progress=1.0, result={"moments": len(report.get("key_moments", []))})
    except Exception as exc:
        log.exception("report job %s failed", job_id)
        _update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
