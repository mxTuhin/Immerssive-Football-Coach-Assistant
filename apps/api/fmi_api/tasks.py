"""Celery app + tasks. Start a worker with:

    celery -A fmi_api.tasks worker -Q gpu -c 1 --loglevel=info     # GPU box: video pipeline
    celery -A fmi_api.tasks worker -Q cpu -c 4 --loglevel=info     # CPU: open data, reports

With ``FMI_CELERY_EAGER=1`` tasks run inline (no Redis), which is what tests and quick local dev use.
"""

from __future__ import annotations

from fmi_api import service
from fmi_api.settings import get_settings

try:
    from celery import Celery
except ImportError:  # pragma: no cover - API-only installs without celery
    Celery = None  # type: ignore


def _make_app():
    if Celery is None:
        return None
    s = get_settings()
    app = Celery("fmi", broker=s.redis_url, backend=s.redis_url)
    app.conf.update(
        task_always_eager=s.celery_eager,
        task_eager_propagates=False,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_routes={"fmi.pipeline": {"queue": "gpu"}, "fmi.opendata": {"queue": "cpu"},
                     "fmi.report": {"queue": "cpu"}},
        task_time_limit=6 * 3600,
    )
    return app


celery_app = _make_app()


def _task(name):
    def deco(fn):
        if celery_app is None:
            return fn
        return celery_app.task(name=name)(fn)

    return deco


@_task("fmi.pipeline")
def pipeline_task(job_id: str, match_id: str, targets=None, force=None):
    return service.run_pipeline_job(job_id, match_id, targets, force)


@_task("fmi.opendata")
def opendata_task(job_id: str, match_id: str, provider: str, ref: str, events_source: str = "best"):
    return service.run_opendata_job(job_id, match_id, provider, ref, events_source)


@_task("fmi.report")
def report_task(job_id: str, match_id: str):
    return service.run_report_job(job_id, match_id)


def submit(task, *args) -> None:
    """Queue a task, or run it inline when eager / when Celery is not installed."""
    if celery_app is None or get_settings().celery_eager:
        try:
            task(*args) if celery_app is None else task.apply(args=args)
        except Exception:
            pass  # failure already recorded on the Job row
        return
    task.delay(*args)
