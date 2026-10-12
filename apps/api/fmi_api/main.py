"""FastAPI app: matches, uploads, jobs, Match Package data, review & corrections, AI analyst, open data.

Run locally:   uvicorn fmi_api.main:app --reload
OpenAPI docs:  http://localhost:8000/docs
"""

from __future__ import annotations

import io
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from fmi_api import db, service, tasks
from fmi_api.schemas import (
    AskIn,
    CorrectionIn,
    ImportUrl,
    JobOut,
    MatchCreate,
    MatchOut,
    OpenDataImport,
    ProcessRequest,
)
from fmi_api.settings import get_settings
from fmi_schema.io import to_jsonable

app = FastAPI(title="Football Match Intelligence API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in get_settings().cors_origins.split(",")],
                   allow_methods=["*"], allow_headers=["*"])


def auth(authorization: str | None = Header(default=None)) -> None:
    token = get_settings().api_token
    if token and authorization != f"Bearer {token}":
        raise HTTPException(401, "invalid or missing bearer token")


def match_or_404(s: Session, match_id: str) -> db.Match:
    m = s.get(db.Match, match_id)
    if m is None:
        raise HTTPException(404, f"match {match_id} not found")
    return m


def package_or_409(match_id: str):
    pkg = service.load_package(match_id)
    if pkg is None:
        raise HTTPException(409, "match has no Match Package yet (process it first)")
    return pkg


def _match_out(m: db.Match) -> MatchOut:
    return MatchOut(id=m.id, title=m.title, source=m.source, status=m.status, input=m.input or {},
                    created_at=m.created_at, updated_at=m.updated_at,
                    has_package=(service.package_dir(m.id) / "meta.json").exists())


def _job_out(j: db.Job) -> JobOut:
    return JobOut(**{c: getattr(j, c) for c in JobOut.model_fields})


def _df(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    if limit:
        df = df.head(limit)
    return to_jsonable(df.replace({np.nan: None}).to_dict(orient="records"))


def _new_job(s: Session, match_id: str, kind: str, params: dict) -> db.Job:
    job = db.Job(match_id=match_id, kind=kind, params=params)
    s.add(job)
    s.commit()
    return job


# ----------------------------------------------------------------------------- health

@app.get("/health")
def health():
    return {"ok": True}


# ----------------------------------------------------------------------------- matches

@app.post("/matches", response_model=MatchOut, dependencies=[Depends(auth)])
def create_match(body: MatchCreate, s: Session = Depends(db.get_db)):
    mid = body.match_id or f"m-{uuid.uuid4().hex[:10]}"
    if s.get(db.Match, mid):
        raise HTTPException(409, f"match {mid} exists")
    mi = body.model_dump(exclude={"title", "match_id", "video_url"})
    mi.update(match_id=mid, video=body.video_url or "")
    m = db.Match(id=mid, title=body.title or f"{body.home_name} vs {body.away_name}", input=mi,
                 status="uploaded" if body.video_url else "created")
    s.add(m)
    s.commit()
    return _match_out(m)


@app.get("/matches", response_model=list[MatchOut], dependencies=[Depends(auth)])
def list_matches(s: Session = Depends(db.get_db)):
    return [_match_out(m) for m in s.query(db.Match).order_by(db.Match.created_at.desc()).all()]


@app.get("/matches/{match_id}", response_model=MatchOut, dependencies=[Depends(auth)])
def get_match(match_id: str, s: Session = Depends(db.get_db)):
    return _match_out(match_or_404(s, match_id))


@app.post("/matches/{match_id}/video", response_model=MatchOut, dependencies=[Depends(auth)])
def upload_video(match_id: str, file: UploadFile = File(...), s: Session = Depends(db.get_db)):
    m = match_or_404(s, match_id)
    name = re.sub(r"[^A-Za-z0-9_.\-]", "_", Path(file.filename or "video.mp4").name)
    dest = service.storage().match_dir(match_id) / "source" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit = get_settings().max_upload_gb * (1 << 30)
    size = 0
    with open(dest, "wb") as fh:
        while chunk := file.file.read(8 << 20):
            size += len(chunk)
            if size > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "upload too large")
            fh.write(chunk)
    m.input = {**m.input, "video": str(dest)}
    m.status = "uploaded"
    s.commit()
    return _match_out(m)


@app.post("/matches/{match_id}/video/import", response_model=MatchOut, dependencies=[Depends(auth)])
def import_video_url(match_id: str, body: ImportUrl, s: Session = Depends(db.get_db)):
    """Register a URL; the worker downloads it at the start of processing."""
    m = match_or_404(s, match_id)
    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(422, "url must be http(s)")
    m.input = {**m.input, "video": body.url}
    m.status = "uploaded"
    s.commit()
    return _match_out(m)


@app.post("/matches/{match_id}/squad", dependencies=[Depends(auth)])
def upload_squad(match_id: str, file: UploadFile = File(...), s: Session = Depends(db.get_db)):
    from fmi_pipeline.stages.ingest import parse_squad

    m = match_or_404(s, match_id)
    text = file.file.read().decode("utf-8-sig")
    try:
        roster = parse_squad(text, m.input.get("home_name", "Home"), m.input.get("away_name", "Away"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    dest = service.storage().match_dir(match_id) / "source" / "squad.csv"
    dest.write_text(text)
    m.input = {**m.input, "squad_csv": str(dest)}
    s.commit()
    return {"players": len(roster), "home": int((roster["team"] == "home").sum()),
            "away": int((roster["team"] == "away").sum())}


@app.post("/matches/{match_id}/process", response_model=JobOut, dependencies=[Depends(auth)])
def process(match_id: str, body: ProcessRequest | None = None, s: Session = Depends(db.get_db)):
    m = match_or_404(s, match_id)
    if not m.input.get("video"):
        raise HTTPException(409, "upload a video (or import a URL) first")
    body = body or ProcessRequest()
    job = _new_job(s, match_id, "pipeline", body.model_dump())
    tasks.submit(tasks.pipeline_task, job.id, match_id, body.targets, body.force)
    s.refresh(job)
    return _job_out(job)


@app.get("/matches/{match_id}/jobs", response_model=list[JobOut], dependencies=[Depends(auth)])
def match_jobs(match_id: str, s: Session = Depends(db.get_db)):
    return [_job_out(j) for j in s.query(db.Job).filter(db.Job.match_id == match_id).order_by(db.Job.created_at.desc())]


@app.get("/jobs/{job_id}", response_model=JobOut, dependencies=[Depends(auth)])
def get_job(job_id: str, s: Session = Depends(db.get_db)):
    j = s.get(db.Job, job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return _job_out(j)


# ----------------------------------------------------------------------------- open data

@app.get("/opendata/catalog", dependencies=[Depends(auth)])
def opendata_catalog():
    from fmi_schema.adapters import skillcorner

    root = Path(get_settings().opendata_root)
    sc = []
    if (root / "skillcorner").exists():
        for mt in skillcorner.list_matches(root / "skillcorner"):
            sc.append({"match": str(mt["id"]), "date": mt.get("date_time"),
                       "home": (mt.get("home_team") or {}).get("short_name"),
                       "away": (mt.get("away_team") or {}).get("short_name")})
    metrica = [{"match": str(g)} for g in (1, 2) if (root / "metrica" / f"Sample_Game_{g}").exists()]
    return {"skillcorner": sc, "metrica": metrica}


@app.post("/opendata/import", response_model=JobOut, dependencies=[Depends(auth)])
def opendata_import(body: OpenDataImport, s: Session = Depends(db.get_db)):
    mid = f"{'sc' if body.provider == 'skillcorner' else 'metrica'}-{body.match}"
    m = s.get(db.Match, mid)
    if m is None:
        m = db.Match(id=mid, title=f"{body.provider} {body.match}", source=body.provider, status="processing",
                     input={"provider": body.provider, "match": body.match})
        s.add(m)
        s.commit()
    job = _new_job(s, mid, "opendata", body.model_dump())
    tasks.submit(tasks.opendata_task, job.id, mid, body.provider, body.match, body.events_source)
    s.refresh(job)
    return _job_out(job)


# ----------------------------------------------------------------------------- Match Package data

@app.get("/matches/{match_id}/summary", dependencies=[Depends(auth)])
def summary(match_id: str):
    pkg = package_or_409(match_id)
    return to_jsonable({"meta": pkg.meta.model_dump(mode="json"), "summary": pkg.derived.get("summary"),
                        "qa": {k: {"ok": v.ok, "metrics": v.metrics, "warnings": v.warnings} for k, v in pkg.qa.items()}})


@app.get("/matches/{match_id}/roster", dependencies=[Depends(auth)])
def roster(match_id: str):
    return _df(package_or_409(match_id).roster)


@app.get("/matches/{match_id}/events", dependencies=[Depends(auth)])
def events(match_id: str, type: str | None = None, team: str | None = None, period: int | None = None,
           player_id: str | None = None, limit: int = Query(2000, le=20000)):
    ev = package_or_409(match_id).events
    if type:
        ev = ev[ev["type"].isin(type.split(","))]
    if team:
        ev = ev[ev["team"] == team]
    if period:
        ev = ev[ev["period"] == period]
    if player_id:
        ev = ev[(ev["player_id"] == player_id) | (ev["receiver_id"] == player_id)]
    return _df(ev, limit)


@app.get("/matches/{match_id}/metrics/{name}", dependencies=[Depends(auth)])
def metrics(match_id: str, name: str):
    allowed = {"physical", "sprints", "shape_timeseries", "phases", "pressure", "xt_moves", "auto_vs_provider_f1"}
    if name not in allowed:
        raise HTTPException(404, f"unknown metric set; one of {sorted(allowed)}")
    v = package_or_409(match_id).derived.get(name)
    return _df(v) if isinstance(v, pd.DataFrame) else to_jsonable(v)


@app.get("/matches/{match_id}/key-moments", dependencies=[Depends(auth)])
def key_moments(match_id: str):
    return to_jsonable(package_or_409(match_id).derived.get("key_moments", []))


@app.get("/matches/{match_id}/tracking", dependencies=[Depends(auth)])
def tracking_window(match_id: str, period: int, t0: float, t1: float, fps: float = Query(10.0, gt=0, le=25)):
    if t1 <= t0 or t1 - t0 > 120:
        raise HTTPException(422, "window must be 0 < t1 - t0 <= 120 s")
    from fmi_analytics.scenes import build_scene

    return build_scene(package_or_409(match_id), period, t0, t1, fps=fps)


@app.get("/matches/{match_id}/frame", dependencies=[Depends(auth)])
def frame(match_id: str, period: int, t: float):
    from fmi_analytics.scenes import frame_snapshot

    return frame_snapshot(package_or_409(match_id), period, t)


@app.get("/matches/{match_id}/scenes/{moment_id}", dependencies=[Depends(auth)])
def scene(match_id: str, moment_id: str):
    sc = (package_or_409(match_id).derived.get("scenes") or {}).get(moment_id)
    if sc is None:
        raise HTTPException(404, "scene not found")
    return sc


@app.get("/matches/{match_id}/heatmap/{player_id}", dependencies=[Depends(auth)])
def heatmap(match_id: str, player_id: str):
    from fmi_analytics.physical import heatmap as hm

    pkg = package_or_409(match_id)
    return hm(pkg.tracking, player_id, pkg.meta)


@app.get("/matches/{match_id}/export/{name}.csv", dependencies=[Depends(auth)])
def export_csv(match_id: str, name: str):
    pkg = package_or_409(match_id)
    df = {"events": pkg.events, "roster": pkg.roster, "physical": pkg.derived.get("physical")}.get(name)
    if not isinstance(df, pd.DataFrame):
        raise HTTPException(404, "export one of: events, roster, physical")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={match_id}_{name}.csv"})


# ----------------------------------------------------------------------------- review

@app.get("/matches/{match_id}/review", dependencies=[Depends(auth)])
def review_queue(match_id: str, limit: int = 200):
    pkg = package_or_409(match_id)
    q = pkg.derived.get("review_queue")
    if q is None:
        q = sorted([r.model_dump(mode="json") for v in pkg.qa.values() for r in v.review], key=lambda r: r["priority"])
    return q[:limit]


@app.post("/matches/{match_id}/corrections", dependencies=[Depends(auth)])
def add_correction(match_id: str, body: CorrectionIn, s: Session = Depends(db.get_db)):
    from fmi_pipeline import corrections
    from fmi_pipeline.config import MatchInput
    from fmi_pipeline.runner import PipelineRunner

    m = match_or_404(s, match_id)
    if m.source != "pipeline":
        raise HTTPException(409, "corrections apply to matches processed from video")
    ws = service.storage().match_dir(match_id)
    try:
        corrections.add(ws, body.kind, body.value, author=body.author)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    s.add(db.CorrectionLog(match_id=match_id, kind=body.kind, value=body.value, author=body.author))
    s.commit()
    runner = PipelineRunner(service.storage(), MatchInput.model_validate(m.input), service.pipeline_config())
    plan = runner.plan()
    return {"saved": True, "stage": corrections.KIND_TO_STAGE[body.kind],
            "will_rerun": [p["stage"] for p in plan if not p["cached"]]}


@app.post("/matches/{match_id}/rerun", response_model=JobOut, dependencies=[Depends(auth)])
def rerun(match_id: str, s: Session = Depends(db.get_db)):
    """Apply saved corrections: only stages whose inputs changed are recomputed."""
    return process(match_id, ProcessRequest(), s)


# ----------------------------------------------------------------------------- AI analyst

@app.post("/matches/{match_id}/report", response_model=JobOut, dependencies=[Depends(auth)])
def make_report(match_id: str, s: Session = Depends(db.get_db)):
    match_or_404(s, match_id)
    package_or_409(match_id)
    job = _new_job(s, match_id, "report", {})
    tasks.submit(tasks.report_task, job.id, match_id)
    s.refresh(job)
    return _job_out(job)


@app.get("/matches/{match_id}/report", dependencies=[Depends(auth)])
def get_report(match_id: str):
    rep = package_or_409(match_id).derived.get("ai_report")
    if rep is None:
        raise HTTPException(404, "no report yet; POST /matches/{id}/report")
    return rep


@app.post("/matches/{match_id}/ask", dependencies=[Depends(auth)])
def ask(match_id: str, body: AskIn):
    from fmi_agent.analyst import MatchAnalyst

    st = get_settings()
    analyst = MatchAnalyst(package_or_409(match_id), model=st.agent_model, effort=st.agent_effort,
                           fallbacks=st.agent_fallbacks)
    return analyst.ask(body.question)
