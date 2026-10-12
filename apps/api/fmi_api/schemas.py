"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MatchCreate(BaseModel):
    title: str = ""
    match_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_\-]{1,64}$")
    home_name: str = "Home"
    away_name: str = "Away"
    home_color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    away_color: str | None = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    video_tier: Literal["broadcast", "panoramic", "follow_cam", "static"] = "broadcast"
    video_url: str | None = None
    segment: tuple[float, float] | None = None
    periods: list[tuple[float, float]] | None = None
    home_attack_sign_p1: Literal[-1, 1] | None = None
    competition: str | None = None
    date: str | None = None


class MatchOut(BaseModel):
    id: str
    title: str
    source: str
    status: str
    input: dict
    created_at: datetime
    updated_at: datetime
    has_package: bool = False


class JobOut(BaseModel):
    id: str
    match_id: str
    kind: str
    status: str
    progress: float
    stage: str | None
    message: str | None
    params: dict
    result: dict | None
    error: str | None
    created_at: datetime
    updated_at: datetime


class ProcessRequest(BaseModel):
    targets: list[str] | None = None
    force: list[str] | None = None


class ImportUrl(BaseModel):
    url: str


class OpenDataImport(BaseModel):
    provider: Literal["skillcorner", "metrica"]
    match: str
    events_source: Literal["best", "auto", "provider"] = "best"


class CorrectionIn(BaseModel):
    kind: Literal["calibration_points", "split", "merge", "team", "assign", "event"]
    value: Any
    author: str | None = None


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
