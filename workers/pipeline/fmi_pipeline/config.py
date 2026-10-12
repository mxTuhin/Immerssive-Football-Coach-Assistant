"""Pipeline configuration. Every threshold from the build plan lives here, versioned with the stage hash."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class IngestConfig(BaseModel):
    proxy_fps: float = 25.0
    proxy_max_width: int = 3840     # keep 4K for panoramic (tiling); broadcast is usually 1920
    scene_cut_threshold: float = 0.4


class DetectConfig(BaseModel):
    imgsz: int = 1280
    player_conf: float = 0.1        # low: weak boxes go into the tracker's second pass
    ball_conf: float = 0.05
    stride: int = 1
    tiling: Literal["auto", "on", "off"] = "auto"   # auto: on when width > 2500 px (4K panoramic)
    tile_size: int = 1280
    tile_overlap: float = 0.2
    nms_iou: float = 0.5


class CalibConfig(BaseModel):
    camera: Literal["auto", "static", "moving"] = "auto"
    keyframe_every: int = 5         # frames (moving camera)
    static_samples: int = 30        # frames sampled to solve a static camera once
    kp_conf: float = 0.5
    ransac_px: float = 8.0
    min_points: int = 4
    line_score_ok: float = 0.7      # below this a segment goes to review
    smooth_window: int = 9          # keyframes (median smoothing of H)


class TrackConfig(BaseModel):
    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6   # new IDs only above ~0.6 (lesson from the reviewed system)
    gate_m: float = 3.0             # metric gate (plus speed slack)
    w_iou: float = 0.3
    w_dist: float = 0.5
    w_app: float = 0.2
    match_thresh: float = 0.8
    second_match_thresh: float = 0.6
    max_lost_s: float = 1.5
    ball_inflate: float = 3.0       # ball box inflated for matching only
    use_appearance: bool = True
    appearance_every: int = 5       # compute ReID embeddings every N frames per track


class LinkConfig(BaseModel):
    max_gap_s: float = 8.0
    max_speed: float = 10.0         # m/s feasible across gaps
    slack_m: float = 1.5
    w_dist: float = 0.5
    w_app: float = 0.35
    w_gap: float = 0.15
    link_thresh: float = 0.75
    min_len_s: float = 0.4          # shorter tracklets are dropped (noise)


class TeamConfig(BaseModel):
    embedder: Literal["siglip", "color"] = "siglip"
    crops_per_track: int = 12
    umap_dims: int = 3
    referee_outlier_z: float = 2.5


class IdentityConfig(BaseModel):
    crops_per_track: int = 24
    min_vote_share: float = 0.5
    min_votes: float = 1.5
    reid_link_sim: float = 0.7
    review_conf: float = 0.6


class BallConfig(BaseModel):
    max_candidates: int = 10
    max_speed: float = 40.0
    max_skip_s: float = 0.5
    jump_penalty: float = 4.0
    interp_max_gap_s: float = 2.0
    in_air_speed: float = 30.0


class OutputConfig(BaseModel):
    fps: float = 10.0               # tracking table rate


class ModelPaths(BaseModel):
    player_detector: Path = Path("models/football-player-detection.pt")
    ball_detector: Path = Path("models/football-ball-detection.pt")
    pitch_keypoints: Path = Path("models/football-pitch-detection.pt")
    siglip: str = "google/siglip-base-patch16-224"
    osnet: Path | None = Path("models/osnet_x1_0_msmt17.pth")
    parseq: str = "baudm/parseq"
    device: str = "cuda"


class PipelineConfig(BaseModel):
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    calib: CalibConfig = Field(default_factory=CalibConfig)
    track: TrackConfig = Field(default_factory=TrackConfig)
    link: LinkConfig = Field(default_factory=LinkConfig)
    team: TeamConfig = Field(default_factory=TeamConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    ball: BallConfig = Field(default_factory=BallConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    models: ModelPaths = Field(default_factory=ModelPaths)

    def section(self, name: str) -> dict:
        sec = getattr(self, name, None)
        return sec.model_dump(mode="json") if sec is not None else {}


class MatchInput(BaseModel):
    """What a coach provides: the video, the squad list, and a little context."""

    match_id: str
    video: str                          # local path, s3://, or synthetic://
    squad_csv: str | None = None        # columns: team,number,name[,position,player_id]
    home_name: str = "Home"
    away_name: str = "Away"
    home_color: str | None = None       # "#rrggbb" kit colour (helps naming the team clusters)
    away_color: str | None = None
    pitch_length: float = 105.0
    pitch_width: float = 68.0
    video_tier: Literal["broadcast", "panoramic", "follow_cam", "static"] = "broadcast"
    segment: tuple[float, float] | None = None   # seconds of the video to process (demo: 10 minutes)
    #: video seconds where periods start/end; default: one period = the whole segment
    periods: list[tuple[float, float]] | None = None
    #: +1 if home attacks +x in period 1; None -> infer from positions at kick-off
    home_attack_sign_p1: int | None = None
    competition: str | None = None
    date: str | None = None
