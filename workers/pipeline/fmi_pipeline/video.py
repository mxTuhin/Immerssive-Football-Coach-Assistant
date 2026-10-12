"""Video access: ffmpeg/ffprobe for ingest, OpenCV for frame reading, synthetic:// for tests."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


class TaggedFrame(np.ndarray):
    """ndarray carrying its frame index (and, for crops, the source box). Lets fake models look up truth."""

    index: int = -1
    box: tuple | None = None

    def __new__(cls, arr: np.ndarray, index: int, box: tuple | None = None):
        obj = np.asarray(arr).view(cls)
        obj.index = index
        obj.box = box
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.index = getattr(obj, "index", -1)
        self.box = getattr(obj, "box", None)


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    n_frames: int
    duration: float


class VideoSource(Protocol):
    info: VideoInfo

    def iter_frames(self, stride: int = 1, start: int = 0, end: int | None = None) -> Iterator[tuple[int, np.ndarray]]: ...
    def read(self, indices: list[int]) -> dict[int, np.ndarray]: ...


class CvVideo:
    def __init__(self, path: str | Path):
        import cv2  # type: ignore

        self._cv2 = cv2
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info = VideoInfo(fps=fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), n_frames=n, duration=n / fps)
        cap.release()

    def iter_frames(self, stride: int = 1, start: int = 0, end: int | None = None):
        cap = self._cv2.VideoCapture(self.path)
        if start:
            cap.set(self._cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        end = self.info.n_frames if end is None else end
        while i < end:
            if (i - start) % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, TaggedFrame(frame, i)
            else:
                if not cap.grab():
                    break
            i += 1
        cap.release()

    def read(self, indices: list[int]) -> dict[int, np.ndarray]:
        """Read specific frames (sorted sequential access; seeks only across large jumps)."""
        want = sorted(set(int(i) for i in indices))
        out: dict[int, np.ndarray] = {}
        if not want:
            return out
        cap = self._cv2.VideoCapture(self.path)
        pos = 0
        for idx in want:
            if idx - pos > 50 or idx < pos:
                cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
                pos = idx
            while pos < idx:
                cap.grab()
                pos += 1
            ok, frame = cap.read()
            pos += 1
            if ok:
                out[idx] = TaggedFrame(frame, idx)
        cap.release()
        return out


def open_video(path: str | Path) -> VideoSource:
    s = str(path)
    if s.startswith("synthetic://"):
        from fmi_pipeline.synthetic import SyntheticVideo

        return SyntheticVideo.from_uri(s)
    return CvVideo(s)


def crop(frame: np.ndarray, box, pad: float = 0.0, region: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Crop ``box`` (x1, y1, x2, y2); ``region`` selects a vertical band of the box (e.g. torso)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    h = y2 - y1
    y1, y2 = y1 + region[0] * h, y1 + region[1] * h
    w = x2 - x1
    x1, x2 = x1 - pad * w, x2 + pad * w
    H, W = frame.shape[:2]
    xa, xb = int(max(0, np.floor(x1))), int(min(W, np.ceil(x2)))
    ya, yb = int(max(0, np.floor(y1))), int(min(H, np.ceil(y2)))
    c = frame[ya:max(yb, ya + 1), xa:max(xb, xa + 1)]
    return TaggedFrame(c, getattr(frame, "index", -1), tuple(box))


# ----------------------------------------------------------------------------- ffmpeg helpers

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(st.get("duration") or 0.0)
    n = int(st.get("nb_frames") or round(duration * fps))
    return VideoInfo(fps=fps, width=int(st["width"]), height=int(st["height"]), n_frames=n, duration=duration)


def make_proxy(src: str | Path, dst: str | Path, fps: float, max_width: int,
               segment: tuple[float, float] | None = None) -> Path:
    """Constant-frame-rate H.264 proxy; scaled down only if wider than ``max_width``."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if segment:
        cmd += ["-ss", f"{segment[0]:.3f}", "-to", f"{segment[1]:.3f}"]
    cmd += ["-i", str(src), "-an", "-vf", f"fps={fps},scale='min({max_width},iw)':-2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)


def detect_scene_cuts(path: str | Path, threshold: float = 0.4) -> list[float]:
    """Shot-cut times (s) using ffmpeg's scene score."""
    proc = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vf",
                           f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-"],
                          capture_output=True, text=True)
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
