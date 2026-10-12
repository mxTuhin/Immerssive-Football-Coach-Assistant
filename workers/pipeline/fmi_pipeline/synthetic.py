"""Synthetic match for end-to-end tests without GPU, ffmpeg or real video.

``synthetic://seed=1&seconds=20&fps=25`` opens a static-camera "video" whose frames are blank arrays tagged
with their index; the fake models in ``fmi_pipeline.models.fakes`` read the ground truth from the scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import parse_qs

import numpy as np

from fmi_pipeline.geometry import apply_h, dlt
from fmi_pipeline.video import TaggedFrame, VideoInfo
from fmi_schema.pitch import PitchDims

FORMATION = [(-48, 0), (-35, -20), (-37, -7), (-37, 7), (-35, 20), (-18, -22), (-20, 0), (-18, 22),
             (-5, -18), (-3, 0), (-5, 18)]


@dataclass
class SyntheticScene:
    seed: int = 1
    seconds: float = 20.0
    fps: float = 25.0
    width: int = 1920
    height: int = 1080
    pitch: PitchDims = field(default_factory=PitchDims)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        L, W = self.pitch.length, self.pitch.width
        img = np.array([[260, 230], [1660, 230], [2250, 1060], [-330, 1060]], float)
        pit = np.array([[-L / 2, W / 2], [L / 2, W / 2], [L / 2, -W / 2], [-L / 2, -W / 2]], float)
        self.H_pitch2img = dlt(pit, img)
        self.H_img2pitch = np.linalg.inv(self.H_pitch2img)
        self.H_img2pitch /= self.H_img2pitch[2, 2]
        n = int(self.seconds * self.fps)
        t = np.arange(n) / self.fps
        ids, teams, numbers, roles, xy = [], [], [], [], []
        for team, sign in (("home", 1), ("away", -1)):
            for k, (bx, by) in enumerate(FORMATION):
                ph = rng.uniform(0, 2 * np.pi, 4)
                amp = rng.uniform(3, 9, 2) if k else np.array([1.0, 2.0])
                x = sign * bx + amp[0] * np.sin(2 * np.pi * t / rng.uniform(8, 20) + ph[0])
                y = sign * by + amp[1] * np.sin(2 * np.pi * t / rng.uniform(8, 20) + ph[1])
                ids.append(f"{team}_{k + 1}")
                teams.append(team)
                numbers.append(k + 1)
                roles.append("goalkeeper" if k == 0 else "player")
                xy.append(np.stack([np.clip(x, -L / 2 + 1, L / 2 - 1), np.clip(y, -W / 2 + 1, W / 2 - 1)], 1))
        ref = np.stack([10 * np.sin(2 * np.pi * t / 15), 8 * np.cos(2 * np.pi * t / 17)], 1)
        ids.append("ref")
        teams.append("referee")
        numbers.append(None)
        roles.append("referee")
        xy.append(ref)
        self.ids, self.teams, self.numbers, self.roles = ids, teams, numbers, roles
        self.xy = np.stack(xy, 1)  # (n, objects, 2)
        self.n_frames = n
        self.ball = self._ball_path(rng)

    def _ball_path(self, rng) -> np.ndarray:
        n = self.n_frames
        ball = np.zeros((n, 2))
        outfield = [i for i, r in enumerate(self.roles) if r != "referee"]
        owner = outfield[rng.integers(len(outfield))]
        i = 0
        while i < n:
            hold = int(rng.uniform(1.0, 2.5) * self.fps)
            for k in range(i, min(n, i + hold)):
                ball[k] = self.xy[k, owner] + [0.4, 0.0]
            i += hold
            if i >= n:
                break
            nxt = outfield[rng.integers(len(outfield))]
            dist = np.linalg.norm(self.xy[i, nxt] - ball[i - 1])
            flight = max(2, int(dist / 15.0 * self.fps))
            start = ball[i - 1].copy()
            for k in range(flight):
                if i + k >= n:
                    break
                a = (k + 1) / flight
                ball[i + k] = (1 - a) * start + a * self.xy[min(i + flight, n - 1), nxt]
            i += flight
            owner = nxt
        return ball

    # -------------------------------------------------------------- per-frame truth in image space
    def boxes(self, f: int) -> list[dict]:
        out = []
        foot = apply_h(self.H_pitch2img, self.xy[f])
        for k, (u, v) in enumerate(foot):
            h = 45 + 75 * (v - 230) / 830
            w = 0.42 * h
            out.append({"id": self.ids[k], "team": self.teams[k], "role": self.roles[k], "number": self.numbers[k],
                        "box": (u - w / 2, v - h, u + w / 2, v), "xy": tuple(self.xy[f, k])})
        bu, bv = apply_h(self.H_pitch2img, self.ball[f:f + 1])[0]
        out.append({"id": "ball", "team": "ball", "role": "ball", "number": None,
                    "box": (bu - 5, bv - 10, bu + 5, bv), "xy": tuple(self.ball[f])})
        return [o for o in out if -50 < o["box"][0] < self.width + 50 and -50 < o["box"][1] < self.height + 50]

    def nearest(self, f: int, box) -> dict | None:
        cx, cy = (box[0] + box[2]) / 2, box[3]
        best, bd = None, 1e9
        for o in self.boxes(f):
            ox, oy = (o["box"][0] + o["box"][2]) / 2, o["box"][3]
            d = np.hypot(ox - cx, oy - cy)
            if d < bd:
                best, bd = o, d
        return best if bd < 40 else None

    def squad_csv(self) -> str:
        lines = ["team,number,name,position,player_id"]
        for k, pid in enumerate(self.ids):
            if self.teams[k] in ("home", "away"):
                pos = "GK" if self.roles[k] == "goalkeeper" else "OUT"
                lines.append(f"{self.teams[k]},{self.numbers[k]},{pid.replace('_', ' ').title()},{pos},{pid}")
        return "\n".join(lines) + "\n"


@lru_cache(maxsize=8)
def get_scene(seed: int, seconds: float, fps: float) -> SyntheticScene:
    return SyntheticScene(seed=seed, seconds=seconds, fps=fps)


class SyntheticVideo:
    def __init__(self, scene: SyntheticScene):
        self.scene = scene
        self.info = VideoInfo(fps=scene.fps, width=scene.width, height=scene.height, n_frames=scene.n_frames,
                              duration=scene.n_frames / scene.fps)
        self._blank = np.zeros((scene.height, scene.width, 3), np.uint8)

    @classmethod
    def from_uri(cls, uri: str) -> "SyntheticVideo":
        q = {k: v[0] for k, v in parse_qs(uri.split("://", 1)[1].lstrip("/?")).items()}
        return cls(get_scene(int(q.get("seed", 1)), float(q.get("seconds", 20)), float(q.get("fps", 25))))

    def iter_frames(self, stride: int = 1, start: int = 0, end: int | None = None):
        end = self.info.n_frames if end is None else min(end, self.info.n_frames)
        for i in range(start, end, stride):
            yield i, TaggedFrame(self._blank, i)

    def read(self, indices):
        return {int(i): TaggedFrame(self._blank, int(i)) for i in indices if 0 <= int(i) < self.info.n_frames}
