"""Human corrections from the review UI. Stored per match in ``corrections.json``.

Kinds and the stage that consumes them (so a fix re-runs only that stage and what follows it):

    calibration_points -> calib     [{"frame": int, "image": [[u, v] x4+], "pitch": [[x, y] x4+]}]
    split              -> link      [{"track_id": str, "frame": int}]
    merge              -> link      [{"track_ids": [str, str, ...]}]
    team               -> team      {"<track_id>": "home" | "away" | "referee"}
    assign             -> identity  {"<track_id>": "<player_id>"}
    event              -> events    [{"op": "add" | "delete" | "edit", "event_id"?: str, "event"?: {...}}]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

KIND_TO_STAGE = {
    "calibration_points": "calib",
    "split": "link",
    "merge": "link",
    "team": "team",
    "assign": "identity",
    "event": "events",
}
LIST_KINDS = {"calibration_points", "split", "merge", "event"}


def load(workspace: Path) -> dict[str, Any]:
    p = Path(workspace) / "corrections.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save(workspace: Path, corrections: dict[str, Any]) -> None:
    p = Path(workspace) / "corrections.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(corrections, indent=2, sort_keys=True))


def add(workspace: Path, kind: str, value: Any, author: str | None = None) -> dict[str, Any]:
    """Append (list kinds) or update (mapping kinds) a correction; returns the new corrections dict."""
    if kind not in KIND_TO_STAGE:
        raise ValueError(f"unknown correction kind {kind!r}; expected one of {sorted(KIND_TO_STAGE)}")
    cur = load(workspace)
    if kind in LIST_KINDS:
        items = value if isinstance(value, list) else [value]
        cur.setdefault(kind, []).extend(items)
    else:
        if not isinstance(value, dict):
            raise ValueError(f"correction {kind!r} must be a mapping track_id -> value")
        cur.setdefault(kind, {}).update({str(k): v for k, v in value.items()})
    log = cur.setdefault("_log", [])
    log.append({"kind": kind, "author": author, "n": len(value) if hasattr(value, "__len__") else 1})
    save(workspace, cur)
    return cur


def earliest_affected_stage(kinds: list[str]) -> list[str]:
    return sorted({KIND_TO_STAGE[k] for k in kinds if k in KIND_TO_STAGE})
