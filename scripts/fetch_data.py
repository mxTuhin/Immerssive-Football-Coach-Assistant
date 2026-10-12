#!/usr/bin/env python3
"""Fetch every large asset that is flagged with a ``*.download`` file instead of being committed.

Each flag file sits where the asset belongs (``data/videos``, ``models``, ``data/soccertrack``) and says how to
get it (``method: gdown | http | huggingface | manual | auto``).

    python scripts/fetch_data.py --list                 # status of every flagged asset
    python scripts/fetch_data.py --models               # pretrained weights
    python scripts/fetch_data.py --videos               # V1 clip (+ optional extras with --optional)
    python scripts/fetch_data.py --soccertrack --list-remote
    python scripts/fetch_data.py --soccertrack --pattern "<glob of one half + labels>"
    python scripts/fetch_data.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ["data/videos", "data/soccertrack", "models"]


def parse_flag(path: Path) -> dict:
    info = {"flag": path}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        info[k.strip()] = v.strip()
    info["dest"] = (path.parent / info.get("target", "")).resolve()
    return info


def flags() -> list[dict]:
    out = []
    for d in SEARCH:
        for p in sorted((ROOT / d).glob("*.download")):
            out.append(parse_flag(p))
    return out


def present(f: dict) -> bool:
    if f.get("method") in ("auto",):
        return True
    t = f.get("target", "")
    return bool(t) and t != "." and f["dest"].exists() and f["dest"].stat().st_size > 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_http(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    tmp.rename(dest)


def fetch_gdown(url: str, dest: Path) -> None:
    try:
        import gdown  # type: ignore
    except ImportError:
        sys.exit("gdown missing: pip install gdown  (or pip install -r requirements/vision.txt)")
    out = gdown.download(url, str(dest), quiet=False)
    if not out:
        raise RuntimeError(f"gdown failed for {url} (Google Drive quota? retry later or download in a browser)")


def fetch_hf(f: dict, pattern: str | None, list_remote: bool) -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download  # type: ignore
    except ImportError:
        sys.exit("huggingface_hub missing: pip install huggingface_hub")
    repo, rtype = f["repo"], f.get("repo_type", "dataset")
    if list_remote:
        files = HfApi().list_repo_files(repo, repo_type=rtype)
        print(f"{len(files)} files in {repo}:")
        for x in files:
            print("  ", x)
        return
    pattern = pattern or f.get("pattern")
    if not pattern:
        sys.exit("pass --pattern (download ONE half-match + labels; run with --list-remote to see names)")
    snapshot_download(repo, repo_type=rtype, allow_patterns=[pattern], local_dir=str(f["flag"].parent))


def run(f: dict, args) -> None:
    m = f.get("method")
    name = f.get("name", f["flag"].name)
    if m == "auto":
        print(f"[auto]   {name}: fetched automatically on first use ({f.get('url')})")
        return
    if m == "manual":
        print(f"[manual] {name}: see instructions in {f['flag'].relative_to(ROOT)} -> {f.get('url') or ''}")
        return
    if m == "huggingface":
        fetch_hf(f, args.pattern, args.list_remote)
        return
    if present(f) and not args.force:
        print(f"[ok]     {name}: {f['dest'].relative_to(ROOT)}")
        return
    f["dest"].parent.mkdir(parents=True, exist_ok=True)
    print(f"[get]    {name} -> {f['dest'].relative_to(ROOT)}")
    if m == "gdown":
        fetch_gdown(f["url"], f["dest"])
    elif m == "http":
        fetch_http(f["url"], f["dest"])
    else:
        print(f"         unknown method {m!r}")
        return
    if f.get("sha256") and sha256(f["dest"]) != f["sha256"]:
        f["dest"].unlink()
        raise RuntimeError(f"sha256 mismatch for {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--soccertrack", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--optional", action="store_true", help="include optional assets")
    ap.add_argument("--only", help="target file name")
    ap.add_argument("--pattern", help="huggingface allow pattern")
    ap.add_argument("--list-remote", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    fl = flags()
    if args.list or not any([args.models, args.videos, args.soccertrack, args.all, args.only]):
        for f in fl:
            state = "present" if present(f) else ("auto" if f.get("method") == "auto" else "MISSING")
            req = "required" if f.get("required") == "true" else "optional"
            print(f"{state:8s} {req:8s} {f.get('method', '?'):12s} {f['flag'].relative_to(ROOT)}")
        return 0
    groups = {g for g, on in (("models", args.models), ("videos", args.videos), ("soccertrack", args.soccertrack)) if on}
    failed = 0
    for f in fl:
        if args.only and f.get("target") != args.only:
            continue
        if not args.only and not args.all and f.get("group") not in groups:
            continue
        if not args.only and f.get("required") != "true" and not args.optional:
            continue
        try:
            run(f, args)
        except Exception as exc:  # keep going, report at the end
            failed += 1
            print(f"[fail]   {f.get('name')}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
