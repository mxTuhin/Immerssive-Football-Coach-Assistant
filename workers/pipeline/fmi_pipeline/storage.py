"""Storage: a local working directory, optionally mirrored to Cloudflare R2 / any S3-compatible bucket.

Layout per match (identical locally and in the bucket under ``<prefix>/matches/<id>/``)::

    source/        uploaded video, proxy, squad.csv, match_input.json
    stages/<stage>/<hash>/   artifacts + manifest.json
    stages/<stage>/latest.json
    corrections.json
    package/       the final Match Package
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    root: Path

    def match_dir(self, match_id: str) -> Path: ...
    def persist(self, path: Path) -> None: ...
    def fetch(self, match_id: str, rel: str) -> Path: ...


class LocalStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def match_dir(self, match_id: str) -> Path:
        d = self.root / "matches" / match_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def persist(self, path: Path) -> None:  # already on disk
        return None

    def fetch(self, match_id: str, rel: str) -> Path:
        return self.match_dir(match_id) / rel


class S3Storage(LocalStorage):
    """Local cache + bucket mirror. Works with R2 (``endpoint_url=https://<acct>.r2.cloudflarestorage.com``)."""

    def __init__(self, root: str | Path, bucket: str, prefix: str = "fmi", endpoint_url: str | None = None,
                 region: str | None = None):
        super().__init__(root)
        import boto3  # type: ignore

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url, region_name=region or "auto")

    def _key(self, path: Path) -> str:
        rel = Path(path).resolve().relative_to(self.root.resolve())
        return f"{self.prefix}/{rel.as_posix()}"

    def persist(self, path: Path) -> None:
        path = Path(path)
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        for f in files:
            self.s3.upload_file(str(f), self.bucket, self._key(f))

    def fetch(self, match_id: str, rel: str) -> Path:
        local = self.match_dir(match_id) / rel
        if local.exists():
            return local
        prefix = f"{self.prefix}/matches/{match_id}/{rel}"
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                dest = self.root / obj["Key"][len(self.prefix) + 1:]
                dest.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket, obj["Key"], str(dest))
        return local

    def presigned_url(self, path: Path, expires: int = 3600) -> str:
        return self.s3.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": self._key(path)},
                                              ExpiresIn=expires)


def storage_from_env() -> LocalStorage:
    root = os.environ.get("FMI_STORAGE_ROOT", "storage")
    bucket = os.environ.get("FMI_S3_BUCKET")
    if bucket:
        return S3Storage(root, bucket=bucket, prefix=os.environ.get("FMI_S3_PREFIX", "fmi"),
                         endpoint_url=os.environ.get("FMI_S3_ENDPOINT"), region=os.environ.get("FMI_S3_REGION"))
    return LocalStorage(root)
