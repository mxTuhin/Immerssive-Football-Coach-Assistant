"""Table IO. Parquet is the storage format; CSV(.gz) is accepted for reading fixtures."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


def have_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame as Parquet. Falls back to ``.csv.gz`` only when pyarrow is missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet" and have_parquet():
        df.to_parquet(path, index=False)
        return path
    if path.suffix == ".parquet":
        path = path.with_suffix(".csv.gz")
    df.to_csv(path, index=False)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() and path.suffix == ".parquet":
        alt = path.with_suffix(".csv.gz")
        if alt.exists():
            path = alt
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def table_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() or (path.suffix == ".parquet" and path.with_suffix(".csv.gz").exists())


def open_text(path: str | Path):
    """Open a text file, transparently decompressing ``.gz``."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve(path: str | Path) -> Path:
    """Return ``path`` or its ``.gz`` sibling, whichever exists."""
    p = Path(path)
    if p.exists():
        return p
    gz = p.with_name(p.name + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"{p} (or {gz.name}) not found")


class _Encoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="json")
        return super().default(o)


def dump_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), cls=_Encoder, indent=2))
    return path


def to_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(_clean(obj), cls=_Encoder))


def _clean(obj: Any) -> Any:
    """Replace NaN/inf floats with None so the JSON is valid everywhere."""
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _clean(obj.model_dump(mode="json"))
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    return obj


def load_json(path: str | Path) -> Any:
    with open_text(path) as fh:
        return json.load(fh)
