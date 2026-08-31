from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


LOGGER = logging.getLogger(__name__)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:length]


def slugify(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        else:
            allowed.append('_')
    slug = ''.join(allowed)
    while '__' in slug:
        slug = slug.replace('__', '_')
    return slug.strip('_') or 'item'


def stable_identifier(prefix: str, text: str, length: int = 10) -> str:
    return f'{slugify(prefix)}_{stable_hash(text, length=length)}'


def seed_from_text(*parts: Any) -> int:
    joined = '::'.join(str(part) for part in parts)
    return int(hashlib.sha1(joined.encode('utf-8')).hexdigest()[:8], 16)


def numpy_random_state(*parts: Any) -> np.random.RandomState:
    return np.random.RandomState(seed_from_text(*parts))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value)!r} is not JSON serializable')


def dump_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return default


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default) + os.linesep)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, default=json_default) + os.linesep)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def relative_path(path: str | Path, root: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def configure_logging(log_level: str = 'INFO') -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    )


