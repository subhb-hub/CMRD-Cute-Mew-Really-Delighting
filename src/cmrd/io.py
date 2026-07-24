from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _replace_with_retry(temporary: str | Path, destination: Path, attempts: int = 8) -> None:
    """Atomically replace a file, tolerating short-lived Windows scan locks.

    Antivirus, indexers, and sync clients can briefly open a just-updated JSON
    file without delete sharing, causing ``os.replace`` to raise WinError 5.
    Retrying the same atomic operation preserves the previous destination until
    replacement succeeds and avoids treating a transient external lock as a
    failed experiment.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (2 ** attempt))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_retry(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
