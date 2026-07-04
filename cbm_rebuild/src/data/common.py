from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrialRecord:
    signal: np.ndarray
    label: int
    subject: int
    session: int
    trial: int
    source_file: str
    source_key: str


def visible_mat_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"MAT directory does not exist: {directory}")
    files = sorted(p for p in directory.glob("*.mat") if not p.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No visible .mat files found under: {directory}")
    return files


def trial_keys(mat: dict[str, object], expected_count: int, dataset: str) -> dict[int, str]:
    """Resolve canonical eeg_N/cz_eegN names and official subject-prefixed variants."""
    resolved: dict[int, str] = {}
    for key in mat:
        if key.startswith("__"):
            continue
        match = re.search(r"(?:^|_)eeg_?(\d+)$", key, flags=re.IGNORECASE)
        if match:
            trial_id = int(match.group(1))
            if 1 <= trial_id <= expected_count:
                if trial_id in resolved:
                    raise ValueError(
                        f"[{dataset}] duplicate variables for trial {trial_id}: "
                        f"{resolved[trial_id]!r} and {key!r}"
                    )
                resolved[trial_id] = key
    expected = set(range(1, expected_count + 1))
    missing = sorted(expected.difference(resolved))
    if missing:
        shown = sorted(k for k in mat if not k.startswith("__"))[:12]
        raise KeyError(
            f"[{dataset}] missing EEG trial variables {missing}. Expected names ending in "
            f"eeg1..eeg{expected_count} (optional underscore). Available sample: {shown}"
        )
    return resolved


def channels_first(signal: np.ndarray, channels: int, context: str) -> np.ndarray:
    data = np.asarray(signal)
    if data.ndim != 2:
        raise ValueError(f"{context}: expected a 2-D EEG array, got shape {data.shape}")
    if data.shape[0] == channels:
        result = data
    elif data.shape[1] == channels:
        result = data.T
    else:
        raise ValueError(
            f"{context}: neither axis has the expected {channels} channels; shape={data.shape}"
        )
    if result.shape[1] < 2:
        raise ValueError(f"{context}: fewer than two time points; shape={result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{context}: signal contains NaN or infinite values")
    return np.asarray(result, dtype=np.float32, order="C")

