from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.io import loadmat

from .common import TrialRecord, channels_first, trial_keys, visible_mat_files


def _labels(label_path: Path) -> np.ndarray:
    if not label_path.is_file():
        raise FileNotFoundError(f"SEED label file does not exist: {label_path}")
    mat = loadmat(label_path)
    for key in ("label", "labels", "Label", "Labels"):
        if key in mat:
            raw = np.asarray(mat[key]).reshape(-1)
            break
    else:
        raise KeyError(f"No label variable found in {label_path}; keys={sorted(mat)}")
    if raw.size != 15:
        raise ValueError(f"SEED requires 15 labels, found {raw.size} in {label_path}")
    mapping = {-1: 0, 0: 1, 1: 2}
    try:
        return np.asarray([mapping[int(value)] for value in raw], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unexpected SEED label {exc.args[0]}; expected -1, 0, or 1") from exc


def discover_sessions(data_dir: str | Path) -> list[tuple[int, int, Path]]:
    root = Path(data_dir)
    files = [p for p in visible_mat_files(root) if p.name.lower() != "label.mat"]
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        try:
            subject = int(path.stem.split("_", 1)[0])
        except ValueError as exc:
            raise ValueError(f"SEED filename must start with numeric subject id: {path.name}") from exc
        grouped[subject].append(path)
    if sorted(grouped) != list(range(1, 16)):
        raise ValueError(f"SEED expected subjects 1..15, found {sorted(grouped)}")
    sessions: list[tuple[int, int, Path]] = []
    for subject, paths in sorted(grouped.items()):
        ordered = sorted(paths, key=lambda p: p.stem.split("_", 1)[1])
        if len(ordered) != 3:
            raise ValueError(f"SEED subject {subject} expected 3 sessions, found {len(ordered)}")
        sessions.extend((subject, index + 1, path) for index, path in enumerate(ordered))
    return sessions


def iter_seed_trials(
    data_dir: str | Path, label_path: str | Path, channels: int = 62
) -> Iterator[TrialRecord]:
    labels = _labels(Path(label_path))
    for subject, session, path in discover_sessions(data_dir):
        mat = loadmat(path)
        keys = trial_keys(mat, expected_count=15, dataset="SEED")
        for trial in range(1, 16):
            key = keys[trial]
            signal = channels_first(mat[key], channels, f"SEED {path.name}:{key}")
            yield TrialRecord(
                signal=signal,
                label=int(labels[trial - 1]),
                subject=subject,
                session=session,
                trial=trial,
                source_file=str(path.resolve()),
                source_key=key,
            )

