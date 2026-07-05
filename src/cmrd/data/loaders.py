from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from cmrd.config import ExperimentConfig

from .records import TrialRecord

SEEDIV_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
_TRIAL_PATTERN = re.compile(r"(?:^|_)eeg_?(\d+)$", re.IGNORECASE)


def _visible_mat_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")
    return sorted(path for path in directory.glob("*.mat") if not path.name.startswith("._"))


def _channels_first(value: object, channels: int, context: str) -> np.ndarray:
    signal = np.asarray(value)
    if signal.ndim != 2:
        raise ValueError(f"{context}: expected 2-D signal, got {signal.shape}")
    if signal.shape[0] == channels:
        result = signal
    elif signal.shape[1] == channels:
        result = signal.T
    else:
        raise ValueError(f"{context}: neither dimension contains {channels} channels: {signal.shape}")
    result = np.asarray(result, dtype=np.float32, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{context}: signal contains NaN or infinite values")
    return result


def _trial_keys(mat: dict[str, object], count: int, context: str) -> dict[int, str]:
    keys: dict[int, str] = {}
    for key in mat:
        match = _TRIAL_PATTERN.search(key)
        if match:
            number = int(match.group(1))
            if 1 <= number <= count:
                if number in keys:
                    raise ValueError(f"{context}: duplicate trial {number}: {keys[number]}, {key}")
                keys[number] = key
    missing = sorted(set(range(1, count + 1)) - set(keys))
    if missing:
        raise ValueError(f"{context}: missing EEG trials {missing}")
    return keys


def _seed_labels(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"SEED label file does not exist: {path}")
    mat = loadmat(path)
    for key in ("label", "labels", "Label", "Labels"):
        if key in mat:
            raw = np.asarray(mat[key]).reshape(-1)
            break
    else:
        raise KeyError(f"No label array found in {path}")
    if raw.size != 15:
        raise ValueError(f"SEED requires 15 labels, found {raw.size}")
    mapping = {-1: 0, 0: 1, 1: 2}
    try:
        return np.asarray([mapping[int(value)] for value in raw], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unexpected SEED label {exc.args[0]}") from exc


def _seed_sessions(raw_dir: Path) -> list[tuple[int, int, Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in _visible_mat_files(raw_dir):
        if path.name.lower() == "label.mat":
            continue
        try:
            subject = int(path.stem.split("_", 1)[0])
        except ValueError as exc:
            raise ValueError(f"SEED filename must start with a subject number: {path.name}") from exc
        grouped[subject].append(path)
    if sorted(grouped) != list(range(1, 16)):
        raise ValueError(f"SEED requires subjects 1..15, found {sorted(grouped)}")
    sessions: list[tuple[int, int, Path]] = []
    for subject, paths in sorted(grouped.items()):
        ordered = sorted(paths, key=lambda path: path.stem.split("_", 1)[1])
        if len(ordered) != 3:
            raise ValueError(f"SEED subject {subject} requires 3 sessions, found {len(ordered)}")
        sessions.extend((subject, session, path) for session, path in enumerate(ordered, 1))
    return sessions


def _seediv_sessions(raw_dir: Path) -> list[tuple[int, int, Path]]:
    sessions: list[tuple[int, int, Path]] = []
    for session in range(1, 4):
        for path in _visible_mat_files(raw_dir / str(session)):
            try:
                subject = int(path.stem.split("_", 1)[0])
            except ValueError as exc:
                raise ValueError(f"SEED-IV filename must start with a subject number: {path.name}") from exc
            sessions.append((subject, session, path))
    sessions.sort(key=lambda item: (item[0], item[1]))
    expected = {(subject, session) for subject in range(1, 16) for session in range(1, 4)}
    found = {(subject, session) for subject, session, _ in sessions}
    if found != expected or len(sessions) != 45:
        raise ValueError(f"SEED-IV requires 45 subject/session files; missing={sorted(expected - found)}")
    return sessions


def dataset_paths(config: ExperimentConfig) -> tuple[Path, Path | None]:
    raw_dir = config.data_root / str(config.raw["dataset"]["raw_dir"])
    label_value = config.raw["dataset"].get("label_file")
    label_path = config.data_root / str(label_value) if label_value else None
    return raw_dir.resolve(), label_path.resolve() if label_path else None


def validate_dataset(config: ExperimentConfig) -> dict[str, object]:
    raw_dir, label_path = dataset_paths(config)
    if config.dataset == "seed":
        sessions = _seed_sessions(raw_dir)
        assert label_path is not None
        labels = _seed_labels(label_path)
        trials_per_session = 15
        mapping = {"negative": 0, "neutral": 1, "positive": 2}
    else:
        sessions = _seediv_sessions(raw_dir)
        labels = None
        trials_per_session = 24
        mapping = {"neutral": 0, "sad": 1, "fear": 2, "happy": 3}
    return {
        "dataset": config.dataset,
        "raw_dir": str(raw_dir),
        "label_file": str(label_path) if label_path else None,
        "session_files": len(sessions),
        "expected_trials": len(sessions) * trials_per_session,
        "labels": labels.tolist() if labels is not None else SEEDIV_LABELS,
        "label_mapping": mapping,
        "files": [
            {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for _, _, path in sessions
        ],
    }


def iter_trials(config: ExperimentConfig) -> Iterator[TrialRecord]:
    raw_dir, label_path = dataset_paths(config)
    channels = int(config.raw["dataset"]["channels"])
    if config.dataset == "seed":
        assert label_path is not None
        labels = _seed_labels(label_path)
        for subject, session, path in _seed_sessions(raw_dir):
            mat = loadmat(path)
            keys = _trial_keys(mat, 15, f"SEED {path.name}")
            for trial in range(1, 16):
                key = keys[trial]
                yield TrialRecord(
                    _channels_first(mat[key], channels, f"SEED {path.name}:{key}"),
                    int(labels[trial - 1]), subject, session, trial, str(path), key,
                )
    else:
        for subject, session, path in _seediv_sessions(raw_dir):
            mat = loadmat(path)
            keys = _trial_keys(mat, 24, f"SEED-IV {path.name}")
            for trial in range(1, 25):
                key = keys[trial]
                yield TrialRecord(
                    _channels_first(mat[key], channels, f"SEED-IV {path.name}:{key}"),
                    int(SEEDIV_LABELS[session][trial - 1]), subject, session, trial, str(path), key,
                )

