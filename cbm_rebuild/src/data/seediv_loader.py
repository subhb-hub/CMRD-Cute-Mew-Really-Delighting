from __future__ import annotations

from pathlib import Path
from typing import Iterator

from scipy.io import loadmat

from .common import TrialRecord, channels_first, trial_keys, visible_mat_files


LABELS_BY_SESSION = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


def discover_sessions(raw_dir: str | Path) -> list[tuple[int, int, Path]]:
    root = Path(raw_dir)
    sessions: list[tuple[int, int, Path]] = []
    for session in range(1, 4):
        for path in visible_mat_files(root / str(session)):
            try:
                subject = int(path.stem.split("_", 1)[0])
            except ValueError as exc:
                raise ValueError(f"SEED-IV filename must start with numeric subject id: {path.name}") from exc
            sessions.append((subject, session, path))
    sessions.sort(key=lambda item: (item[0], item[1]))
    found = {(subject, session) for subject, session, _ in sessions}
    expected = {(subject, session) for subject in range(1, 16) for session in range(1, 4)}
    if found != expected or len(sessions) != 45:
        missing = sorted(expected.difference(found))
        raise ValueError(f"SEED-IV expected one file per subject/session; missing={missing}, files={len(sessions)}")
    return sessions


def iter_seediv_trials(raw_dir: str | Path, channels: int = 62) -> Iterator[TrialRecord]:
    for subject, session, path in discover_sessions(raw_dir):
        mat = loadmat(path)
        keys = trial_keys(mat, expected_count=24, dataset="SEED-IV")
        labels = LABELS_BY_SESSION[session]
        for trial in range(1, 25):
            key = keys[trial]
            signal = channels_first(mat[key], channels, f"SEED-IV {path.name}:{key}")
            yield TrialRecord(
                signal=signal,
                label=labels[trial - 1],
                subject=subject,
                session=session,
                trial=trial,
                source_file=str(path.resolve()),
                source_key=key,
            )

