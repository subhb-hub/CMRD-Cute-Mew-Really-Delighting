from __future__ import annotations

import csv
import hashlib
import pickle
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


SUBJECTS = 123
VIDEOS = 28
RECORDED_CHANNELS = 32
EEG_CHANNELS = 30
SAMPLES = 7_500
RATE_HZ = 250
FOLDS = 10
FOLD_WIDTH = 12

EMOTION_NAMES = (
    "anger",
    "disgust",
    "fear",
    "sadness",
    "neutral",
    "amusement",
    "inspiration",
    "joy",
    "tenderness",
)
VIDEO_LABELS = np.asarray(
    [0] * 3
    + [1] * 3
    + [2] * 3
    + [3] * 3
    + [4] * 4
    + [5] * 3
    + [6] * 3
    + [7] * 3
    + [8] * 3,
    dtype=np.int64,
)

# Processed_data follows cohort 2 after the official cohort-1 reordering.
CHANNEL_NAMES = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1",
    "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7",
    "P8", "PO3", "PO4", "Oz", "O1", "O2", "HEOR", "HEOL",
)
EEG_CHANNEL_NAMES = CHANNEL_NAMES[:EEG_CHANNELS]

CRITICAL_METADATA = (
    "Readme.md",
    "Recording_info.csv",
    "Dataset_description.md",
    "DataStructureOfBehaviouralData.xlsx",
    "Task_event.xlsx",
    "Electrode_Location.xlsx",
    "Stimuli_info.xlsx",
    "Supplementary Information .docx",
    "A Large Finer-grained Affective Computing EEG Dataset.pdf",
)


def subject_name(subject: int) -> str:
    if not 0 <= int(subject) < SUBJECTS:
        raise ValueError(f"FACED subject must be in [0,{SUBJECTS - 1}], got {subject}")
    return f"sub{int(subject):03d}"


def subject_path(processed_dir: Path, subject: int) -> Path:
    return processed_dir / f"{subject_name(subject)}.pkl"


def load_processed_subject(processed_dir: Path, subject: int) -> np.ndarray:
    path = subject_path(processed_dir, subject)
    if not path.is_file():
        raise FileNotFoundError(path)
    # FACED's official Processed_data files are Python pickles. Only load the
    # locally supplied official dataset; arbitrary pickle files are unsafe.
    with path.open("rb") as stream:
        value = pickle.load(stream)
    array = np.asarray(value)
    expected = (VIDEOS, RECORDED_CHANNELS, SAMPLES)
    if array.shape != expected or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{path.name} must be a floating array {expected}, got {array.shape}/{array.dtype}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{path.name} contains NaN/Inf")
    return array


def official_fold_subjects(fold: int) -> tuple[list[int], list[int]]:
    """Return source/test subjects for the official contiguous 10-fold split."""
    if not 1 <= int(fold) <= FOLDS:
        raise ValueError(f"FACED fold must be in [1,{FOLDS}], got {fold}")
    start = (int(fold) - 1) * FOLD_WIDTH
    stop = int(fold) * FOLD_WIDTH if int(fold) < FOLDS else SUBJECTS
    target = list(range(start, stop))
    target_set = set(target)
    source = [subject for subject in range(SUBJECTS) if subject not in target_set]
    return source, target


def trial_entries(subjects: Iterable[int]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for subject in subjects:
        subject = int(subject)
        for video_index, label in enumerate(VIDEO_LABELS, 1):
            entries.append({
                "trial_id": f"sub-{subject:03d}_video-{video_index:02d}",
                "subject": subject,
                "session": 1,
                "trial": video_index,
                "video": video_index,
                "label": int(label),
                "emotion": EMOTION_NAMES[int(label)],
                "source_index": subject * VIDEOS + video_index - 1,
            })
    return entries


def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_md5(metadata_dir: Path) -> dict[str, str]:
    path = metadata_dir / "manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            str(row["name"]): str(row["dataFileMD5Hex"]).lower()
            for row in rows
            if row.get("name") and row.get("dataFileMD5Hex")
        }


def _recording_subjects(metadata_dir: Path) -> tuple[list[str], dict[str, int]]:
    path = metadata_dir / "Recording_info.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    subjects = [str(row["sub"]).strip() for row in rows]
    counts = {
        "cohort_1": sum(str(row.get("Cohort ", "")).strip() == "1" for row in rows),
        "cohort_2": sum(str(row.get("Cohort ", "")).strip() == "2" for row in rows),
        "recorded_250_hz": sum(str(row.get("Sample_rate", "")).strip() == "250" for row in rows),
        "recorded_1000_hz": sum(str(row.get("Sample_rate", "")).strip() == "1000" for row in rows),
    }
    return subjects, counts


def validate_faced_data(
    processed_dir: Path,
    metadata_dir: Path,
    *,
    deep: bool = False,
    sample_subjects: Sequence[int] = (0, 36, 60, 61, 122),
) -> dict[str, Any]:
    expected_names = [f"sub{subject:03d}.pkl" for subject in range(SUBJECTS)]
    actual_names = sorted(path.name for path in processed_dir.glob("sub*.pkl"))
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(f"FACED Processed_data subject mismatch: missing={missing}, extra={extra}")

    missing_metadata = [name for name in CRITICAL_METADATA if not (metadata_dir / name).is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"Missing FACED metadata: {missing_metadata}")

    recording_subjects, recording_counts = _recording_subjects(metadata_dir)
    expected_subjects = [subject_name(subject) for subject in range(SUBJECTS)]
    if recording_subjects != expected_subjects:
        raise ValueError("Recording_info.csv subject order does not match sub000..sub122")
    if recording_counts["cohort_1"] != 61 or recording_counts["cohort_2"] != 62:
        raise ValueError(f"Unexpected FACED cohort counts: {recording_counts}")

    manifest_hashes = _manifest_md5(metadata_dir)
    checked_hashes: dict[str, str] = {}
    for name in CRITICAL_METADATA:
        expected = manifest_hashes.get(name)
        if expected is None:
            continue
        actual = _md5(metadata_dir / name)
        if actual != expected:
            raise ValueError(f"Metadata checksum mismatch for {name}: {actual} != {expected}")
        checked_hashes[name] = actual

    checked_subjects = list(range(SUBJECTS)) if deep else sorted(set(map(int, sample_subjects)))
    dtypes: set[str] = set()
    for subject in checked_subjects:
        dtypes.add(str(load_processed_subject(processed_dir, subject).dtype))

    fold_sizes = [len(official_fold_subjects(fold)[1]) for fold in range(1, FOLDS + 1)]
    return {
        "dataset": "FACED",
        "processed_dir": str(processed_dir.resolve()),
        "metadata_dir": str(metadata_dir.resolve()),
        "subjects": SUBJECTS,
        "videos_per_subject": VIDEOS,
        "recorded_channels": RECORDED_CHANNELS,
        "eeg_channels_used": EEG_CHANNELS,
        "samples_per_trial": SAMPLES,
        "sampling_rate_hz": RATE_HZ,
        "trial_seconds": SAMPLES / RATE_HZ,
        "classes": len(EMOTION_NAMES),
        "emotion_names": list(EMOTION_NAMES),
        "video_labels": VIDEO_LABELS.tolist(),
        "channel_names": list(CHANNEL_NAMES),
        "eeg_channel_names_used": list(EEG_CHANNEL_NAMES),
        "excluded_mastoid_channels": list(CHANNEL_NAMES[EEG_CHANNELS:]),
        "official_fold_sizes": fold_sizes,
        "recording_counts": recording_counts,
        "metadata_md5": checked_hashes,
        "deep": bool(deep),
        "checked_subjects": checked_subjects,
        "observed_dtypes": sorted(dtypes),
    }
