from __future__ import annotations

import csv
import io
import shutil
import tempfile
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import mne
import numpy as np
from scipy.signal import resample_poly

from cmrd.config import ExperimentConfig

from .records import TrialRecord


DEAP_EEG_CHANNELS = (
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3",
    "CP1", "CP5", "P7", "P3", "Pz", "PO3", "O1", "Oz",
    "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
)
DEAP_EOG_CHANNELS = ("EXG1", "EXG2", "EXG3", "EXG4")
DEAP_RATING_COLUMNS = ("Valence", "Arousal", "Dominance", "Liking")
DEAP_SUBJECTS = 32
DEAP_TRIALS_PER_SUBJECT = 40
DEAP_ORIGINAL_RATE = 512.0
DEAP_STIMULUS_SECONDS = 60.0
DEAP_VIDEO_START_CODE = 4
DEAP_VIDEO_END_CODE = 5


def _deap_paths(config: ExperimentConfig) -> tuple[Path, Path]:
    raw_archive = (config.data_root / str(config.raw["dataset"]["raw_dir"])).resolve()
    ratings_archive = (config.data_root / str(config.raw["dataset"]["label_file"])).resolve()
    return raw_archive, ratings_archive


def read_deap_ratings(path: Path) -> dict[tuple[int, int], dict[str, float | int | None]]:
    if not path.is_file():
        raise FileNotFoundError(f"DEAP metadata archive does not exist: {path}")
    with zipfile.ZipFile(path) as archive:
        try:
            stream = archive.open("participant_ratings.csv")
        except KeyError as exc:
            raise KeyError(f"participant_ratings.csv is missing from {path}") from exc
        with stream, io.TextIOWrapper(stream, encoding="utf-8-sig", newline="") as text:
            rows = list(csv.DictReader(text))

    ratings: dict[tuple[int, int], dict[str, float | int | None]] = {}
    for row in rows:
        subject = int(row["Participant_id"])
        trial = int(row["Trial"])
        key = (subject, trial)
        if key in ratings:
            raise ValueError(f"Duplicate DEAP rating row for subject/trial={key}")
        ratings[key] = {
            "participant_id": subject,
            "trial": trial,
            "experiment_id": int(row["Experiment_id"]),
            "start_time_ms": int(row["Start_time"]),
            "valence": float(row["Valence"]),
            "arousal": float(row["Arousal"]),
            "dominance": float(row["Dominance"]),
            "liking": float(row["Liking"]),
            "familiarity": int(row["Familiarity"]) if row["Familiarity"].strip() else None,
        }
    expected = {
        (subject, trial)
        for subject in range(1, DEAP_SUBJECTS + 1)
        for trial in range(1, DEAP_TRIALS_PER_SUBJECT + 1)
    }
    if set(ratings) != expected:
        missing = sorted(expected - set(ratings))
        extra = sorted(set(ratings) - expected)
        raise ValueError(f"DEAP ratings coverage mismatch; missing={missing[:5]} extra={extra[:5]}")
    return ratings


def deap_label(rating: dict[str, float | int | None], target: str) -> int:
    target = str(target).lower()
    high_valence = float(rating["valence"]) >= 5.0
    high_arousal = float(rating["arousal"]) >= 5.0
    if target == "valence":
        return int(high_valence)
    if target == "arousal":
        return int(high_arousal)
    if target == "quadrant":
        # 0=LVLA, 1=LVHA, 2=HVLA, 3=HVHA.
        return 2 * int(high_valence) + int(high_arousal)
    raise ValueError(f"Unknown DEAP label target: {target}")


def pair_deap_video_events(
    events: np.ndarray,
    sampling_rate: float,
) -> list[tuple[int, int, int]]:
    value = np.asarray(events, dtype=np.int64)
    starts = value[value[:, 2] == DEAP_VIDEO_START_CODE, 0]
    observed_stops = value[value[:, 2] == DEAP_VIDEO_END_CODE, 0]
    if starts.size != DEAP_TRIALS_PER_SUBJECT:
        raise ValueError(f"DEAP requires 40 video starts, found {starts.size}")
    samples = int(round(DEAP_STIMULUS_SECONDS * float(sampling_rate)))
    pairs: list[tuple[int, int, int]] = []
    for start in starts:
        deltas = observed_stops - int(start)
        candidates = observed_stops[
            (deltas >= int(round(55.0 * sampling_rate)))
            & (deltas <= int(round(65.0 * sampling_rate)))
        ]
        observed_stop = (
            int(candidates[np.argmin(np.abs(candidates - (int(start) + samples)))])
            if candidates.size
            else -1
        )
        pairs.append((int(start), int(start) + samples, observed_stop))
    return pairs


def normalize_deap_event_codes(events: np.ndarray) -> np.ndarray:
    """Normalize BioSemi status words to DEAP's low-byte trigger codes."""
    value = np.asarray(events, dtype=np.int64).copy()
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"Expected MNE events shaped [N,3], got {value.shape}")
    value[:, 1] &= 0xFF
    value[:, 2] &= 0xFF
    return value


def validate_deap_dataset(config: ExperimentConfig) -> dict[str, Any]:
    raw_archive, ratings_archive = _deap_paths(config)
    if not raw_archive.is_file():
        raise FileNotFoundError(f"DEAP BDF archive does not exist: {raw_archive}")
    ratings = read_deap_ratings(ratings_archive)
    with zipfile.ZipFile(raw_archive) as archive:
        members = {info.filename: info for info in archive.infolist() if not info.is_dir()}
    expected_members = {f"s{subject:02d}.bdf" for subject in range(1, DEAP_SUBJECTS + 1)}
    if set(members) != expected_members:
        raise ValueError(
            f"DEAP BDF archive members mismatch; missing={sorted(expected_members - set(members))} "
            f"extra={sorted(set(members) - expected_members)}"
        )
    target = str(config.raw["dataset"]["label_target"]).lower()
    labels = Counter(deap_label(row, target) for row in ratings.values())
    return {
        "dataset": "deap",
        "raw_archive": str(raw_archive),
        "ratings_archive": str(ratings_archive),
        "subjects": DEAP_SUBJECTS,
        "trials_per_subject": DEAP_TRIALS_PER_SUBJECT,
        "expected_trials": DEAP_SUBJECTS * DEAP_TRIALS_PER_SUBJECT,
        "original_sampling_rate": DEAP_ORIGINAL_RATE,
        "target_sampling_rate": float(config.raw["signal"]["target_rate"]),
        "stimulus_seconds": DEAP_STIMULUS_SECONDS,
        "baseline_policy": "exclude; cut exactly 60 s from status code 4",
        "eeg_channels": list(DEAP_EEG_CHANNELS),
        "channel_order_policy": "select every EEG channel by name into canonical BioSemi32 order",
        "eog_channels_used_for_ica": list(DEAP_EOG_CHANNELS),
        "label_target": target,
        "label_mapping": (
            {"LVLA": 0, "LVHA": 1, "HVLA": 2, "HVHA": 3}
            if target == "quadrant"
            else {"low": 0, "high": 1}
        ),
        "label_threshold": 5.0,
        "label_counts": {str(label): int(count) for label, count in sorted(labels.items())},
        "files": [
            {
                "member": name,
                "size": int(members[name].file_size),
                "compressed_size": int(members[name].compress_size),
            }
            for name in sorted(expected_members)
        ],
    }


def _resample_to_target(signal_volt: np.ndarray, original_rate: float, target_rate: float) -> np.ndarray:
    ratio = Fraction(float(target_rate) / float(original_rate)).limit_denominator(10_000)
    value = resample_poly(signal_volt, ratio.numerator, ratio.denominator, axis=-1)
    return np.ascontiguousarray(value * 1e6, dtype=np.float32)


def iter_deap_trials(config: ExperimentConfig) -> Iterator[TrialRecord]:
    raw_archive, ratings_archive = _deap_paths(config)
    ratings = read_deap_ratings(ratings_archive)
    target = str(config.raw["dataset"]["label_target"]).lower()
    target_rate = float(config.raw["signal"]["target_rate"])
    if not np.isclose(float(config.raw["signal"]["original_rate"]), DEAP_ORIGINAL_RATE):
        raise ValueError("DEAP original_rate must be 512 Hz")

    picks = [*DEAP_EEG_CHANNELS, *DEAP_EOG_CHANNELS]
    with zipfile.ZipFile(raw_archive) as archive, tempfile.TemporaryDirectory(prefix="cmrd_deap_") as temporary:
        temporary_root = Path(temporary)
        for subject in range(1, DEAP_SUBJECTS + 1):
            member = f"s{subject:02d}.bdf"
            extracted = temporary_root / member
            with archive.open(member) as source, extracted.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)

            raw = mne.io.read_raw_bdf(extracted, preload=False, verbose="ERROR")
            try:
                if not np.isclose(float(raw.info["sfreq"]), DEAP_ORIGINAL_RATE):
                    raise ValueError(f"{member} sampling rate is {raw.info['sfreq']}, expected 512 Hz")
                missing = [name for name in picks if name not in raw.ch_names]
                if missing:
                    raise ValueError(f"{member} is missing required channels: {missing}")
                source_eeg_order = tuple(raw.ch_names[: len(DEAP_EEG_CHANNELS)])
                pick_indices = [raw.ch_names.index(name) for name in picks]
                stim_channel = "Status" if "Status" in raw.ch_names else raw.ch_names[-1]
                events = mne.find_events(
                    raw,
                    stim_channel=stim_channel,
                    shortest_event=1,
                    initial_event=True,
                    uint_cast=True,
                    verbose="ERROR",
                )
                events = normalize_deap_event_codes(events)
                event_pairs = pair_deap_video_events(events, float(raw.info["sfreq"]))
                for trial, (start, stop, observed_stop) in enumerate(event_pairs, 1):
                    # Subjects 23-32 use a different on-disk EEG order. Explicit
                    # integer picks guarantee the canonical BioSemi32 order.
                    data_volt = raw.get_data(picks=pick_indices, start=start, stop=stop)
                    if data_volt.shape != (len(picks), int(DEAP_STIMULUS_SECONDS * DEAP_ORIGINAL_RATE)):
                        raise ValueError(f"{member} trial {trial} has unexpected shape {data_volt.shape}")
                    data_microvolt = _resample_to_target(data_volt, DEAP_ORIGINAL_RATE, target_rate)
                    rating = ratings[(subject, trial)]
                    metadata = {
                        **rating,
                        "label_target": target,
                        "label_threshold": 5.0,
                        "labels": {
                            "valence": deap_label(rating, "valence"),
                            "arousal": deap_label(rating, "arousal"),
                            "quadrant": deap_label(rating, "quadrant"),
                        },
                        "event_start_sample_512hz": start,
                        "requested_stop_sample_512hz": stop,
                        "observed_end_event_sample_512hz": observed_stop,
                        "source_stim_channel_name": stim_channel,
                        "source_eeg_channel_order": list(source_eeg_order),
                        "channel_reordered_to_biosemi32": source_eeg_order != DEAP_EEG_CHANNELS,
                    }
                    yield TrialRecord(
                        signal=data_microvolt[: len(DEAP_EEG_CHANNELS)],
                        label=deap_label(rating, target),
                        subject=subject,
                        session=1,
                        trial=trial,
                        source_file=f"{raw_archive}::{member}",
                        source_key=f"status-{DEAP_VIDEO_START_CODE}@{start}",
                        eog_signal=data_microvolt[len(DEAP_EEG_CHANNELS) :],
                        eog_names=DEAP_EOG_CHANNELS,
                        metadata=metadata,
                    )
            finally:
                raw.close()
                del raw
            extracted.unlink(missing_ok=True)
