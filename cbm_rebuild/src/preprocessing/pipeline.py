from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from src.config import project_path
from src.data.common import TrialRecord
from src.data.seed_loader import iter_seed_trials
from src.data.seediv_loader import iter_seediv_trials

from .de_extraction import extract_de_features
from .padding import pad_trials
from .signal_processing import unified_preprocess


def _logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"preprocess.{path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _records(config: dict, dataset_name: str) -> Iterable[TrialRecord]:
    data = config["data"]
    channels = int(data.get("channels", 62))
    if dataset_name == "SEED":
        return iter_seed_trials(
            project_path(data["time_domain_dir"]), project_path(data["label_path"]), channels
        )
    if dataset_name == "SEED-IV":
        return iter_seediv_trials(project_path(data["time_domain_dir"]), channels)
    raise ValueError(f"Unsupported dataset {dataset_name!r}")


def run_preprocessing(config: dict, dataset_name: str) -> Path:
    preprocessing = config["preprocessing"]
    output_path = project_path(config["data"]["processed_path"])
    log_path = project_path(config["output"]["log_dir"]) / f"preprocess_{dataset_name.lower().replace('-', '')}.log"
    logger = _logger(log_path)
    logger.info("Starting %s preprocessing", dataset_name)

    fs_in = float(preprocessing["original_sampling_rate"])
    fs_out = float(preprocessing["target_sampling_rate"])
    broad_band = tuple(float(x) for x in preprocessing["broad_band_hz"])
    order = int(preprocessing["filter_order"])
    window_seconds = float(preprocessing["window_seconds"])
    hop_seconds = float(preprocessing["hop_seconds"])
    bands = preprocessing["bands_hz"]

    features: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    sessions: list[int] = []
    trials: list[int] = []
    sources: list[dict[str, object]] = []
    for index, record in enumerate(_records(config, dataset_name), start=1):
        filtered = unified_preprocess(record.signal, fs_in, fs_out, broad_band, order)
        de = extract_de_features(
            filtered, fs_out, window_seconds, hop_seconds, bands, order
        )
        features.append(de)
        labels.append(record.label)
        subjects.append(record.subject)
        sessions.append(record.session)
        trials.append(record.trial)
        sources.append(
            {
                "index": index - 1,
                "subject": record.subject,
                "session": record.session,
                "trial": record.trial,
                "label": record.label,
                "source_file": record.source_file,
                "source_key": record.source_key,
                "original_samples": int(record.signal.shape[1]),
                "resampled_samples": int(filtered.shape[1]),
                "num_windows": int(de.shape[0]),
            }
        )
        logger.info(
            "%s %d | subject=%02d session=%d trial=%02d windows=%d key=%s",
            dataset_name,
            index,
            record.subject,
            record.session,
            record.trial,
            de.shape[0],
            record.source_key,
        )

    data, mask, lengths = pad_trials(features)
    if data.shape[-1] != int(config["data"]["channels"]) * len(bands):
        raise RuntimeError(f"Unexpected feature dimension: {data.shape}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=data,
        mask=mask,
        y=np.asarray(labels, dtype=np.int64),
        subject=np.asarray(subjects, dtype=np.int16),
        session=np.asarray(sessions, dtype=np.int8),
        trial=np.asarray(trials, dtype=np.int8),
        lengths=lengths,
    )

    metadata = {
        "dataset": dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "processed_file": str(output_path),
        "shape": list(data.shape),
        "mask_shape": list(mask.shape),
        "feature_layout": "channel-major flattening of [62, 5]",
        "band_order": list(bands),
        "label_mapping": (
            {"negative": 0, "neutral": 1, "positive": 2}
            if dataset_name == "SEED"
            else {"neutral": 0, "sad": 1, "fear": 2, "happy": 3}
        ),
        "trials_per_subject": dict(sorted(Counter(subjects).items())),
        "max_sequence_length": int(data.shape[1]),
        "window_seconds": window_seconds,
        "hop_seconds": hop_seconds,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "trials": sources,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %s with shape=%s", output_path, data.shape)
    logger.info("Saved metadata %s", metadata_path)
    _close_logger(logger)
    return output_path
