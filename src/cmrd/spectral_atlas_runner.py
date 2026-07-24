from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import time
import traceback
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cmrd.config import ExperimentConfig, load_config
from cmrd.faced import (
    EEG_CHANNELS,
    EMOTION_NAMES,
    RATE_HZ,
    SUBJECTS,
    VIDEO_LABELS,
    VIDEOS,
    load_processed_subject,
    official_fold_subjects,
    trial_entries,
    validate_faced_data,
)
from cmrd.faced_adversarial_runner import stimulus_split
from cmrd.features.de import extract_de
from cmrd.features.spectral_atlas import (
    LandmarkBandState,
    ProjectionState,
    apply_projection,
    capped_dimension,
    extract_native_shape_power,
    fit_channel_pca,
    fit_landmark_band,
    fit_random_projection,
    full_dimension,
    full_ilr_power,
    full_log_psd,
    landmark_power,
    scalar_jsd_power,
)
from cmrd.fixed_protocol_runner import validate_fixed_cache
from cmrd.io import read_json, write_json, write_npz
from cmrd.training.metrics import classification_metrics
from cmrd.training.runtime import seed_everything, select_device


LOGGER = logging.getLogger("cmrd.spectral_atlas")
SCHEMA_VERSION = 1
FAMILY = "Spectral-Atlas-v1"
SEED = 42
CAPS = (1, 2, 4, 8)
FACED_PROTOCOLS = ("conventional_subject_holdout", "subject_and_stimulus_holdout")
SEEDIV_PROTOCOLS = ("loso",)
MODELS = ("logistic_regression", "linear_svm", "pooled_mlp")
BASE_CONDITIONS = (
    "de",
    "log_band_power",
    "scalar_jsd_power",
    "ilr_power_full",
    "log_psd_full",
)


def all_conditions(caps: Sequence[int] = CAPS) -> tuple[str, ...]:
    output = list(BASE_CONDITIONS)
    for cap in caps:
        output.extend((
            f"raw_landmark_power_cap{cap}",
            f"nystrom_landmark_power_cap{cap}",
            f"pca_ilr_power_cap{cap}",
            f"random_ilr_power_cap{cap}",
        ))
    return tuple(output)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_balanced_accuracy": metrics["balanced_accuracy"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
    }


def experiment_settings(config: ExperimentConfig) -> dict[str, Any]:
    raw = dict(config.raw.get("spectral_atlas", {}))
    settings = {
        "seed": int(raw.get("seed", SEED)),
        "caps": tuple(int(value) for value in raw.get("caps", CAPS)),
        "conditions": tuple(str(value) for value in raw.get("conditions", all_conditions())),
        "models": tuple(str(value) for value in raw.get("models", MODELS)),
        "protocols": tuple(str(value) for value in raw.get(
            "protocols", FACED_PROTOCOLS if config.dataset == "faced" else SEEDIV_PROTOCOLS
        )),
        "folds": tuple(int(value) for value in raw.get("folds", [1])),
        "development_subjects": int(raw.get("development_subjects", 12)),
        "candidate_windows_per_trial": int(raw.get("candidate_windows_per_trial", 2)),
        "maximum_anchor_candidates": int(raw.get("maximum_anchor_candidates", 8192)),
        "landmark_ridge": float(raw.get("landmark_ridge", 1e-6)),
        "monitor_interval": int(raw.get("monitor_interval", 10)),
        "pooling": str(raw.get("pooling", "mean_std")),
    }
    if settings["seed"] != SEED:
        raise ValueError("Spectral-Atlas-v1 is frozen to seed=42")
    if settings["caps"] != CAPS:
        raise ValueError(f"Spectral-Atlas-v1 caps must be {CAPS}")
    if settings["conditions"] != all_conditions(settings["caps"]):
        raise ValueError("spectral_atlas.conditions must contain the complete ordered v1 matrix")
    if settings["models"] != MODELS:
        raise ValueError(f"spectral_atlas.models must be {MODELS}")
    expected_protocols = FACED_PROTOCOLS if config.dataset == "faced" else SEEDIV_PROTOCOLS
    if settings["protocols"] != expected_protocols:
        raise ValueError(f"{config.dataset} protocols must be {expected_protocols}")
    maximum_fold = 10 if config.dataset == "faced" else 15
    if not settings["folds"] or any(not 1 <= fold <= maximum_fold for fold in settings["folds"]):
        raise ValueError(f"Invalid {config.dataset} folds: {settings['folds']}")
    if settings["candidate_windows_per_trial"] < 1 or settings["maximum_anchor_candidates"] < 32:
        raise ValueError("Candidate sampling settings are too small")
    if settings["monitor_interval"] != 10 or settings["pooling"] != "mean_std":
        raise ValueError("v1 requires mean_std pooling and 10-epoch monitoring")
    if config.dataset == "faced" and settings["development_subjects"] != 12:
        raise ValueError("FACED v1 requires 12 complete source-development subjects")
    training = config.raw["training"]
    if int(training["epochs"]) < 10 or int(training["epochs"]) % 10:
        raise ValueError("training.epochs must be a positive multiple of 10")
    return settings


def validate_sources(config: ExperimentConfig) -> dict[str, Any]:
    settings = experiment_settings(config)
    if config.dataset == "faced":
        processed = config.data_root / str(config.raw["dataset"]["raw_dir"])
        metadata = config.data_root / str(config.raw["dataset"]["metadata_dir"])
        audit = validate_faced_data(processed, metadata, deep=False)
        audit.update({
            "preprocessing": "official FACED Processed_data only",
            "target_monitoring": "exploratory diagnostic every 10 epochs",
        })
        return audit
    if config.dataset != "seediv":
        raise ValueError("Spectral-Atlas-v1 currently supports FACED and SEED-IV only")
    audit = validate_fixed_cache(config)
    audit["target_monitoring"] = "exploratory diagnostic every 10 epochs"
    audit["configured_folds"] = list(settings["folds"])
    return audit


def _base_signature(config: ExperimentConfig) -> str:
    return _json_hash({
        "family": FAMILY,
        "dataset": config.dataset,
        "signal": config.raw["signal"],
        "channels": config.raw["dataset"]["channels"],
        "estimator": "Hann modified periodogram nfft=nperseg=one window; no zero padding",
        "power": "log sum of native-grid PSD inside each half-open band",
        "de": "fourth-order Butterworth band DE from the same cleaned trial",
    })


def _base_root(config: ExperimentConfig, run_root: Path) -> Path:
    return run_root / "cache" / "base" / config.dataset / _base_signature(config)


def _base_path(root: Path, trial_id: str) -> Path:
    return root / "trials" / f"{trial_id}.npz"


def _base_valid(path: Path, band_names: Sequence[str], channels: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            time_points = int(archive[band_names[0]].shape[0])
            # ``ndarray.all`` returns ``numpy.bool_``.  Without the explicit
            # conversion, summing validation results promotes ``completed``
            # to ``numpy.int64``, which ``json.dump`` cannot write to the base
            # cache manifest after a successful full-cache pass.
            return bool(
                all(archive[name].ndim == 3 and archive[name].shape[:2] == (time_points, channels) for name in band_names)
                and archive["log_power"].shape == (time_points, channels, len(band_names))
                and archive["de"].shape == (time_points, channels, len(band_names))
                and np.isfinite(archive["log_power"]).all()
                and np.isfinite(archive["de"]).all()
            )
    except (KeyError, OSError, ValueError):
        return False


def _save_base_trial(
    output: Path,
    signal: np.ndarray,
    entry: dict[str, Any],
    config: ExperimentConfig,
) -> None:
    shapes, log_power, frequencies = extract_native_shape_power(
        signal,
        float(config.raw["signal"]["target_rate"]),
        float(config.raw["signal"]["window_seconds"]),
        float(config.raw["signal"]["hop_seconds"]),
        config.raw["signal"]["bands_hz"],
    )
    de = extract_de(
        signal,
        float(config.raw["signal"]["target_rate"]),
        float(config.raw["signal"]["window_seconds"]),
        float(config.raw["signal"]["hop_seconds"]),
        config.raw["signal"]["bands_hz"],
        int(config.raw["signal"].get("filter_order", 4)),
    ).reshape(log_power.shape)
    arrays: dict[str, Any] = {
        name: value.astype(np.float16)
        for name, value in zip(config.raw["signal"]["bands_hz"], shapes, strict=True)
    }
    for name, grid in zip(config.raw["signal"]["bands_hz"], frequencies, strict=True):
        arrays[f"frequency_{name}"] = grid
    write_npz(
        output,
        **arrays,
        log_power=log_power.astype(np.float32),
        de=de.astype(np.float32),
        label=np.int64(entry["label"]),
        subject=np.int64(entry["subject"]),
        session=np.int64(entry["session"]),
        trial=np.int64(entry["trial"]),
        source_index=np.int64(entry["source_index"]),
        trial_id=np.asarray(str(entry["trial_id"])),
    )


def _seediv_context(config: ExperimentConfig) -> tuple[Path, Path, dict[int, dict[str, list[dict[str, Any]]]]]:
    audit = validate_fixed_cache(config)
    cache_root = Path(str(audit["cache_root"]))
    environment = read_json(cache_root / "environment.json")
    ica_root = Path(str(environment["ica_cache_root"]))
    folds: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for fold in range(1, 16):
        manifest = read_json(cache_root / "folds" / f"fold-{fold:02d}" / "manifest.json")
        folds[fold] = {key: list(value) for key, value in manifest["groups"].items()}
    return cache_root, ica_root, folds


def prepare_base_cache(
    config: ExperimentConfig,
    run_root: Path,
    folds: Sequence[int] | None = None,
) -> dict[str, Any]:
    settings = experiment_settings(config)
    requested_folds = tuple(folds or settings["folds"])
    root = _base_root(config, run_root)
    band_names = list(config.raw["signal"]["bands_hz"])
    channels = int(config.raw["dataset"]["channels"])
    completed = 0
    requested = 0
    started = time.perf_counter()
    if config.dataset == "faced":
        processed = config.data_root / str(config.raw["dataset"]["raw_dir"])
        subjects: set[int] = set()
        for fold in requested_folds:
            source, target = official_fold_subjects(fold)
            subjects.update(source)
            subjects.update(target)
        for subject_position, subject in enumerate(sorted(subjects), 1):
            entries = trial_entries([subject])
            missing = [entry for entry in entries if not _base_valid(
                _base_path(root, str(entry["trial_id"])), band_names, channels
            )]
            requested += len(entries)
            if missing:
                signal = load_processed_subject(processed, subject)
                for entry in missing:
                    video = int(entry["video"]) - 1
                    _save_base_trial(
                        _base_path(root, str(entry["trial_id"])),
                        signal[video, :EEG_CHANNELS],
                        entry,
                        config,
                    )
                del signal
            completed += sum(_base_valid(
                _base_path(root, str(entry["trial_id"])), band_names, channels
            ) for entry in entries)
            LOGGER.info("FACED base cache subject %d/%d", subject_position, len(subjects))
    else:
        _, ica_root, fold_groups = _seediv_context(config)
        entries_by_id: dict[str, dict[str, Any]] = {}
        for fold in requested_folds:
            for group in ("train", "validation", "test"):
                for entry in fold_groups[fold][group]:
                    entries_by_id[str(entry["trial_id"])] = entry
        requested = len(entries_by_id)
        for position, entry in enumerate(entries_by_id.values(), 1):
            output = _base_path(root, str(entry["trial_id"]))
            if not _base_valid(output, band_names, channels):
                cleaned_path = ica_root / "trials" / f"{entry['trial_id']}.npz"
                with np.load(cleaned_path, allow_pickle=False) as archive:
                    signal = np.asarray(archive["cleaned"], dtype=np.float32)
                _save_base_trial(output, signal, entry, config)
            completed += int(_base_valid(output, band_names, channels))
            if position % 25 == 0 or position == requested:
                LOGGER.info("SEED-IV base cache trial %d/%d", position, requested)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "dataset": config.dataset,
        "status": "complete" if completed == requested else "partial",
        "signature": _base_signature(config),
        "root": str(root.resolve()),
        "folds": list(requested_folds),
        "requested_trials": requested,
        "completed_trials": completed,
        "band_names": band_names,
        "storage": "float16 normalized shapes; float32 log power and DE",
        "elapsed_seconds": time.perf_counter() - started,
        "updated_at": utc_now(),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _faced_groups(fold: int, protocol: str, development_subjects: int) -> dict[str, list[dict[str, Any]]]:
    source, target = official_fold_subjects(fold)
    development = source[:development_subjects]
    training = source[development_subjects:]
    if protocol == "conventional_subject_holdout":
        videos = {
            "train": np.arange(VIDEOS),
            "development": np.arange(VIDEOS),
            "test": np.arange(VIDEOS),
        }
    elif protocol == "subject_and_stimulus_holdout":
        videos = stimulus_split()
    else:
        raise ValueError(protocol)

    def selected(subjects: Iterable[int], allowed: np.ndarray) -> list[dict[str, Any]]:
        allowed_set = {int(value) + 1 for value in allowed}
        return [entry for entry in trial_entries(subjects) if int(entry["video"]) in allowed_set]

    return {
        "train": selected(training, videos["train"]),
        "development": selected(development, videos["development"]),
        "test": selected(target, videos["test"]),
    }


def split_groups(config: ExperimentConfig, fold: int, protocol: str) -> dict[str, list[dict[str, Any]]]:
    settings = experiment_settings(config)
    if config.dataset == "faced":
        return _faced_groups(fold, protocol, settings["development_subjects"])
    if protocol != "loso":
        raise ValueError("SEED-IV only supports LOSO")
    _, _, folds = _seediv_context(config)
    value = folds[fold]
    return {
        "train": list(value["train"]),
        "development": list(value["validation"]),
        "test": list(value["test"]),
    }


def _load_base(config: ExperimentConfig, run_root: Path, entry: dict[str, Any]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    path = _base_path(_base_root(config, run_root), str(entry["trial_id"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing base feature {path}; run prepare-base first")
    band_names = list(config.raw["signal"]["bands_hz"])
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["source_index"].item()) != int(entry["source_index"]):
            raise ValueError(f"Base feature metadata mismatch: {path}")
        return (
            [np.asarray(archive[name], dtype=np.float32) for name in band_names],
            np.asarray(archive["log_power"], dtype=np.float32),
            np.asarray(archive["de"], dtype=np.float32),
        )


def _fold_protocol(config: ExperimentConfig, fold: int, protocol: str, groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    settings = experiment_settings(config)
    grids = [
        np.asarray(value, dtype=float).tolist()
        for value in _native_grids(config)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "dataset": config.dataset,
        "fold": fold,
        "protocol": protocol,
        "seed": SEED,
        "source_train_subjects": sorted({int(item["subject"]) for item in groups["train"]}),
        "source_development_subjects": sorted({int(item["subject"]) for item in groups["development"]}),
        "target_subjects": sorted({int(item["subject"]) for item in groups["test"]}),
        "source_train_trials": len(groups["train"]),
        "source_development_trials": len(groups["development"]),
        "target_trials": len(groups["test"]),
        "native_frequency_grids": grids,
        "caps": list(settings["caps"]),
        "conditions": list(settings["conditions"]),
        "models": list(settings["models"]),
        "candidate_windows_per_trial": settings["candidate_windows_per_trial"],
        "maximum_anchor_candidates": settings["maximum_anchor_candidates"],
        "state_fit_scope": "source-train trials only",
        "normalization_fit_scope": "source-train trials only",
        "target_monitor_interval": settings["monitor_interval"],
        "target_metrics_affect_training": False,
        "checkpoint_selection": "fixed final epoch only",
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "training": config.raw["training"],
    }


def _native_grids(config: ExperimentConfig) -> list[np.ndarray]:
    from cmrd.features.rd import native_frequency_grid

    return native_frequency_grid(
        float(config.raw["signal"]["target_rate"]),
        float(config.raw["signal"]["window_seconds"]),
        config.raw["signal"]["bands_hz"],
    )


def _state_root(run_root: Path, config: ExperimentConfig, fold: int, protocol: str, protocol_hash: str) -> Path:
    return run_root / "cache" / "folds" / config.dataset / protocol / f"fold-{fold:02d}" / protocol_hash


def _sample_candidates(
    config: ExperimentConfig,
    run_root: Path,
    entries: Sequence[dict[str, Any]],
) -> tuple[list[np.ndarray], np.ndarray]:
    settings = experiment_settings(config)
    by_band: list[list[np.ndarray]] = [[] for _ in config.raw["signal"]["bands_hz"]]
    powers: list[np.ndarray] = []
    for position, entry in enumerate(entries, 1):
        shapes, log_power, _ = _load_base(config, run_root, entry)
        count = min(settings["candidate_windows_per_trial"], log_power.shape[0])
        indices = np.unique(np.linspace(0, log_power.shape[0] - 1, count, dtype=np.int64))
        for values, selected in zip(by_band, shapes, strict=True):
            values.append(selected[indices])
        powers.append(log_power[indices])
        if position % 100 == 0:
            LOGGER.info("Candidate pass %d/%d trials", position, len(entries))
    candidates = [np.concatenate(values, axis=0) for values in by_band]
    power = np.concatenate(powers, axis=0)
    maximum = settings["maximum_anchor_candidates"]
    if power.shape[0] > maximum:
        selected = np.unique(np.linspace(0, power.shape[0] - 1, maximum, dtype=np.int64))
        candidates = [value[selected] for value in candidates]
        power = power[selected]
    return candidates, power


def _slice_band_state(state: LandmarkBandState, cap: int) -> LandmarkBandState:
    coordinates = min(cap, state.reference.shape[-1] - 1)
    if state.anchors.shape[1] != coordinates:
        raise ValueError("State cap mismatch")
    return state


def _save_state(
    root: Path,
    states: dict[int, list[LandmarkBandState]],
    pca: dict[int, ProjectionState],
    random: dict[int, ProjectionState],
    metadata: dict[str, Any],
    band_names: Sequence[str],
) -> None:
    arrays: dict[str, Any] = {}
    for cap, band_states in states.items():
        for name, state in zip(band_names, band_states, strict=True):
            prefix = f"cap{cap}_{name}"
            arrays[f"{prefix}_reference"] = state.reference
            arrays[f"{prefix}_anchors"] = state.anchors
            arrays[f"{prefix}_d0"] = state.anchor_to_reference_jsd
            arrays[f"{prefix}_whitening"] = state.whitening
            arrays[f"{prefix}_eigenvalues"] = state.eigenvalues
        arrays[f"pca_cap{cap}_mean"] = pca[cap].mean
        arrays[f"pca_cap{cap}_components"] = pca[cap].components
        arrays[f"random_cap{cap}_mean"] = random[cap].mean
        arrays[f"random_cap{cap}_components"] = random[cap].components
    write_npz(root / "state.npz", **arrays)
    write_json(root / "state.json", metadata)


def _load_state(
    root: Path,
    caps: Sequence[int],
    band_names: Sequence[str],
) -> tuple[dict[int, list[LandmarkBandState]], dict[int, ProjectionState], dict[int, ProjectionState]]:
    states: dict[int, list[LandmarkBandState]] = {}
    pca: dict[int, ProjectionState] = {}
    random: dict[int, ProjectionState] = {}
    with np.load(root / "state.npz", allow_pickle=False) as archive:
        for cap in caps:
            band_states = []
            for name in band_names:
                prefix = f"cap{cap}_{name}"
                band_states.append(LandmarkBandState(
                    reference=np.asarray(archive[f"{prefix}_reference"], dtype=np.float32),
                    anchors=np.asarray(archive[f"{prefix}_anchors"], dtype=np.float32),
                    anchor_to_reference_jsd=np.asarray(archive[f"{prefix}_d0"], dtype=np.float32),
                    whitening=np.asarray(archive[f"{prefix}_whitening"], dtype=np.float32),
                    eigenvalues=np.asarray(archive[f"{prefix}_eigenvalues"], dtype=np.float32),
                ))
            states[cap] = band_states
            pca[cap] = ProjectionState(
                mean=np.asarray(archive[f"pca_cap{cap}_mean"], dtype=np.float32),
                components=np.asarray(archive[f"pca_cap{cap}_components"], dtype=np.float32),
            )
            random[cap] = ProjectionState(
                mean=np.asarray(archive[f"random_cap{cap}_mean"], dtype=np.float32),
                components=np.asarray(archive[f"random_cap{cap}_components"], dtype=np.float32),
            )
    return states, pca, random


def fit_fold_state(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol: str,
    groups: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[Path, str]:
    groups = groups or split_groups(config, fold, protocol)
    payload = _fold_protocol(config, fold, protocol, groups)
    protocol_hash = _json_hash(payload)
    root = _state_root(run_root, config, fold, protocol, protocol_hash)
    metadata_path = root / "state.json"
    if metadata_path.is_file() and (root / "state.npz").is_file():
        metadata = read_json(metadata_path)
        if metadata.get("status") == "complete" and metadata.get("protocol_hash") == protocol_hash:
            return root, protocol_hash
    candidates, candidate_power = _sample_candidates(config, run_root, groups["train"])
    settings = experiment_settings(config)
    band_names = list(config.raw["signal"]["bands_hz"])
    states: dict[int, list[LandmarkBandState]] = {}
    pca: dict[int, ProjectionState] = {}
    random: dict[int, ProjectionState] = {}
    full = full_ilr_power(candidates, candidate_power)
    band_sizes = [value.shape[-1] for value in candidates]
    for cap in settings["caps"]:
        states[cap] = [
            fit_landmark_band(value, cap, ridge=settings["landmark_ridge"])
            for value in candidates
        ]
        dimension = capped_dimension(band_sizes, cap)
        pca[cap] = fit_channel_pca(full, dimension)
        random[cap] = fit_random_projection(full.shape[1], full.shape[2], dimension, SEED + cap)
    effective_rank = {
        f"cap{cap}_{name}": [int(np.count_nonzero(row > 0)) for row in state.eigenvalues]
        for cap, band_states in states.items()
        for name, state in zip(band_names, band_states, strict=True)
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "protocol_hash": protocol_hash,
        "protocol": payload,
        "candidate_count": int(candidate_power.shape[0]),
        "candidate_source_trials": len(groups["train"]),
        "candidate_source_subjects": sorted({int(item["subject"]) for item in groups["train"]}),
        "band_sizes": band_sizes,
        "full_dimension_per_channel": full_dimension(band_sizes),
        "capped_dimensions_per_channel": {
            str(cap): capped_dimension(band_sizes, cap) for cap in settings["caps"]
        },
        "effective_anchor_rank_by_channel": effective_rank,
        "target_data_used_to_fit_state": False,
        "created_at": utc_now(),
    }
    _save_state(root, states, pca, random, metadata, band_names)
    return root, protocol_hash


def _condition_cap(condition: str) -> int:
    marker = "_cap"
    if marker not in condition:
        raise ValueError(f"Condition has no cap: {condition}")
    return int(condition.rsplit(marker, 1)[1])


def _condition_features(
    condition: str,
    shapes: Sequence[np.ndarray],
    log_power: np.ndarray,
    de: np.ndarray,
    states: dict[int, list[LandmarkBandState]],
    pca: dict[int, ProjectionState],
    random: dict[int, ProjectionState],
) -> np.ndarray:
    if condition == "de":
        structured = de
    elif condition == "log_band_power":
        structured = log_power
    elif condition == "scalar_jsd_power":
        structured = scalar_jsd_power(shapes, log_power, states[max(states)])
    elif condition == "ilr_power_full":
        structured = full_ilr_power(shapes, log_power)
    elif condition == "log_psd_full":
        structured = full_log_psd(shapes, log_power)
    else:
        cap = _condition_cap(condition)
        if condition.startswith("raw_landmark"):
            structured = landmark_power(shapes, log_power, states[cap], orthogonalized=False)
        elif condition.startswith("nystrom_landmark"):
            structured = landmark_power(shapes, log_power, states[cap], orthogonalized=True)
        elif condition.startswith("pca_ilr"):
            structured = apply_projection(full_ilr_power(shapes, log_power), pca[cap])
        elif condition.startswith("random_ilr"):
            structured = apply_projection(full_ilr_power(shapes, log_power), random[cap])
        else:
            raise ValueError(condition)
    sequence = structured.reshape(structured.shape[0], -1).astype(np.float32)
    return np.concatenate(
        [sequence.mean(axis=0, dtype=np.float64), sequence.std(axis=0, dtype=np.float64)]
    ).astype(np.float32)


def prepare_fold_features(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol: str,
) -> dict[str, Any]:
    settings = experiment_settings(config)
    groups = split_groups(config, fold, protocol)
    state_root, protocol_hash = fit_fold_state(config, run_root, fold, protocol, groups)
    feature_path = state_root / "pooled_features.npz"
    manifest_path = state_root / "feature_manifest.json"
    if feature_path.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete" and manifest.get("protocol_hash") == protocol_hash:
            return manifest
    band_names = list(config.raw["signal"]["bands_hz"])
    states, pca, random = _load_state(state_root, settings["caps"], band_names)
    arrays: dict[str, Any] = {}
    feature_dimensions: dict[str, int] = {}
    for group_name, entries in groups.items():
        vectors: dict[str, list[np.ndarray]] = {condition: [] for condition in settings["conditions"]}
        labels: list[int] = []
        subjects: list[int] = []
        trial_ids: list[str] = []
        for position, entry in enumerate(entries, 1):
            shapes, log_power, de = _load_base(config, run_root, entry)
            for condition in settings["conditions"]:
                vectors[condition].append(_condition_features(
                    condition, shapes, log_power, de, states, pca, random
                ))
            labels.append(int(entry["label"]))
            subjects.append(int(entry["subject"]))
            trial_ids.append(str(entry["trial_id"]))
            if position % 100 == 0:
                LOGGER.info("%s feature transform %d/%d", group_name, position, len(entries))
        for condition, values in vectors.items():
            array = np.stack(values).astype(np.float32)
            arrays[f"{condition}__{group_name}"] = array
            feature_dimensions[condition] = int(array.shape[1])
        arrays[f"labels__{group_name}"] = np.asarray(labels, dtype=np.int64)
        arrays[f"subjects__{group_name}"] = np.asarray(subjects, dtype=np.int64)
        arrays[f"trial_ids__{group_name}"] = np.asarray(trial_ids)
    write_npz(feature_path, **arrays)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "dataset": config.dataset,
        "fold": fold,
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "state_root": str(state_root.resolve()),
        "feature_path": str(feature_path.resolve()),
        "conditions": list(settings["conditions"]),
        "feature_dimensions_after_mean_std_pooling": feature_dimensions,
        "group_trials": {name: len(entries) for name, entries in groups.items()},
        "target_data_used_to_fit_state": False,
        "updated_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_matrix_arrays(feature_path: Path, condition: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with np.load(feature_path, allow_pickle=False) as archive:
        return {
            group: (
                np.asarray(archive[f"{condition}__{group}"], dtype=np.float32),
                np.asarray(archive[f"labels__{group}"], dtype=np.int64),
            )
            for group in ("train", "development", "test")
        }


def _save_confusion(output: Path, split: str, epoch: int | str, metrics: dict[str, Any], class_names: Sequence[str]) -> None:
    value = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    stem = f"{split}_confusion_epoch_{int(epoch):03d}" if isinstance(epoch, int) else f"{split}_confusion_{epoch}"
    _write_csv(
        output / "confusions" / f"{stem}.csv",
        [
            {"target": class_names[row], **{class_names[col]: int(value[row, col]) for col in range(value.shape[1])}}
            for row in range(value.shape[0])
        ],
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(max(5, len(class_names) * 0.65), max(4, len(class_names) * 0.55)))
        image = axis.imshow(value, cmap="Blues")
        axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for row in range(value.shape[0]):
            for column in range(value.shape[1]):
                axis.text(column, row, str(value[row, column]), ha="center", va="center", fontsize=7)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        path = output / "confusions" / f"{stem}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=140)
        plt.close(figure)
    except ModuleNotFoundError:
        pass


def _evaluate_predictions(targets: np.ndarray, predictions: np.ndarray, classes: int) -> dict[str, Any]:
    return classification_metrics(np.asarray(targets), np.asarray(predictions), classes)


class PooledMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _fit_classical(
    model_name: str,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
    classes: int,
    class_names: Sequence[str],
) -> dict[str, Any]:
    scaler = StandardScaler().fit(arrays["train"][0])
    x_train = scaler.transform(arrays["train"][0])
    y_train = arrays["train"][1]
    estimator = (
        LogisticRegression(max_iter=3000, random_state=SEED, class_weight="balanced")
        if model_name == "logistic_regression"
        else LinearSVC(random_state=SEED, dual="auto", max_iter=5000, tol=1e-3, class_weight="balanced")
    )
    estimator.fit(x_train, y_train)
    metrics: dict[str, Any] = {}
    for group, (features, targets) in arrays.items():
        predictions = estimator.predict(scaler.transform(features))
        metrics[group] = _evaluate_predictions(targets, predictions, classes)
        _save_confusion(output, group, "final", metrics[group], class_names)
    with (output / "model.pkl").open("wb") as stream:
        pickle.dump({"model": estimator, "scaler": scaler, "seed": SEED}, stream)
    return {"metrics": metrics, "monitoring": []}


def _fit_mlp(
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    output: Path,
    config: ExperimentConfig,
    classes: int,
    class_names: Sequence[str],
    *,
    epochs_override: int | None = None,
    monitor_override: int | None = None,
) -> dict[str, Any]:
    training = config.raw["training"]
    epochs = int(epochs_override or training["epochs"])
    monitor = int(monitor_override or experiment_settings(config)["monitor_interval"])
    seed_everything(SEED, bool(training.get("deterministic", True)))
    device = select_device(str(training.get("device", "auto")))
    scaler = StandardScaler().fit(arrays["train"][0])
    scaled = {
        name: (scaler.transform(features).astype(np.float32), targets)
        for name, (features, targets) in arrays.items()
    }
    model = PooledMLP(
        scaled["train"][0].shape[1],
        int(training.get("hidden", 128)),
        classes,
        float(training.get("dropout", 0.15)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    counts = np.bincount(scaled["train"][1], minlength=classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts * classes, 1.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(weights, dtype=torch.float32, device=device),
        label_smoothing=float(training.get("label_smoothing", 0.0)),
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(scaled["train"][0]),
            torch.from_numpy(scaled["train"][1]),
        ),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: list[dict[str, Any]] = []
    monitoring: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * labels.shape[0]
            examples += int(labels.shape[0])
        history.append({"epoch": epoch, "train_loss": loss_sum / max(examples, 1)})
        if epoch % monitor == 0 or epoch == epochs:
            row: dict[str, Any] = {"epoch": epoch}
            for group, (features, targets) in scaled.items():
                model.eval()
                with torch.no_grad():
                    logits = model(torch.from_numpy(features).to(device))
                    predictions = logits.argmax(dim=1).cpu().numpy()
                metrics = _evaluate_predictions(targets, predictions, classes)
                row[group] = metrics
                _save_confusion(output, group, epoch, metrics, class_names)
            monitoring.append(row)
            write_json(output / "monitoring.json", monitoring)
            _write_csv(output / "training_history.csv", history)
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "input_dim": scaled["train"][0].shape[1],
        "hidden": int(training.get("hidden", 128)),
        "classes": classes,
        "dropout": float(training.get("dropout", 0.15)),
        "epochs": epochs,
        "seed": SEED,
        "target_monitoring_during_training": True,
        "checkpoint_selection": "fixed_final_epoch_only",
    }
    torch.save(checkpoint, output / "final_model.pt")
    return {"metrics": monitoring[-1] | {}, "monitoring": monitoring, "history": history}


def _task_output(run_root: Path, dataset: str, protocol: str, fold: int, condition: str, model: str) -> Path:
    return run_root / "tasks" / dataset / protocol / f"fold-{fold:02d}" / condition / model


def run_task(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol: str,
    condition: str,
    model_name: str,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    settings = experiment_settings(config)
    if condition not in settings["conditions"] or model_name not in settings["models"]:
        raise ValueError(f"Unknown task {condition}/{model_name}")
    manifest = prepare_fold_features(config, run_root, fold, protocol)
    output = _task_output(run_root, config.dataset, protocol, fold, condition, model_name)
    result_path = output / "result.json"
    if resume and result_path.is_file() and read_json(result_path).get("status") == "complete":
        return read_json(result_path)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "status.json", {
        "status": "running", "started_at": utc_now(), "condition": condition, "model": model_name
    })
    started = time.perf_counter()
    try:
        arrays = _load_matrix_arrays(Path(str(manifest["feature_path"])), condition)
        classes = int(config.raw["dataset"]["classes"])
        class_names = list(EMOTION_NAMES) if config.dataset == "faced" else [str(index) for index in range(classes)]
        trained = (
            _fit_mlp(arrays, output, config, classes, class_names)
            if model_name == "pooled_mlp"
            else _fit_classical(model_name, arrays, output, classes, class_names)
        )
        final_metrics = trained["metrics"]
        if model_name == "pooled_mlp":
            final_metrics = {name: final_metrics[name] for name in ("train", "development", "test")}
        result = {
            "schema_version": SCHEMA_VERSION,
            "family": FAMILY,
            "status": "complete",
            "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
            "dataset": config.dataset,
            "fold": fold,
            "protocol": protocol,
            "protocol_hash": manifest["protocol_hash"],
            "condition": condition,
            "model": model_name,
            "feature_dimension": manifest["feature_dimensions_after_mean_std_pooling"][condition],
            "final_metrics": final_metrics,
            "target_metrics_used_for_selection": False,
            "checkpoint_selection": "fixed_final_epoch_only",
            "target_monitor_interval": settings["monitor_interval"] if model_name == "pooled_mlp" else None,
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at": utc_now(),
        }
        write_json(result_path, result)
        write_json(output / "status.json", {"status": "complete", "completed_at": utc_now()})
        return result
    except BaseException as error:
        write_json(output / "status.json", {
            "status": "failed",
            "failed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
        raise


def run_matrix(
    config: ExperimentConfig,
    run_root: Path,
    folds: Sequence[int],
    protocols: Sequence[str],
    conditions: Sequence[str],
    models: Sequence[str],
    *,
    resume: bool,
) -> dict[str, Any]:
    prepare_base_cache(config, run_root, folds)
    completed = 0
    failed = 0
    for fold in folds:
        for protocol in protocols:
            prepare_fold_features(config, run_root, fold, protocol)
            for condition in conditions:
                for model in models:
                    try:
                        run_task(config, run_root, fold, protocol, condition, model, resume=resume)
                        completed += 1
                    except BaseException:
                        LOGGER.exception("Task failed: %s/%s/fold-%02d/%s/%s", config.dataset, protocol, fold, condition, model)
                        failed += 1
    payload = {
        "dataset": config.dataset,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "evidence_status": "exploratory_target_monitored",
        "updated_at": utc_now(),
    }
    write_json(run_root / f"matrix_{config.dataset}.json", payload)
    return payload


def matrix_status(run_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((run_root / "tasks").glob("**/status.json")) if (run_root / "tasks").is_dir() else []:
        value = read_json(path)
        rows.append({"path": str(path.parent.relative_to(run_root)), "status": value.get("status")})
    counts = {status: sum(row["status"] == status for row in rows) for status in ("complete", "running", "failed")}
    return {"run_root": str(run_root.resolve()), "tasks": len(rows), "counts": counts, "rows": rows}


def summarize(run_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "tasks").glob("**/result.json")) if (run_root / "tasks").is_dir() else []:
        value = read_json(path)
        if value.get("status") != "complete":
            continue
        metrics = value["final_metrics"]
        rows.append({
            "dataset": value["dataset"],
            "protocol": value["protocol"],
            "fold": value["fold"],
            "condition": value["condition"],
            "model": value["model"],
            "feature_dimension": value["feature_dimension"],
            **_flatten_metrics("source_train", metrics["train"]),
            **_flatten_metrics("source_development", metrics["development"]),
            **_flatten_metrics("target_diagnostic", metrics["test"]),
        })
    _write_csv(run_root / "summary.csv", rows)
    payload = {
        "family": FAMILY,
        "completed_results": len(rows),
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "summary_csv": str((run_root / "summary.csv").resolve()),
        "updated_at": utc_now(),
    }
    write_json(run_root / "summary.json", payload)
    return payload


def synthetic_smoke(config: ExperimentConfig, run_root: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    channels = 3
    sizes = [3, 4, 6, 8, 9]
    classes = int(config.raw["dataset"]["classes"])
    group_counts = {
        "train": max(2 * classes, 12),
        "development": max(classes, 8),
        "test": max(classes, 8),
    }
    trial_count = sum(group_counts.values())
    windows = trial_count * 2
    candidates = []
    for size in sizes:
        value = rng.random((windows, channels, size), dtype=np.float32)
        value /= value.sum(axis=-1, keepdims=True)
        candidates.append(value)
    power = rng.normal(size=(windows, channels, 5)).astype(np.float32)
    states = {cap: [fit_landmark_band(value, cap) for value in candidates] for cap in CAPS}
    full = full_ilr_power(candidates, power)
    pca = {cap: fit_channel_pca(full, capped_dimension(sizes, cap)) for cap in CAPS}
    random = {
        cap: fit_random_projection(channels, full.shape[-1], capped_dimension(sizes, cap), SEED + cap)
        for cap in CAPS
    }
    feature_shapes = {}
    de = rng.normal(size=(windows, channels, 5)).astype(np.float32)
    for condition in all_conditions():
        feature_shapes[condition] = list(_condition_features(
            condition, candidates, power, de, states, pca, random
        ).shape)
    condition = "nystrom_landmark_power_cap4"
    pooled = np.stack([
        _condition_features(
            condition,
            [value[index:index + 2] for value in candidates],
            power[index:index + 2],
            de[index:index + 2],
            states,
            pca,
            random,
        )
        for index in range(0, windows, 2)
    ])
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    offset = 0
    for group, count in group_counts.items():
        arrays[group] = (
            pooled[offset:offset + count],
            np.resize(np.arange(classes, dtype=np.int64), count),
        )
        offset += count
    smoke_output = run_root / "smoke_training" / config.dataset
    smoke_output.mkdir(parents=True, exist_ok=True)
    trained = _fit_mlp(
        arrays,
        smoke_output,
        config,
        classes,
        list(EMOTION_NAMES) if config.dataset == "faced" else [str(index) for index in range(classes)],
        epochs_override=2,
        monitor_override=1,
    )
    payload = {
        "status": "complete",
        "dataset_config": config.dataset,
        "conditions": feature_shapes,
        "finite": True,
        "training_smoke": {
            "condition": condition,
            "epochs": 2,
            "monitor_points": len(trained["monitoring"]),
            "output": str(smoke_output.resolve()),
        },
        "created_at": utc_now(),
    }
    write_json(run_root / f"smoke_{config.dataset}.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Source-only spectral-atlas experiments for FACED and SEED-IV")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name: str) -> argparse.ArgumentParser:
        value = subparsers.add_parser(name)
        value.add_argument("--config", required=True)
        value.add_argument("--run-root")
        value.add_argument("--fold", type=int, action="append")
        value.add_argument("--protocol", action="append")
        return value

    configured("validate")
    configured("prepare-base")
    configured("prepare-features")
    configured("smoke")
    matrix = configured("matrix")
    matrix.add_argument("--condition", action="append")
    matrix.add_argument("--model", action="append")
    matrix.add_argument("--no-resume", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    if args.command in {"status", "summarize"}:
        payload = matrix_status(Path(args.run_root)) if args.command == "status" else summarize(Path(args.run_root))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    config = load_config(args.config)
    settings = experiment_settings(config)
    run_root = Path(args.run_root).resolve() if args.run_root else config.run_root
    folds = tuple(args.fold or settings["folds"])
    protocols = tuple(args.protocol or settings["protocols"])
    if args.command == "validate":
        payload = validate_sources(config)
    elif args.command == "prepare-base":
        payload = prepare_base_cache(config, run_root, folds)
    elif args.command == "prepare-features":
        payload = {
            f"{protocol}/fold-{fold:02d}": prepare_fold_features(config, run_root, fold, protocol)
            for fold in folds for protocol in protocols
        }
    elif args.command == "smoke":
        payload = synthetic_smoke(config, run_root)
    else:
        payload = run_matrix(
            config,
            run_root,
            folds,
            protocols,
            tuple(args.condition or settings["conditions"]),
            tuple(args.model or settings["models"]),
            resume=not args.no_resume,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.command == "matrix" and int(payload.get("failed_tasks", 0)):
        return 1
    return 0
