from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from cmrd.config import ExperimentConfig, load_config
from cmrd.data.records import TrialSample
from cmrd.faced import (
    EEG_CHANNELS,
    EMOTION_NAMES,
    FOLDS,
    RATE_HZ,
    RECORDED_CHANNELS,
    SAMPLES,
    SUBJECTS,
    VIDEO_LABELS,
    VIDEOS,
    load_processed_subject,
    official_fold_subjects,
    validate_faced_data,
)
from cmrd.features.rd import (
    extract_native_spectral_distributions,
    fisher_rao_log_map,
    native_frequency_grid,
    normalize_histograms,
    transform_native_fisher_rao_pca,
    transform_native_sqrt_jsd,
)
from cmrd.features.de import extract_de
from cmrd.fixed_protocol import evaluate_locked_checkpoint, fit_locked_source_model
from cmrd.io import read_json, write_json, write_npz
from cmrd.training.runtime import select_device


LOGGER = logging.getLogger("cmrd.faced_native")
SCHEMA_VERSION = 1
FAMILY = "FACED-Native-Compact-v1"
SEED = 42
CONDITIONS = {
    "de_base": "de",
    "native_sqrt_jsd_base": "native_sqrt_jsd",
    "native_fisher_rao_base": "native_fisher_rao_pca",
}


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


def _data_dirs(config: ExperimentConfig) -> tuple[Path, Path]:
    return (
        config.data_root / str(config.raw["dataset"]["raw_dir"]),
        config.data_root / str(config.raw["dataset"]["metadata_dir"]),
    )


def experiment_settings(config: ExperimentConfig) -> dict[str, Any]:
    if config.dataset != "faced":
        raise ValueError("FACED runner requires experiment.dataset=faced")
    raw = copy.deepcopy(config.raw.get("faced_native", {}))
    settings = {
        "seed": int(raw.get("seed", SEED)),
        "pca_max_windows_per_trial": int(raw.get("pca_max_windows_per_trial", 8)),
        "architecture": copy.deepcopy(raw.get("architecture", {})),
        "conditions": copy.deepcopy(raw.get("conditions", {})),
    }
    if settings["seed"] != SEED:
        raise ValueError("FACED native protocol is frozen to seed=42")
    if settings["pca_max_windows_per_trial"] < 2:
        raise ValueError("pca_max_windows_per_trial must be at least 2")
    if tuple(settings["conditions"]) != tuple(CONDITIONS):
        raise ValueError(f"Conditions must be declared in order: {tuple(CONDITIONS)}")
    for condition, representation in CONDITIONS.items():
        if settings["conditions"][condition].get("representation") != representation:
            raise ValueError(f"{condition} must use {representation}")

    expected_architecture = {
        "d_model": 128,
        "heads": 4,
        "layers": 3,
        "feedforward": 512,
        "dropout": 0.15,
    }
    for key, expected in expected_architecture.items():
        if float(settings["architecture"].get(key, float("nan"))) != float(expected):
            raise ValueError(f"Base architecture requires {key}={expected}")

    signal = config.raw["signal"]
    if (
        float(signal["original_rate"]) != RATE_HZ
        or float(signal["target_rate"]) != RATE_HZ
        or float(signal["window_seconds"]) != 1.0
        or float(signal["hop_seconds"]) != 1.0
    ):
        raise ValueError("FACED Processed_data requires 250 Hz and non-overlapping 1 s windows")
    expected_bands = {
        "delta": [1.0, 4.0],
        "theta": [4.0, 8.0],
        "alpha": [8.0, 14.0],
        "beta": [14.0, 30.0],
        "gamma": [30.0, 47.0],
    }
    if signal["bands_hz"] != expected_bands:
        raise ValueError(f"FACED bands must match the published bands: {expected_bands}")
    if int(config.raw["feature"].get("de_filter_order", 0)) != 4:
        raise ValueError("FACED DE baseline requires feature.de_filter_order=4")

    training = config.raw["training"]
    adapted = {
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "minimum_learning_rate": 1e-6,
        "weight_decay": 0.01,
        "label_smoothing": 0.05,
        "gradient_clip_norm": 1.0,
    }
    for key, expected in adapted.items():
        if float(training.get(key, float("nan"))) != float(expected):
            raise ValueError(f"FACED adapted training requires {key}={expected}")
    if training.get("deterministic") is not True or [int(x) for x in training.get("seeds", [])] != [SEED]:
        raise ValueError("FACED adapted training requires deterministic=true and seeds=[42]")
    return settings


def validate_sources(config: ExperimentConfig, *, deep: bool = False) -> dict[str, Any]:
    experiment_settings(config)
    processed_dir, metadata_dir = _data_dirs(config)
    audit = validate_faced_data(processed_dir, metadata_dir, deep=deep)
    audit.update({
        "preprocessing": "official FACED Processed_data; no additional signal preprocessing",
        "preprocessing_steps": [
            "last 30 seconds", "250 Hz", "0.05-47 Hz bandpass",
            "bad-channel detection/interpolation", "ICA ocular removal",
            "common-average rereference", "cohort channel-order harmonization",
        ],
        "feature_only_from_processed_data": True,
    })
    return audit


def protocol_payload(config: ExperimentConfig, audit: dict[str, Any]) -> dict[str, Any]:
    settings = experiment_settings(config)
    bands = config.raw["signal"]["bands_hz"]
    grids = native_frequency_grid(RATE_HZ, 1.0, bands)
    return {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "dataset": "faced",
        "metadata_md5": audit["metadata_md5"],
        "outer_protocol": "official contiguous subject 10-fold: 12 subjects in folds 1-9, 15 in fold 10",
        "fold_subjects": {
            f"fold-{fold:02d}": official_fold_subjects(fold)[1]
            for fold in range(1, FOLDS + 1)
        },
        "labels": list(EMOTION_NAMES),
        "video_labels": VIDEO_LABELS.tolist(),
        "processed_shape": [VIDEOS, RECORDED_CHANNELS, SAMPLES],
        "channel_selection": "first 30 EEG channels; exclude final HEOR/HEOL mastoid channels",
        "signal_preprocessing": "none beyond official Processed_data",
        "window_seconds": 1.0,
        "hop_seconds": 1.0,
        "spectral_estimator": "Hann modified periodogram; nfft=nperseg=250; no zero padding",
        "native_frequencies_hz": {
            name: grid.astype(float).tolist()
            for name, grid in zip(bands, grids, strict=True)
        },
        "reference_fit_scope": "outer-fold source subjects only",
        "reference_rules": {
            "de": "five band-specific fourth-order Butterworth filters followed by Gaussian DE",
            "sqrt_jsd": "pooled arithmetic probability mean",
            "fisher_rao": "normalized mean square-root embedding",
            "fisher_rao_pca": "per-channel-band source-only tangent PC1",
        },
        "pca_max_windows_per_trial": settings["pca_max_windows_per_trial"],
        "feature_shape": [30, EEG_CHANNELS, len(bands)],
        "conditions": settings["conditions"],
        "source_zscore": True,
        "architecture": settings["architecture"],
        "training": config.raw["training"],
        "target_used_during_training": False,
        "checkpoint_selection": "fixed final epoch",
    }


def lock_experiment(config: ExperimentConfig, run_root: Path) -> dict[str, Any]:
    audit = validate_sources(config)
    protocol = protocol_payload(config, audit)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "protocol_hash": _json_hash(protocol),
        "target_used_during_training": False,
        "checkpoint_selection": "fixed final epoch",
        "locked_at": utc_now(),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "experiment_lock.json"
    if path.is_file():
        existing = read_json(path)
        left = {key: value for key, value in existing.items() if key != "locked_at"}
        right = {key: value for key, value in payload.items() if key != "locked_at"}
        if left != right:
            raise ValueError(f"Existing FACED lock conflicts with active config: {path}")
        return existing
    write_json(path, payload)
    write_json(run_root / "protocol.json", protocol)
    write_json(run_root / "data_audit.json", audit)
    return payload


def _require_lock(config: ExperimentConfig, run_root: Path) -> tuple[dict[str, Any], str]:
    audit = validate_sources(config)
    expected = _json_hash(protocol_payload(config, audit))
    path = run_root / "experiment_lock.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run the Lock stage first: {path}")
    lock = read_json(path)
    if lock.get("protocol_hash") != expected:
        raise ValueError("FACED experiment lock does not match the active config/data")
    return audit, expected


def _spectra_signature(config: ExperimentConfig) -> str:
    return _json_hash({
        "schema_version": SCHEMA_VERSION,
        "dataset": config.raw["dataset"],
        "signal": config.raw["signal"],
        "de_filter_order": int(config.raw["feature"]["de_filter_order"]),
        "processed_shape": [VIDEOS, RECORDED_CHANNELS, SAMPLES],
        "eeg_channels_used": EEG_CHANNELS,
        "estimators": ["scipy-welch-hann-native-nfft", "butterworth-gaussian-de"],
    })


def _spectra_root(config: ExperimentConfig, run_root: Path) -> Path:
    return run_root / "cache" / "native_spectra" / _spectra_signature(config)


def _spectra_path(root: Path, subject: int) -> Path:
    return root / "subjects" / f"sub{subject:03d}.npz"


def _valid_spectra(path: Path, band_names: Sequence[str], band_sizes: Sequence[int]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                archive["de"].shape == (VIDEOS, 30, EEG_CHANNELS * 5)
                and np.isfinite(archive["de"]).all()
                and all(
                archive[name].shape == (VIDEOS, 30, EEG_CHANNELS, size)
                and np.isfinite(archive[name]).all()
                for name, size in zip(band_names, band_sizes, strict=True)
                )
            )
    except (OSError, ValueError, KeyError):
        return False


def prepare_spectra(
    config: ExperimentConfig,
    run_root: Path,
    subjects: Sequence[int] | None = None,
) -> dict[str, Any]:
    experiment_settings(config)
    processed_dir, _ = _data_dirs(config)
    root = _spectra_root(config, run_root)
    band_names = list(config.raw["signal"]["bands_hz"])
    grids = native_frequency_grid(RATE_HZ, 1.0, config.raw["signal"]["bands_hz"])
    band_sizes = [len(grid) for grid in grids]
    selected = list(range(SUBJECTS)) if subjects is None else sorted(set(map(int, subjects)))
    for subject in selected:
        if not 0 <= subject < SUBJECTS:
            raise ValueError(f"Invalid FACED subject: {subject}")
        output = _spectra_path(root, subject)
        if _valid_spectra(output, band_names, band_sizes):
            continue
        signal = load_processed_subject(processed_dir, subject)
        by_band: list[list[np.ndarray]] = [[] for _ in band_names]
        de_features: list[np.ndarray] = []
        for video in range(VIDEOS):
            distributions, actual_grids = extract_native_spectral_distributions(
                signal[video, :EEG_CHANNELS], RATE_HZ, 1.0, 1.0, config.raw["signal"]["bands_hz"]
            )
            if any(not np.array_equal(left, right) for left, right in zip(grids, actual_grids, strict=True)):
                raise RuntimeError("FACED native frequency grid changed during extraction")
            for values, distribution in zip(by_band, distributions, strict=True):
                values.append(distribution)
            de_features.append(extract_de(
                signal[video, :EEG_CHANNELS], RATE_HZ, 1.0, 1.0,
                config.raw["signal"]["bands_hz"],
                int(config.raw["feature"]["de_filter_order"]),
            ))
        arrays = {
            name: np.stack(values).astype(np.float32)
            for name, values in zip(band_names, by_band, strict=True)
        }
        arrays["de"] = np.stack(de_features).astype(np.float32)
        write_npz(output, **arrays)
        LOGGER.info("Prepared native spectra for sub%03d (%d/%d)", subject, selected.index(subject) + 1, len(selected))
        del signal, by_band, de_features
        gc.collect()

    complete = [
        subject for subject in range(SUBJECTS)
        if _valid_spectra(_spectra_path(root, subject), band_names, band_sizes)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "FACED",
        "signature": _spectra_signature(config),
        "root": str(root.resolve()),
        "estimator": "Hann modified periodogram, native 250-point FFT",
        "de_estimator": "fourth-order Butterworth subband filtering plus Gaussian differential entropy",
        "band_names": band_names,
        "band_sizes": band_sizes,
        "subjects_complete": complete,
        "all_subjects_complete": len(complete) == SUBJECTS,
        "updated_at": utc_now(),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def _load_spectra(config: ExperimentConfig, run_root: Path, subject: int) -> list[np.ndarray]:
    path = _spectra_path(_spectra_root(config, run_root), subject)
    band_names = list(config.raw["signal"]["bands_hz"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing spectra for sub{subject:03d}; run the Spectra stage first")
    with np.load(path, allow_pickle=False) as archive:
        return [np.asarray(archive[name], dtype=np.float32) for name in band_names]


def _load_de(config: ExperimentConfig, run_root: Path, subject: int) -> np.ndarray:
    path = _spectra_path(_spectra_root(config, run_root), subject)
    if not path.is_file():
        raise FileNotFoundError(f"Missing DE/native cache for sub{subject:03d}; run the Spectra stage first")
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["de"], dtype=np.float32)
    if value.shape != (VIDEOS, 30, EEG_CHANNELS * 5) or not np.isfinite(value).all():
        raise ValueError(f"Invalid FACED DE cache for sub{subject:03d}: {value.shape}")
    return value


def _fold_cache_root(run_root: Path, fold: int, protocol_hash: str) -> Path:
    return run_root / "cache" / f"fold-{fold:02d}" / protocol_hash


def _fit_fold_state(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol_hash: str,
    source_subjects: Sequence[int],
) -> dict[str, list[np.ndarray]]:
    root = _fold_cache_root(run_root, fold, protocol_hash)
    state_path = root / "feature_state.npz"
    metadata_path = root / "feature_state.json"
    band_names = list(config.raw["signal"]["bands_hz"])
    if state_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("source_subjects") == list(source_subjects):
            with np.load(state_path, allow_pickle=False) as archive:
                return {
                    key: [np.asarray(archive[f"{key}_{name}"], dtype=np.float32) for name in band_names]
                    for key in ("arithmetic", "fisher", "means", "components")
                }

    grids = native_frequency_grid(RATE_HZ, 1.0, config.raw["signal"]["bands_hz"])
    arithmetic_sums = [np.zeros((EEG_CHANNELS, len(grid)), dtype=np.float64) for grid in grids]
    root_sums = [np.zeros_like(value) for value in arithmetic_sums]
    count = 0
    for index, subject in enumerate(source_subjects, 1):
        for band, values in enumerate(_load_spectra(config, run_root, subject)):
            flat = values.reshape(-1, values.shape[-2], values.shape[-1])
            arithmetic_sums[band] += flat.sum(axis=0, dtype=np.float64)
            root_sums[band] += np.sqrt(flat).sum(axis=0, dtype=np.float64)
        count += VIDEOS * 30
        if index % 12 == 0:
            LOGGER.info("Fold %02d reference pass %d/%d subjects", fold, index, len(source_subjects))
    arithmetic = [normalize_histograms(value / count).astype(np.float32) for value in arithmetic_sums]
    fisher = [normalize_histograms(np.square(value / count)).astype(np.float32) for value in root_sums]

    maximum = experiment_settings(config)["pca_max_windows_per_trial"]
    selected = np.unique(np.linspace(0, 29, min(maximum, 30), dtype=np.int64))
    sums = [np.zeros_like(reference, dtype=np.float64) for reference in fisher]
    crosses = [
        np.zeros((EEG_CHANNELS, reference.shape[-1], reference.shape[-1]), dtype=np.float64)
        for reference in fisher
    ]
    pca_count = 0
    for index, subject in enumerate(source_subjects, 1):
        for band, (values, reference) in enumerate(zip(_load_spectra(config, run_root, subject), fisher, strict=True)):
            sampled = values[:, selected].reshape(-1, EEG_CHANNELS, values.shape[-1])
            tangent = fisher_rao_log_map(sampled, reference)
            sums[band] += tangent.sum(axis=0, dtype=np.float64)
            crosses[band] += np.einsum("ncf,ncg->cfg", tangent, tangent, dtype=np.float64)
        pca_count += VIDEOS * len(selected)
        if index % 12 == 0:
            LOGGER.info("Fold %02d Fisher-Rao PCA pass %d/%d subjects", fold, index, len(source_subjects))

    means: list[np.ndarray] = []
    components: list[np.ndarray] = []
    explained: dict[str, list[float]] = {}
    for name, total, cross in zip(band_names, sums, crosses, strict=True):
        mean = total / pca_count
        covariance = cross / pca_count - np.einsum("cf,cg->cfg", mean, mean)
        axes = np.empty_like(mean, dtype=np.float32)
        ratios: list[float] = []
        for channel in range(EEG_CHANNELS):
            eigenvalues, eigenvectors = np.linalg.eigh(covariance[channel])
            axis = eigenvectors[:, -1]
            pivot = int(np.argmax(np.abs(axis)))
            if axis[pivot] < 0:
                axis = -axis
            axes[channel] = axis.astype(np.float32)
            denominator = max(float(np.maximum(eigenvalues, 0).sum()), 1e-12)
            ratios.append(float(max(eigenvalues[-1], 0.0) / denominator))
        means.append(mean.astype(np.float32))
        components.append(axes)
        explained[name] = ratios

    arrays: dict[str, np.ndarray] = {}
    for key, values in {
        "arithmetic": arithmetic,
        "fisher": fisher,
        "means": means,
        "components": components,
    }.items():
        for name, value in zip(band_names, values, strict=True):
            arrays[f"{key}_{name}"] = value
    write_npz(state_path, **arrays)
    write_json(metadata_path, {
        "schema_version": SCHEMA_VERSION,
        "fold": fold,
        "protocol_hash": protocol_hash,
        "source_subjects": list(source_subjects),
        "target_subjects": official_fold_subjects(fold)[1],
        "reference_windows": count,
        "pca_windows": pca_count,
        "pca_sample_indices": selected.tolist(),
        "pca_explained_variance_ratio_pc1_by_channel": explained,
        "target_data_used": False,
        "created_at": utc_now(),
    })
    return {"arithmetic": arithmetic, "fisher": fisher, "means": means, "components": components}


def _valid_feature_subject(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return all(
                archive[key].shape == (VIDEOS, 30, EEG_CHANNELS * 5)
                and np.isfinite(archive[key]).all()
                for key in ("native_sqrt_jsd", "native_fisher_rao_pca")
            )
    except (OSError, ValueError, KeyError):
        return False


def prepare_fold_features(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol_hash: str | None = None,
    *,
    source_override: Sequence[int] | None = None,
    subjects: Sequence[int] | None = None,
) -> dict[str, Any]:
    if protocol_hash is None:
        _, protocol_hash = _require_lock(config, run_root)
    official_source, target = official_fold_subjects(fold)
    source = list(source_override) if source_override is not None else official_source
    selected_subjects = list(range(SUBJECTS)) if subjects is None else sorted(set(map(int, subjects)))
    state = _fit_fold_state(config, run_root, fold, protocol_hash, source)
    root = _fold_cache_root(run_root, fold, protocol_hash)
    for index, subject in enumerate(selected_subjects, 1):
        output = root / "subjects" / f"sub{subject:03d}.npz"
        if _valid_feature_subject(output):
            continue
        spectra = _load_spectra(config, run_root, subject)
        flattened = [values.reshape(-1, EEG_CHANNELS, values.shape[-1]) for values in spectra]
        sqrt_jsd = transform_native_sqrt_jsd(flattened, state["arithmetic"])
        fisher = transform_native_fisher_rao_pca(
            flattened, state["fisher"], state["means"], state["components"]
        )
        write_npz(
            output,
            native_sqrt_jsd=sqrt_jsd.reshape(VIDEOS, 30, EEG_CHANNELS * 5),
            native_fisher_rao_pca=fisher.reshape(VIDEOS, 30, EEG_CHANNELS * 5),
        )
        if index % 12 == 0 or index == len(selected_subjects):
            LOGGER.info("Fold %02d feature pass %d/%d subjects", fold, index, len(selected_subjects))

    complete = [
        subject for subject in selected_subjects
        if _valid_feature_subject(root / "subjects" / f"sub{subject:03d}.npz")
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fold": fold,
        "protocol_hash": protocol_hash,
        "source_state_subjects": source,
        "official_target_subjects": target,
        "subjects_requested": selected_subjects,
        "subjects_complete": complete,
        "representations": list(CONDITIONS.values()),
        "shape_per_subject": [VIDEOS, 30, EEG_CHANNELS * 5],
        "target_used_to_fit_state": False,
        "updated_at": utc_now(),
    }
    write_json(root / "feature_manifest.json", manifest)
    return manifest


def _load_samples(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol_hash: str,
    subjects: Iterable[int],
    representation: str,
) -> list[TrialSample]:
    root = _fold_cache_root(run_root, fold, protocol_hash) / "subjects"
    samples: list[TrialSample] = []
    for subject in subjects:
        if representation == "de":
            values = _load_de(config, run_root, int(subject))
        else:
            path = root / f"sub{int(subject):03d}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing fold feature {path}; run PrepareFold first")
            with np.load(path, allow_pickle=False) as archive:
                values = np.asarray(archive[representation], dtype=np.float32)
        for video in range(VIDEOS):
            samples.append(TrialSample(
                np.ascontiguousarray(values[video]),
                int(VIDEO_LABELS[video]),
                int(subject),
                1,
                video + 1,
                int(subject) * VIDEOS + video,
            ))
    return samples


def _model_config(config: ExperimentConfig) -> dict[str, Any]:
    architecture = experiment_settings(config)["architecture"]
    return {
        "name": "hierarchical_attention",
        "channels": EEG_CHANNELS,
        "d_model": int(architecture["d_model"]),
        "heads": int(architecture["heads"]),
        "layers": int(architecture["layers"]),
        "feedforward": int(architecture["feedforward"]),
        "dropout": float(architecture["dropout"]),
        "architecture_label": "base",
    }


def _training_config(config: ExperimentConfig, epochs: int | None = None) -> dict[str, Any]:
    training = copy.deepcopy(config.raw["training"])
    training["locked_epochs"] = int(epochs or training["epochs"])
    training["checkpoint_selection"] = "fixed_final_epoch_only"
    training["target_monitoring"] = False
    return training


def _task_output(run_root: Path, condition: str, fold: int) -> Path:
    return run_root / "faced" / condition / f"fold-{fold:02d}" / f"seed-{SEED}"


def run_task(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    condition: str,
    protocol_hash: str,
    *,
    epochs: int | None = None,
    source_override: Sequence[int] | None = None,
    target_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown FACED condition: {condition}")
    started = time.perf_counter()
    source, target = official_fold_subjects(fold)
    if source_override is not None:
        source = list(source_override)
    if target_override is not None:
        target = list(target_override)
    representation = CONDITIONS[condition]
    output = _task_output(run_root, condition, fold)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "fixed_final_epoch_model.pt"
    training = _training_config(config, epochs)
    device = select_device(str(training.get("device", "auto")))

    # Target feature arrays are deliberately not loaded until source training
    # and the single fixed-final-epoch checkpoint are complete.
    source_samples = _load_samples(config, run_root, fold, protocol_hash, source, representation)
    fit_locked_source_model(
        source_samples,
        _model_config(config),
        training,
        len(EMOTION_NAMES),
        device,
        checkpoint,
        seed=SEED,
        scale_inputs=True,
        context={
            "family": FAMILY,
            "fold": fold,
            "condition": condition,
            "protocol_hash": protocol_hash,
            "target_loaded_during_training": False,
        },
    )
    del source_samples
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_samples = _load_samples(config, run_root, fold, protocol_hash, target, representation)
    metrics, predictions = evaluate_locked_checkpoint(checkpoint, target_samples, device)
    _write_csv(output / "predictions.csv", predictions)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parameter_count = int(sum(value.numel() for value in checkpoint_payload["model_state_dict"].values()))
    result = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "dataset": "faced",
        "condition": condition,
        "representation": representation,
        "architecture": "HierarchicalChannelBandTransformer Base",
        "fold": fold,
        "seed": SEED,
        "protocol_hash": protocol_hash,
        "source_subjects": source,
        "target_subjects": target,
        "source_subject_count": len(source),
        "target_subject_count": len(target),
        "epochs": int(training["locked_epochs"]),
        "effective_batch_size": int(training["batch_size"]),
        "parameter_count": parameter_count,
        "target_used_during_training": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "source_only_feature_state": True,
        "source_zscore": True,
        "final_target_test": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
        "diagnostic_smoke": source_override is not None or target_override is not None,
    }
    write_json(output / "result.json", result)
    write_json(output / "COMPLETE.json", {
        "protocol_hash": protocol_hash,
        "condition": condition,
        "fold": fold,
        "completed_at": result["completed_at"],
    })
    write_json(output / "protocol_audit.json", {
        "source_feature_state_subjects": source,
        "target_subjects": target,
        "target_loaded_after_training": True,
        "target_loaded_during_training": False,
        "early_stopping": False,
        "checkpoint_selection": "fixed_final_epoch_only",
    })
    return result


def _declared_tasks(folds: Sequence[int], conditions: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"faced__{condition}__fold-{fold:02d}__seed-{SEED}",
            "fold": fold,
            "condition": condition,
            "status": "pending",
            "attempts": 0,
        }
        for fold in folds for condition in conditions
    ]


def run_matrix(
    config: ExperimentConfig,
    run_root: Path,
    folds: Sequence[int] | None,
    conditions: Sequence[str] | None,
    *,
    resume: bool = False,
    retry_failed: bool = False,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    _, protocol_hash = _require_lock(config, run_root)
    selected_folds = list(folds or range(1, FOLDS + 1))
    selected_conditions = list(conditions or CONDITIONS)
    if any(fold < 1 or fold > FOLDS for fold in selected_folds):
        raise ValueError("Fold filter must be between 1 and 10")
    if set(selected_conditions) - set(CONDITIONS):
        raise ValueError("Unknown condition filter")
    tasks = _declared_tasks(selected_folds, selected_conditions)
    manifest_path = run_root / "matrix_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "protocol_hash": protocol_hash,
        "tasks": {},
        "created_at": utc_now(),
    }
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("Existing matrix manifest uses another FACED protocol")
    for task in tasks:
        manifest["tasks"].setdefault(task["task_id"], task)
    write_json(manifest_path, manifest)

    runnable: list[dict[str, Any]] = []
    for declared in tasks:
        task = manifest["tasks"][declared["task_id"]]
        complete = (_task_output(run_root, task["condition"], task["fold"]) / "COMPLETE.json").is_file()
        if complete or task.get("status") == "complete":
            task["status"] = "complete"
            if not resume:
                raise FileExistsError(f"Task already complete; use --resume: {task['task_id']}")
            continue
        if task.get("status") == "failed" and not retry_failed:
            continue
        runnable.append(task)
    if max_tasks is not None:
        runnable = runnable[:max_tasks]

    for task in runnable:
        task["status"] = "running"
        task["attempts"] = int(task.get("attempts", 0)) + 1
        task["started_at"] = utc_now()
        write_json(manifest_path, manifest)
        try:
            if CONDITIONS[str(task["condition"])] != "de":
                prepare_fold_features(config, run_root, int(task["fold"]), protocol_hash)
            result = run_task(
                config, run_root, int(task["fold"]), str(task["condition"]), protocol_hash
            )
            task["status"] = "complete"
            task["completed_at"] = result["completed_at"]
            task["result_path"] = str(
                (_task_output(run_root, task["condition"], task["fold"]) / "result.json").relative_to(run_root)
            )
        except BaseException as exc:
            task["status"] = "failed"
            task["failed_at"] = utc_now()
            task["error"] = f"{type(exc).__name__}: {exc}"
            write_json(
                _task_output(run_root, task["condition"], task["fold"]) / "failure.json",
                {"error": task["error"], "traceback": traceback.format_exc()},
            )
            LOGGER.exception("FACED task failed: %s", task["task_id"])
        finally:
            write_json(manifest_path, manifest)
    return matrix_status(run_root)


def run_smoke(config: ExperimentConfig, run_root: Path, epochs: int = 2) -> dict[str, Any]:
    experiment_settings(config)
    smoke_root = Path(f"{run_root}_smoke")
    source = [12, 13]
    target = [0]
    prepare_spectra(config, smoke_root, [*source, *target])
    audit = validate_sources(config)
    protocol_hash = "smoke-" + _json_hash(protocol_payload(config, audit), 10)
    prepare_fold_features(
        config,
        smoke_root,
        1,
        protocol_hash,
        source_override=source,
        subjects=[*source, *target],
    )
    return run_task(
        config,
        smoke_root,
        1,
        "native_sqrt_jsd_base",
        protocol_hash,
        epochs=epochs,
        source_override=source,
        target_override=target,
    )


def matrix_status(run_root: Path) -> dict[str, Any]:
    path = run_root / "matrix_manifest.json"
    if not path.is_file():
        return {"status": "not_started", "run_root": str(run_root.resolve())}
    manifest = read_json(path)
    tasks = list(manifest.get("tasks", {}).values())
    counts = {status: sum(task.get("status") == status for task in tasks) for status in ("pending", "running", "complete", "failed")}
    failed = [{"task_id": task["task_id"], "error": task.get("error")} for task in tasks if task.get("status") == "failed"]
    _write_csv(run_root / "failed_tasks.csv", failed)
    return {
        "status": "complete" if len(tasks) == 30 and counts["complete"] == 30 else "in_progress",
        "declared": len(tasks),
        **counts,
        "failed_tasks": failed,
        "run_root": str(run_root.resolve()),
    }


def summarize(run_root: Path, *, allow_partial: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "faced").glob("*/fold-*/seed-42/result.json")):
        result = read_json(path)
        if result.get("diagnostic_smoke"):
            continue
        metrics = result["final_target_test"]
        rows.append({
            "condition": result["condition"],
            "fold": int(result["fold"]),
            "target_subjects": json.dumps(result["target_subjects"]),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        })
    if not allow_partial and len(rows) != 30:
        raise RuntimeError(f"Strict FACED summary requires 30 completed tasks, found {len(rows)}")
    summaries = []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        summaries.append({
            "condition": condition,
            "folds": len(subset),
            **{
                f"{metric}_{stat}": float(getattr(np, stat)([row[metric] for row in subset]))
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                for stat in ("mean", "std")
            },
        })
    _write_csv(run_root / "fold_results.csv", rows)
    _write_csv(run_root / "condition_summary.csv", summaries)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete" if len(rows) == 30 else "partial",
        "completed_tasks": len(rows),
        "expected_tasks": 30,
        "conditions": summaries,
        "updated_at": utc_now(),
    }
    write_json(run_root / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FACED Processed_data DE/sqrt-JSD/Fisher-Rao experiments")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--config", default="configs/faced/native_compact_base.yaml")
        child.add_argument("--run-root")
        return child

    validate = configured("validate-data")
    validate.add_argument("--deep", action="store_true")
    spectra = configured("prepare-spectra")
    spectra.add_argument("--subject", type=int, action="append")
    configured("lock")
    features = configured("prepare-features")
    features.add_argument("--fold", type=int, action="append")
    smoke = configured("smoke")
    smoke.add_argument("--smoke-epochs", type=int, default=2)
    matrix = configured("matrix")
    matrix.add_argument("--fold", type=int, action="append")
    matrix.add_argument("--condition", choices=tuple(CONDITIONS), action="append")
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--retry-failed", action="store_true")
    matrix.add_argument("--max-tasks", type=int)
    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run-root", required=True)
    summary.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.command == "status":
        payload = matrix_status(Path(args.run_root).expanduser().resolve())
    elif args.command == "summarize":
        payload = summarize(Path(args.run_root).expanduser().resolve(), allow_partial=args.allow_partial)
    else:
        config = load_config(args.config)
        run_root = Path(args.run_root).expanduser().resolve() if args.run_root else config.run_root
        if args.command == "validate-data":
            payload = validate_sources(config, deep=args.deep)
        elif args.command == "prepare-spectra":
            payload = prepare_spectra(config, run_root, args.subject)
        elif args.command == "lock":
            payload = lock_experiment(config, run_root)
        elif args.command == "prepare-features":
            _, protocol_hash = _require_lock(config, run_root)
            folds = args.fold or list(range(1, FOLDS + 1))
            payload = {
                f"fold-{fold:02d}": prepare_fold_features(config, run_root, fold, protocol_hash)
                for fold in folds
            }
        elif args.command == "smoke":
            payload = run_smoke(config, run_root, args.smoke_epochs)
        else:
            payload = run_matrix(
                config,
                run_root,
                args.fold,
                args.condition,
                resume=args.resume,
                retry_failed=args.retry_failed,
                max_tasks=args.max_tasks,
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
