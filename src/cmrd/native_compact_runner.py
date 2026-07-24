from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cmrd.config import ExperimentConfig, load_config
from cmrd.data.records import TrialSample
from cmrd.features.rd import (
    extract_native_spectral_distributions,
    fisher_rao_log_map,
    native_frequency_grid,
    normalize_histograms,
    transform_native_fisher_rao_pca,
    transform_native_sqrt_jsd,
    transform_native_wasserstein1,
)
from cmrd.fixed_protocol import (
    FIXED_SEED,
    evaluate_locked_checkpoint,
    fit_exploratory_monitored_source_model,
    scaling_statistics,
)
from cmrd.fixed_protocol_runner import validate_fixed_cache
from cmrd.io import read_json, write_json, write_npz
from cmrd.training.runtime import select_device


LOGGER = logging.getLogger("cmrd.native_compact")
SCHEMA_VERSION = 1
FAMILY = "Native-Compact-v1"
FOLD = 1
EXPECTED_CONDITIONS = (
    "a_native_sqrt_jsd_base_v2",
    "b_native_fisher_rao_pca_base_v2",
    "c_native_wasserstein1_base_v2",
)
DEAP_EXPECTED_CONDITIONS = EXPECTED_CONDITIONS[:2]
CONDITION_REPRESENTATIONS = {
    "a_native_sqrt_jsd_base_v2": "native_sqrt_jsd_zscore",
    "b_native_fisher_rao_pca_base_v2": "native_fisher_rao_pca_zscore",
    "c_native_wasserstein1_base_v2": "native_wasserstein1_zscore",
}
REPRESENTATION_KEYS = {
    "native_sqrt_jsd_zscore": "native_sqrt_jsd",
    "native_fisher_rao_pca_zscore": "native_fisher_rao_pca",
    "native_wasserstein1_zscore": "native_wasserstein1",
}


def _expected_conditions(config: ExperimentConfig) -> tuple[str, ...]:
    return DEAP_EXPECTED_CONDITIONS if config.dataset == "deap" else EXPECTED_CONDITIONS


def _feature_dim(config: ExperimentConfig) -> int:
    return int(config.raw["dataset"]["channels"]) * len(config.raw["signal"]["bands_hz"])


def _active_representation_keys(config: ExperimentConfig) -> dict[str, str]:
    representations = {
        CONDITION_REPRESENTATIONS[condition]
        for condition in _expected_conditions(config)
    }
    return {
        representation: key
        for representation, key in REPRESENTATION_KEYS.items()
        if representation in representations
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
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


def experiment_settings(config: ExperimentConfig) -> dict[str, Any]:
    raw = copy.deepcopy(config.raw.get("native_compact", {}))
    settings = {
        "seed": int(raw.get("seed", FIXED_SEED)),
        "fold": int(raw.get("fold", FOLD)),
        "max_epochs": int(raw.get("max_epochs", 200)),
        "target_monitor_interval": int(raw.get("target_monitor_interval", 10)),
        "pca_max_windows_per_trial": int(raw.get("pca_max_windows_per_trial", 32)),
        "reference_scope": str(raw.get(
            "reference_scope", "source_train" if config.dataset == "deap" else "all_source"
        )),
        "conditions": copy.deepcopy(raw.get("conditions", {})),
        "architecture": copy.deepcopy(raw.get("architecture", {})),
    }
    if settings["seed"] != FIXED_SEED or settings["fold"] != FOLD:
        raise ValueError("Native-Compact-v1 is frozen to seed=42 and fold=1")
    if settings["max_epochs"] != 200 or settings["target_monitor_interval"] != 10:
        raise ValueError("Native-Compact-v1 requires v2 max_epochs=200 and 10-epoch monitoring")
    if settings["pca_max_windows_per_trial"] < 2:
        raise ValueError("pca_max_windows_per_trial must be at least 2")
    expected_conditions = _expected_conditions(config)
    if tuple(settings["conditions"]) != expected_conditions:
        raise ValueError(f"Conditions must be declared in this order: {expected_conditions}")
    if settings["reference_scope"] not in {"all_source", "source_train"}:
        raise ValueError("native_compact.reference_scope must be all_source or source_train")
    if config.dataset == "deap" and settings["reference_scope"] != "source_train":
        raise ValueError("DEAP native features must fit reference/PCA on source_train only")
    for condition in expected_conditions:
        representation = CONDITION_REPRESENTATIONS[condition]
        definition = settings["conditions"][condition]
        if str(definition.get("representation")) != representation:
            raise ValueError(f"Condition {condition} must use {representation}")

    architecture = settings["architecture"]
    expected_architecture = {
        "d_model": 128,
        "heads": 4,
        "layers": 3,
        "feedforward": 512,
        "dropout": 0.15,
    }
    for key, expected in expected_architecture.items():
        if float(architecture.get(key, float("nan"))) != float(expected):
            raise ValueError(f"Base architecture requires {key}={expected}")

    training = config.raw["training"]
    v2_expected = {
        "learning_rate": 2e-4,
        "minimum_learning_rate": 1e-6,
        "warmup_fraction": 0.1,
        "weight_decay": 0.01,
        "label_smoothing": 0.0,
        "gradient_accumulation_steps": 1,
    }
    for key, expected in v2_expected.items():
        if float(training.get(key, float("nan"))) != float(expected):
            raise ValueError(f"v2 training requires {key}={expected}")
    v2_integer_expected = {
        "epochs": 200,
        "batch_size": 16,
        "num_workers": 2,
        "prefetch_factor": 1,
    }
    for key, expected in v2_integer_expected.items():
        if int(training.get(key, -1)) != expected:
            raise ValueError(f"v2 training requires {key}={expected}")
    if float(training.get("gradient_clip_norm", float("nan"))) != 1.0:
        raise ValueError("v2 training requires gradient_clip_norm=1.0")
    if str(training.get("precision")) != "bfloat16":
        raise ValueError("v2 training requires precision=bfloat16")
    if str(training.get("matmul_precision")) != "high":
        raise ValueError("v2 training requires matmul_precision=high")
    if training.get("deterministic") is not True:
        raise ValueError("v2 training requires deterministic=true")
    if training.get("persistent_workers") is not True:
        raise ValueError("v2 training requires persistent_workers=true")
    if training.get("pin_memory") is not True:
        raise ValueError("v2 training requires pin_memory=true")
    if [int(value) for value in training.get("seeds", [])] != [FIXED_SEED]:
        raise ValueError(f"v2 pilot requires training.seeds=[{FIXED_SEED}]")
    return settings


def _fold_groups(
    cache_root: Path,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(cache_root / "folds" / "fold-01" / "manifest.json")
    train = list(manifest["groups"]["train"])
    validation = list(manifest["groups"]["validation"])
    target = list(manifest["groups"]["test"])
    source = train + validation
    source_subjects = {int(entry["subject"]) for entry in source}
    train_subjects = {int(entry["subject"]) for entry in train}
    validation_subjects = {int(entry["subject"]) for entry in validation}
    target_subjects = {int(entry["subject"]) for entry in target}
    subjects = int(config.raw["dataset"]["subjects"])
    validation_count = int(config.raw["split"]["validation_subjects"])
    expected_subjects = set(range(1, subjects + 1))
    if (
        len(source_subjects) != subjects - 1
        or len(train_subjects) != subjects - 1 - validation_count
        or len(validation_subjects) != validation_count
        or train_subjects & validation_subjects
        or source_subjects != expected_subjects - {FOLD}
        or target_subjects != {FOLD}
    ):
        raise ValueError(
            f"Native-Compact-v1 requires a valid fold-1 {subjects - 1}-source/1-target split"
        )
    return train, validation, target


def _fold_entries(
    cache_root: Path,
    config: ExperimentConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, validation, target = _fold_groups(cache_root, config)
    source = train + validation
    return source, target


def _validate_deap_cache(
    config: ExperimentConfig,
    cache_parent: Path | None,
) -> dict[str, Any]:
    parent = (
        cache_parent
        or config.processed_root / "deap" / "de_rjsd_ica_1s_hop1"
    ).resolve()
    if (parent / "pipeline_manifest.json").is_file():
        candidates = [parent]
    elif parent.is_dir():
        candidates = sorted(
            (path for path in parent.iterdir() if (path / "pipeline_manifest.json").is_file()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    else:
        candidates = []
    cache_root: Path | None = None
    pipeline: dict[str, Any] | None = None
    for candidate in candidates:
        value = read_json(candidate / "pipeline_manifest.json")
        if (
            value.get("dataset") == "DEAP"
            and value.get("all_folds_complete") is True
            and int(value.get("expected_folds", 0)) == 32
        ):
            cache_root = candidate
            pipeline = value
            break
    if cache_root is None or pipeline is None:
        raise FileNotFoundError(f"No complete DEAP cache under {parent}")

    validation_path = cache_root / "validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(f"Missing DEAP deep validation result: {validation_path}")
    validation = read_json(validation_path)
    required = {
        "status": "valid",
        "deep": True,
        "subjects": 32,
        "trials": 1280,
        "folds_checked": 32,
        "strict_ica_detection_error_trials": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": validation.get(key)}
        for key, expected in required.items()
        if validation.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"DEAP cache failed its required deep audit: {mismatches}")
    train, validation_entries, target = _fold_groups(cache_root, config)
    signature = str(pipeline["preprocessing_signature"])
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.dataset,
        "cache_root": str(cache_root),
        "preprocessing_signature": signature,
        "outer_protocol": "32-fold LOSO: 31 source subjects / 1 target subject",
        "cache_diagnostic_split": "29 source-train + 2 source-validation subjects",
        "fold": FOLD,
        "source_train_trials": len(train),
        "source_validation_trials": len(validation_entries),
        "target_trials": len(target),
        "deep_validation": str(validation_path),
        "validated_at": utc_now(),
    }


def validate_native_sources(
    config: ExperimentConfig,
    cache_parent: Path | None = None,
) -> dict[str, Any]:
    audit = (
        _validate_deap_cache(config, cache_parent)
        if config.dataset == "deap"
        else validate_fixed_cache(config, cache_parent)
    )
    cache_root = Path(audit["cache_root"])
    environment = read_json(cache_root / "environment.json")
    ica_cache_root = Path(str(environment["ica_cache_root"])).resolve()
    cleaning_signature = str(environment["cleaning_signature"])
    cache_manifest = read_json(ica_cache_root / "cache_manifest.json")
    if str(cache_manifest.get("cleaning_signature")) != cleaning_signature:
        raise ValueError("ICA-cleaned cache signature does not match the fixed feature cache")
    source, target = _fold_entries(cache_root, config)
    for entry in source + target:
        cleaned = ica_cache_root / "trials" / f"{entry['trial_id']}.npz"
        if not cleaned.is_file():
            raise FileNotFoundError(f"Missing ICA-cleaned trial: {cleaned}")
    audit.update({
        "family": FAMILY,
        "ica_cache_root": str(ica_cache_root),
        "cleaning_signature": cleaning_signature,
        "fold": FOLD,
        "source_trials": len(source),
        "target_trials": len(target),
        "native_nfft": int(round(float(config.raw["signal"]["target_rate"]) * float(config.raw["signal"]["window_seconds"]))),
    })
    return audit


def protocol_payload(config: ExperimentConfig, audit: dict[str, Any]) -> dict[str, Any]:
    settings = experiment_settings(config)
    bands = config.raw["signal"]["bands_hz"]
    channels = int(config.raw["dataset"]["channels"])
    subjects = int(config.raw["dataset"]["subjects"])
    feature_dim = _feature_dim(config)
    grid = native_frequency_grid(
        float(config.raw["signal"]["target_rate"]),
        float(config.raw["signal"]["window_seconds"]),
        bands,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "dataset": config.dataset,
        "seed": FIXED_SEED,
        "fold": FOLD,
        "outer_protocol": (
            f"fold-1 LOSO: all {subjects - 1} non-target subjects train / "
            "subject 1 target-monitored"
        ),
        "window_seconds": float(config.raw["signal"]["window_seconds"]),
        "hop_seconds": float(config.raw["signal"]["hop_seconds"]),
        "spectral_estimator": (
            "Hann modified periodogram; "
            f"nfft=nperseg={int(round(float(config.raw['signal']['target_rate']) * float(config.raw['signal']['window_seconds'])))}; "
            "no zero padding"
        ),
        "native_frequencies_hz": {
            name: values.astype(float).tolist()
            for name, values in zip(bands, grid, strict=True)
        },
        "reference_rules": {
            "sqrt_jsd_and_wasserstein1": "source-only pooled arithmetic probability mean",
            "fisher_rao": "source-only normalized mean square-root embedding (extrinsic Hellinger barycenter)",
            "fisher_rao_pca": "per-channel-band first PC of centered source-only tangent vectors",
        },
        "fisher_rao_embedding": "p -> 2*sqrt(p); log-map norm equals 2*arccos(Bhattacharyya coefficient)",
        "pca_max_windows_per_trial": settings["pca_max_windows_per_trial"],
        "reference_fit_scope": settings["reference_scope"],
        "representations": {
            condition: CONDITION_REPRESENTATIONS[condition]
            for condition in _expected_conditions(config)
        },
        "input_shape": (
            f"[T,{channels},{len(bands)}] flattened to [T,{feature_dim}]"
        ),
        "source_zscore": True,
        "architecture": settings["architecture"],
        "training": config.raw["training"],
        "max_epochs": settings["max_epochs"],
        "target_monitor_interval": settings["target_monitor_interval"],
        "target_metrics_affect_training": False,
        "checkpoint_selection": "predeclared final epoch only",
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "preprocessing_signature": audit["preprocessing_signature"],
        "cleaning_signature": audit["cleaning_signature"],
    }


def lock_experiment(config: ExperimentConfig, run_root: Path, cache_parent: Path | None) -> dict[str, Any]:
    audit = validate_native_sources(config, cache_parent)
    protocol = protocol_payload(config, audit)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "dataset": config.dataset,
        "fold": FOLD,
        "protocol_hash": _json_hash(protocol),
        "max_epochs": int(protocol["max_epochs"]),
        "target_monitor_interval": int(protocol["target_monitor_interval"]),
        "target_metrics_affect_training": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "locked_at": utc_now(),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / f"experiment_lock_{config.dataset}.json"
    if path.is_file():
        existing = read_json(path)
        left = {key: value for key, value in existing.items() if key != "locked_at"}
        right = {key: value for key, value in payload.items() if key != "locked_at"}
        if left != right:
            raise ValueError(f"Existing experiment lock conflicts with active config: {path}")
        return existing
    write_json(path, payload)
    return payload


def _require_lock(config: ExperimentConfig, run_root: Path, protocol_hash: str) -> None:
    path = run_root / f"experiment_lock_{config.dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run the Lock stage first: {path}")
    lock = read_json(path)
    if lock.get("protocol_hash") != protocol_hash:
        raise ValueError(f"Experiment lock does not match the active protocol: {path}")
    if lock.get("target_metrics_affect_training") is not False:
        raise ValueError("The lock must forbid target-driven training decisions")


def _native_distributions(
    config: ExperimentConfig,
    audit: dict[str, Any],
    entry: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    path = Path(audit["ica_cache_root"]) / "trials" / f"{entry['trial_id']}.npz"
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["cleaning_signature"].item()) != str(audit["cleaning_signature"]):
            raise ValueError(f"Cleaning signature mismatch: {path}")
        if int(archive["subject"].item()) != int(entry["subject"]):
            raise ValueError(f"Subject metadata mismatch: {path}")
        signal = np.asarray(archive["cleaned"], dtype=np.float32)
    return extract_native_spectral_distributions(
        signal,
        float(config.raw["signal"]["target_rate"]),
        float(config.raw["signal"]["window_seconds"]),
        float(config.raw["signal"]["hop_seconds"]),
        config.raw["signal"]["bands_hz"],
    )


def _feature_cache_root(run_root: Path, dataset: str, protocol_hash: str) -> Path:
    return run_root / "feature_cache" / dataset / "fold-01" / protocol_hash


def _load_reference_state(path: Path, band_names: Sequence[str]) -> dict[str, list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "frequencies": [np.asarray(archive[f"frequency_{name}"], dtype=np.float32) for name in band_names],
            "arithmetic": [np.asarray(archive[f"arithmetic_reference_{name}"], dtype=np.float32) for name in band_names],
            "fisher_rao": [np.asarray(archive[f"fisher_rao_reference_{name}"], dtype=np.float32) for name in band_names],
        }


def _fit_reference_state(
    config: ExperimentConfig,
    audit: dict[str, Any],
    source_entries: Sequence[dict[str, Any]],
    cache_root: Path,
    protocol_hash: str,
) -> dict[str, list[np.ndarray]]:
    band_names = list(config.raw["signal"]["bands_hz"])
    state_path = cache_root / "reference_state.npz"
    metadata_path = cache_root / "reference_state.json"
    if state_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("protocol_hash") == protocol_hash and metadata.get("status") == "complete":
            return _load_reference_state(state_path, band_names)

    frequencies = native_frequency_grid(
        float(config.raw["signal"]["target_rate"]),
        float(config.raw["signal"]["window_seconds"]),
        config.raw["signal"]["bands_hz"],
    )
    channels = int(config.raw["dataset"]["channels"])
    probability_sums = [np.zeros((channels, grid.size), dtype=np.float64) for grid in frequencies]
    square_root_sums = [np.zeros((channels, grid.size), dtype=np.float64) for grid in frequencies]
    windows = 0
    started = time.perf_counter()
    for index, entry in enumerate(source_entries, 1):
        distributions, actual = _native_distributions(config, audit, entry)
        for band, (distribution, expected, observed) in enumerate(
            zip(distributions, frequencies, actual, strict=True)
        ):
            if not np.array_equal(expected, observed):
                raise RuntimeError("Native frequency grid changed while fitting references")
            probability_sums[band] += distribution.sum(axis=0, dtype=np.float64)
            square_root_sums[band] += np.sqrt(distribution).sum(axis=0, dtype=np.float64)
        windows += int(distributions[0].shape[0])
        if index % 25 == 0 or index == len(source_entries):
            LOGGER.info("%s reference pass %d/%d trials", config.dataset, index, len(source_entries))

    arithmetic = [normalize_histograms(total / windows).astype(np.float32) for total in probability_sums]
    fisher_rao = []
    for total in square_root_sums:
        root = total / windows
        norm = np.linalg.norm(root, axis=-1, keepdims=True)
        if np.any(norm <= 0):
            raise FloatingPointError("Invalid source Hellinger barycenter norm")
        root = root / norm
        fisher_rao.append(normalize_histograms(np.square(root)).astype(np.float32))
    arrays: dict[str, Any] = {}
    for name, grid, arithmetic_reference, fisher_reference in zip(
        band_names, frequencies, arithmetic, fisher_rao, strict=True
    ):
        arrays[f"frequency_{name}"] = grid
        arrays[f"arithmetic_reference_{name}"] = arithmetic_reference
        arrays[f"fisher_rao_reference_{name}"] = fisher_reference
    write_npz(state_path, **arrays)
    write_json(metadata_path, {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "dataset": config.dataset,
        "fold": FOLD,
        "protocol_hash": protocol_hash,
        "source_subjects": sorted({int(entry["subject"]) for entry in source_entries}),
        "source_trials": len(source_entries),
        "source_windows": windows,
        "arithmetic_reference": "mean of normalized source-window PSD distributions",
        "fisher_rao_reference": "normalized mean square-root embedding, squared back to simplex",
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
    })
    return {"frequencies": frequencies, "arithmetic": arithmetic, "fisher_rao": fisher_rao}


def _load_pca_state(path: Path, band_names: Sequence[str]) -> dict[str, list[np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "means": [np.asarray(archive[f"tangent_mean_{name}"], dtype=np.float32) for name in band_names],
            "components": [np.asarray(archive[f"component_{name}"], dtype=np.float32) for name in band_names],
            "explained_variance_ratio": [
                np.asarray(archive[f"explained_variance_ratio_{name}"], dtype=np.float32)
                for name in band_names
            ],
        }


def _fit_pca_state(
    config: ExperimentConfig,
    audit: dict[str, Any],
    source_entries: Sequence[dict[str, Any]],
    references: dict[str, list[np.ndarray]],
    cache_root: Path,
    protocol_hash: str,
) -> dict[str, list[np.ndarray]]:
    band_names = list(config.raw["signal"]["bands_hz"])
    state_path = cache_root / "fisher_rao_pca_state.npz"
    metadata_path = cache_root / "fisher_rao_pca_state.json"
    if state_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("protocol_hash") == protocol_hash and metadata.get("status") == "complete":
            return _load_pca_state(state_path, band_names)

    settings = experiment_settings(config)
    channels = int(config.raw["dataset"]["channels"])
    sums = [np.zeros((channels, reference.shape[-1]), dtype=np.float64) for reference in references["fisher_rao"]]
    cross = [
        np.zeros((channels, reference.shape[-1], reference.shape[-1]), dtype=np.float64)
        for reference in references["fisher_rao"]
    ]
    observations = 0
    started = time.perf_counter()
    limit = settings["pca_max_windows_per_trial"]
    for index, entry in enumerate(source_entries, 1):
        distributions, _ = _native_distributions(config, audit, entry)
        available = int(distributions[0].shape[0])
        selected = np.unique(np.linspace(0, available - 1, min(available, limit), dtype=np.int64))
        for band, (distribution, reference) in enumerate(
            zip(distributions, references["fisher_rao"], strict=True)
        ):
            tangent = fisher_rao_log_map(distribution[selected], reference)
            sums[band] += tangent.sum(axis=0, dtype=np.float64)
            cross[band] += np.einsum("tcf,tcg->cfg", tangent, tangent, optimize=True, dtype=np.float64)
        observations += int(selected.size)
        if index % 25 == 0 or index == len(source_entries):
            LOGGER.info("%s Fisher-Rao PCA pass %d/%d trials", config.dataset, index, len(source_entries))
    if observations < 2:
        raise ValueError("Fisher-Rao PCA requires at least two sampled source windows")

    means: list[np.ndarray] = []
    components: list[np.ndarray] = []
    ratios: list[np.ndarray] = []
    arrays: dict[str, Any] = {}
    for name, total, second, reference in zip(
        band_names, sums, cross, references["fisher_rao"], strict=True
    ):
        mean = total / observations
        covariance = second / observations - np.einsum("cf,cg->cfg", mean, mean)
        covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        component = eigenvectors[..., -1]
        root_reference = np.sqrt(reference)
        component -= (component * root_reference).sum(axis=-1, keepdims=True) * root_reference
        norm = np.linalg.norm(component, axis=-1, keepdims=True)
        if np.any(norm <= 1e-10):
            raise FloatingPointError(f"Degenerate Fisher-Rao first component in {name}")
        component /= norm
        pivot = np.argmax(np.abs(component), axis=-1)
        signs = np.take_along_axis(component, pivot[:, None], axis=-1)
        component *= np.where(signs < 0.0, -1.0, 1.0)
        denominator = eigenvalues.sum(axis=-1)
        ratio = np.divide(
            eigenvalues[..., -1], denominator,
            out=np.zeros_like(denominator), where=denominator > 0,
        )
        means.append(np.asarray(mean, dtype=np.float32))
        components.append(np.asarray(component, dtype=np.float32))
        ratios.append(np.asarray(ratio, dtype=np.float32))
        arrays[f"tangent_mean_{name}"] = means[-1]
        arrays[f"component_{name}"] = components[-1]
        arrays[f"explained_variance_ratio_{name}"] = ratios[-1]
    write_npz(state_path, **arrays)
    write_json(metadata_path, {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "dataset": config.dataset,
        "fold": FOLD,
        "protocol_hash": protocol_hash,
        "fit_scope": "source-only per-channel-band PCA in Fisher-Rao tangent space",
        "source_trials": len(source_entries),
        "sampled_windows_per_channel_band": observations,
        "max_windows_per_trial": limit,
        "component_orientation": "largest-absolute loading is positive",
        "mean_explained_variance_ratio_by_band": {
            name: float(np.mean(ratio)) for name, ratio in zip(band_names, ratios, strict=True)
        },
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
    })
    return {"means": means, "components": components, "explained_variance_ratio": ratios}


def _valid_compact_trial(
    path: Path,
    entry: dict[str, Any],
    representation_keys: dict[str, str],
    feature_dim: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                all(key in archive for key in representation_keys.values())
                and all(
                    archive[key].ndim == 2 and archive[key].shape[1] == feature_dim
                    for key in representation_keys.values()
                )
                and int(archive["source_index"].item()) == int(entry["source_index"])
            )
    except (KeyError, OSError, ValueError):
        return False


def prepare_native_feature_cache(
    config: ExperimentConfig,
    run_root: Path,
    cache_parent: Path | None = None,
) -> dict[str, Any]:
    audit = validate_native_sources(config, cache_parent)
    protocol = protocol_payload(config, audit)
    protocol_hash = _json_hash(protocol)
    cache_root = _feature_cache_root(run_root, config.dataset, protocol_hash)
    feature_dim = _feature_dim(config)
    representation_keys = _active_representation_keys(config)
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete" and manifest.get("protocol_hash") == protocol_hash:
            return manifest

    train_entries, validation_entries, target_entries = _fold_groups(
        Path(audit["cache_root"]), config
    )
    source_entries = train_entries + validation_entries
    reference_entries = (
        train_entries
        if experiment_settings(config)["reference_scope"] == "source_train"
        else source_entries
    )
    references = _fit_reference_state(config, audit, reference_entries, cache_root, protocol_hash)
    pca = (
        _fit_pca_state(config, audit, reference_entries, references, cache_root, protocol_hash)
        if "native_fisher_rao_pca_zscore" in representation_keys
        else None
    )
    entries = sorted(source_entries + target_entries, key=lambda item: int(item["source_index"]))
    trial_root = cache_root / "trials"
    started = time.perf_counter()
    completed = 0
    for index, entry in enumerate(entries, 1):
        output = trial_root / f"{entry['trial_id']}.npz"
        if _valid_compact_trial(output, entry, representation_keys, feature_dim):
            completed += 1
            continue
        distributions, frequencies = _native_distributions(config, audit, entry)
        expected_time = int(distributions[0].shape[0])
        features: dict[str, np.ndarray] = {}
        if "native_sqrt_jsd_zscore" in representation_keys:
            features["native_sqrt_jsd"] = transform_native_sqrt_jsd(
                distributions, references["arithmetic"]
            )
        if "native_fisher_rao_pca_zscore" in representation_keys:
            if pca is None:
                raise RuntimeError("Fisher-Rao PCA state was not fitted")
            features["native_fisher_rao_pca"] = transform_native_fisher_rao_pca(
                distributions, references["fisher_rao"], pca["means"], pca["components"]
            )
        if "native_wasserstein1_zscore" in representation_keys:
            features["native_wasserstein1"] = transform_native_wasserstein1(
                distributions, references["arithmetic"], frequencies
            )
        if any(
            value.shape != (expected_time, feature_dim)
            for value in features.values()
        ):
            raise ValueError(
                f"Compact feature shape mismatch for {entry['trial_id']}: "
                f"{ {key: value.shape for key, value in features.items()} }"
            )
        write_npz(
            output,
            **{key: value.astype(np.float32) for key, value in features.items()},
            label=np.int64(entry["label"]),
            subject=np.int64(entry["subject"]),
            session=np.int64(entry["session"]),
            trial=np.int64(entry["trial"]),
            source_index=np.int64(entry["source_index"]),
            protocol_hash=np.asarray(protocol_hash),
        )
        completed += 1
        if index % 25 == 0 or index == len(entries):
            LOGGER.info("%s compact transform %d/%d trials", config.dataset, index, len(entries))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete" if completed == len(entries) else "partial",
        "dataset": config.dataset,
        "fold": FOLD,
        "protocol_hash": protocol_hash,
        "fixed_cache_root": audit["cache_root"],
        "ica_cache_root": audit["ica_cache_root"],
        "source_subjects": sorted({int(entry["subject"]) for entry in source_entries}),
        "target_subject": FOLD,
        "source_trials": len(source_entries),
        "target_trials": len(target_entries),
        "completed_trials": completed,
        "reference_source_subjects": sorted({int(entry["subject"]) for entry in reference_entries}),
        "reference_source_trials": len(reference_entries),
        "reference_fit_scope": experiment_settings(config)["reference_scope"],
        "representations": representation_keys,
        "feature_shape": (
            f"[T,{feature_dim}] = [T,{int(config.raw['dataset']['channels'])},"
            f"{len(config.raw['signal']['bands_hz'])}]"
        ),
        "storage_dtype": "float32",
        "reference_state": str((cache_root / "reference_state.npz").resolve()),
        "fisher_rao_pca_state": (
            str((cache_root / "fisher_rao_pca_state.npz").resolve())
            if pca is not None else None
        ),
        "elapsed_seconds_last_transform_pass": time.perf_counter() - started,
        "completed_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    return manifest


def _load_feature_samples(
    feature_cache: Path,
    entries: Sequence[dict[str, Any]],
    representation: str,
    feature_dim: int,
) -> list[TrialSample]:
    if representation not in REPRESENTATION_KEYS:
        raise ValueError(f"Unknown native compact representation: {representation}")
    key = REPRESENTATION_KEYS[representation]
    samples: list[TrialSample] = []
    for entry in entries:
        path = feature_cache / "trials" / f"{entry['trial_id']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            value = np.asarray(archive[key], dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != feature_dim or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key} feature in {path}: {value.shape}")
            if int(archive["source_index"].item()) != int(entry["source_index"]):
                raise ValueError(f"Feature metadata mismatch in {path}")
        samples.append(TrialSample(
            np.ascontiguousarray(value, dtype=np.float32),
            int(entry["label"]),
            int(entry["subject"]),
            int(entry["session"]),
            int(entry["trial"]),
            int(entry["source_index"]),
        ))
    return samples


def _prepare_bundle(
    config: ExperimentConfig,
    audit: dict[str, Any],
    run_root: Path,
    protocol_hash: str,
    representation: str,
) -> dict[str, Any]:
    train_entries, validation_entries, target_entries = _fold_groups(
        Path(audit["cache_root"]), config
    )
    feature_cache = _feature_cache_root(run_root, config.dataset, protocol_hash)
    feature_dim = _feature_dim(config)
    train_samples = _load_feature_samples(
        feature_cache, train_entries, representation, feature_dim
    )
    validation_samples = _load_feature_samples(
        feature_cache, validation_entries, representation, feature_dim
    )
    source_samples = train_samples + validation_samples
    normalization_samples = (
        train_samples
        if experiment_settings(config)["reference_scope"] == "source_train"
        else source_samples
    )
    normalization = scaling_statistics(normalization_samples, True)
    source_locked_at = utc_now()
    # The target compact cache was created only after source reference/PCA
    # fitting.  Arrays and labels enter the experiment only after the source
    # normalizer above is locked.
    target_samples = _load_feature_samples(
        feature_cache, target_entries, representation, feature_dim
    )
    return {
        "source_samples": source_samples,
        "target_samples": target_samples,
        "normalization": normalization,
        "source_subjects": sorted({int(sample.subject) for sample in source_samples}),
        "normalization_subjects": sorted({
            int(sample.subject) for sample in normalization_samples
        }),
        "source_locked_at": source_locked_at,
        "target_loaded_at": utc_now(),
        "feature_cache": str(feature_cache.resolve()),
    }


def _model_config(config: ExperimentConfig) -> dict[str, Any]:
    architecture = experiment_settings(config)["architecture"]
    return {
        "name": "hierarchical_attention",
        "channels": int(config.raw["dataset"]["channels"]),
        "d_model": int(architecture["d_model"]),
        "heads": int(architecture["heads"]),
        "layers": int(architecture["layers"]),
        "feedforward": int(architecture["feedforward"]),
        "dropout": float(architecture["dropout"]),
        "architecture_label": "base",
    }


def _training_config(config: ExperimentConfig, smoke_epochs: int | None = None) -> dict[str, Any]:
    training = copy.deepcopy(config.raw["training"])
    settings = experiment_settings(config)
    training["locked_epochs"] = int(smoke_epochs or settings["max_epochs"])
    training["target_monitor_interval"] = int(settings["target_monitor_interval"])
    training["epoch_policy"] = "predeclared_exploratory_final_epoch"
    training["checkpoint_selection"] = "fixed_final_epoch_only"
    training.pop("early_stopping_patience", None)
    return training


def task_identifier(dataset: str, condition: str) -> str:
    return f"{dataset}__{condition}__fold-01__seed-{FIXED_SEED}"


def declared_tasks(
    config: ExperimentConfig,
    protocol_hash: str,
    conditions: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    expected_conditions = _expected_conditions(config)
    selected = tuple(conditions or expected_conditions)
    if set(selected) - set(expected_conditions):
        raise ValueError("Unknown condition filter")
    tasks = []
    for condition in expected_conditions:
        if condition not in selected:
            continue
        identifier = task_identifier(config.dataset, condition)
        tasks.append({
            "task_id": identifier,
            "dataset": config.dataset,
            "condition": condition,
            "representation": CONDITION_REPRESENTATIONS[condition],
            "architecture": "base",
            "fold": FOLD,
            "seed": FIXED_SEED,
            "protocol_hash": protocol_hash,
            "status": "pending",
            "attempts": 0,
            "result_path": f"{config.dataset}/{condition}/fold-01/seed-{FIXED_SEED}/result.json",
        })
    return tasks


def _task_output(run_root: Path, task: dict[str, Any]) -> Path:
    return (
        run_root / str(task["dataset"]) / str(task["condition"])
        / "fold-01" / f"seed-{int(task['seed'])}"
    )


def run_task(
    config: ExperimentConfig,
    audit: dict[str, Any],
    task: dict[str, Any],
    run_root: Path,
    device: torch.device,
    bundle: dict[str, Any],
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = _task_output(run_root, task)
    output.mkdir(parents=True, exist_ok=True)
    training = _training_config(config, smoke_epochs)
    source_subject_count = int(config.raw["dataset"]["subjects"]) - 1
    checkpoint_path = output / "final_epoch_model.pt"
    events = [
        {
            "event": "source_feature_state_and_normalization_locked",
            "at": bundle["source_locked_at"],
            "source_subject_count": source_subject_count,
            "normalization_subject_count": len(bundle["normalization_subjects"]),
            "normalization_subjects": bundle["normalization_subjects"],
            "target_arrays_loaded": False,
        },
        {
            "event": "exploratory_target_monitor_constructed",
            "at": bundle["target_loaded_at"],
            "target_arrays_loaded": True,
            "target_metrics_affect_training": False,
            "monitor_interval_epochs": int(training["target_monitor_interval"]),
        },
    ]

    def persist_monitor(
        history: Sequence[dict[str, Any]],
        target_curve: Sequence[dict[str, Any]],
    ) -> None:
        _write_csv(output / "training_history.csv", history)
        curve_rows = [{
            "epoch": int(row["epoch"]),
            "accuracy": float(row["accuracy"]),
            "balanced_accuracy": float(row["balanced_accuracy"]),
            "macro_f1": float(row["macro_f1"]),
            "confusion_matrix": json.dumps(row["confusion_matrix"], separators=(",", ":")),
        } for row in target_curve]
        _write_csv(output / "target_curve.csv", curve_rows)
        write_json(output / "target_curve.json", {"curve": list(target_curve), "updated_at": utc_now()})
        latest = target_curve[-1]
        LOGGER.info(
            "%s epoch %d target: accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f",
            task["task_id"], int(latest["epoch"]), float(latest["accuracy"]),
            float(latest["balanced_accuracy"]), float(latest["macro_f1"]),
        )

    training_result = fit_exploratory_monitored_source_model(
        bundle["source_samples"],
        bundle["target_samples"],
        _model_config(config),
        training,
        int(config.raw["dataset"]["classes"]),
        device,
        checkpoint_path,
        seed=int(task["seed"]),
        scale_inputs=True,
        context={
            "family": FAMILY,
            "dataset": config.dataset,
            "condition": task["condition"],
            "fold": FOLD,
            "protocol_hash": task["protocol_hash"],
            "evidence_status": "exploratory_target_monitored",
        },
        normalization=bundle["normalization"],
        monitor_callback=persist_monitor,
    )
    final_metrics, predictions = evaluate_locked_checkpoint(
        checkpoint_path, bundle["target_samples"], device
    )
    _write_csv(output / "training_history.csv", training_result["history"])
    curve_rows = [{
        "epoch": int(row["epoch"]),
        "accuracy": float(row["accuracy"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "confusion_matrix": json.dumps(row["confusion_matrix"], separators=(",", ":")),
    } for row in training_result["target_curve"]]
    _write_csv(output / "target_curve.csv", curve_rows)
    write_json(output / "target_curve.json", {
        "curve": training_result["target_curve"], "updated_at": utc_now()
    })
    _write_csv(output / "predictions.csv", predictions)
    events.append({
        "event": "fixed_final_epoch_checkpoint_complete",
        "at": utc_now(),
        "final_epoch": int(training_result["final_epoch"]),
        "target_metrics_used_for_selection": False,
    })
    write_json(output / "protocol_audit.json", {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "task_id": task["task_id"],
        "protocol_hash": task["protocol_hash"],
        "outer_protocol": "fold-1 LOSO",
        "formal_source_subjects": bundle["source_subjects"],
        "formal_source_subject_count": source_subject_count,
        "source_state_fit_subjects": bundle["normalization_subjects"],
        "source_state_fit_scope": experiment_settings(config)["reference_scope"],
        "target_subject": FOLD,
        "target_monitoring_during_training": True,
        "target_monitor_interval": int(training["target_monitor_interval"]),
        "target_metrics_affect_training": False,
        "early_stopping": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "native_grid": task["representation"] != "de_zscore",
        "feature_cache": bundle["feature_cache"],
        "events": events,
    })
    if task["representation"] == "native_fisher_rao_pca_zscore":
        reference_rule = "source_hellinger_barycenter_plus_unsupervised_tangent_pca"
    elif task["representation"] == "de_zscore":
        reference_rule = "source_train_channel_band_zscore"
    else:
        reference_rule = "source_pooled_arithmetic_probability_mean"
    result = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "task_id": task["task_id"],
        "dataset": config.dataset,
        "condition": task["condition"],
        "representation": task["representation"],
        "architecture": "base",
        "training_method": "v2",
        "fold": FOLD,
        "seed": int(task["seed"]),
        "protocol_hash": task["protocol_hash"],
        "preprocessing_signature": audit["preprocessing_signature"],
        "cleaning_signature": audit["cleaning_signature"],
        "native_nfft": (
            None if task["representation"] == "de_zscore" else audit["native_nfft"]
        ),
        "source_zscore": True,
        "reference_rule": reference_rule,
        "formal_source_subject_count": source_subject_count,
        "source_state_fit_subject_count": len(bundle["normalization_subjects"]),
        "source_state_fit_scope": experiment_settings(config)["reference_scope"],
        "max_epochs": int(training["locked_epochs"]),
        "target_monitor_interval": int(training["target_monitor_interval"]),
        "target_metrics_affect_training": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "parameter_count": int(training_result["parameter_count"]),
        "effective_batch_size": int(training["batch_size"]) * int(training.get("gradient_accumulation_steps", 1)),
        "precision": training.get("precision", "float32"),
        "final_target_test": final_metrics,
        "target_curve_points": len(curve_rows),
        "feature_cache": bundle["feature_cache"],
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
        "diagnostic_smoke": smoke_epochs is not None,
    }
    write_json(output / "result.json", result)
    write_json(output / "COMPLETE.json", {
        "task_id": task["task_id"],
        "protocol_hash": task["protocol_hash"],
        "completed_at": result["completed_at"],
    })
    return result


def _load_or_merge_manifest(
    run_root: Path,
    tasks: Sequence[dict[str, Any]],
    audit: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    path = run_root / "matrix_manifest.json"
    manifest = read_json(path) if path.is_file() else {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "created_at": utc_now(),
        "expected_tasks": 0,
        "protocols": {},
        "cache_audits": {},
        "tasks": {},
    }
    protocol_hash = _json_hash(protocol)
    manifest["protocols"][audit["dataset"]] = {
        "protocol_hash": protocol_hash,
        "payload": protocol,
    }
    manifest["expected_tasks"] = sum(
        len(item["payload"]["representations"])
        for item in manifest["protocols"].values()
    )
    manifest["cache_audits"][audit["dataset"]] = audit
    for task in tasks:
        existing = manifest["tasks"].get(task["task_id"])
        if existing and existing.get("protocol_hash") != task["protocol_hash"]:
            raise ValueError(f"Task exists under a different protocol: {task['task_id']}")
        if existing is None:
            manifest["tasks"][task["task_id"]] = task
    manifest["updated_at"] = utc_now()
    write_json(path, manifest)
    return manifest


def run_matrix(
    config: ExperimentConfig,
    run_root: Path,
    cache_parent: Path | None,
    conditions: Sequence[str] | None,
    resume: bool,
    retry_failed: bool,
    max_tasks: int | None = None,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    audit = validate_native_sources(config, cache_parent)
    protocol = protocol_payload(config, audit)
    protocol_hash = _json_hash(protocol)
    if smoke_epochs is None:
        _require_lock(config, run_root, protocol_hash)
    tasks = declared_tasks(config, protocol_hash, conditions)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_merge_manifest(run_root, tasks, audit, protocol)
    prepare_native_feature_cache(config, run_root, cache_parent)
    device = select_device(str(config.raw["training"].get("device", "auto")))
    runnable: list[dict[str, Any]] = []
    for declared in tasks:
        task = manifest["tasks"][declared["task_id"]]
        marker_path = _task_output(run_root, task) / "COMPLETE.json"
        if task.get("status") == "complete" or marker_path.is_file():
            marker = read_json(marker_path) if marker_path.is_file() else {}
            if marker.get("protocol_hash", task["protocol_hash"]) != task["protocol_hash"]:
                raise ValueError(f"Completed artifact hash mismatch: {task['task_id']}")
            if not resume:
                raise FileExistsError(f"Task already complete; rerun with --resume: {task['task_id']}")
            task["status"] = "complete"
            continue
        if task.get("status") == "failed" and not retry_failed:
            continue
        runnable.append(task)
    if max_tasks is not None:
        runnable = runnable[:max_tasks]

    for task in runnable:
        output = _task_output(run_root, task)
        task["status"] = "running"
        task["started_at"] = utc_now()
        task["attempts"] = int(task.get("attempts", 0)) + 1
        task.pop("error", None)
        write_json(output / "status.json", {"status": "running", "task": task})
        manifest["updated_at"] = utc_now()
        write_json(run_root / "matrix_manifest.json", manifest)
        LOGGER.info("Running %s", task["task_id"])
        bundle: dict[str, Any] | None = None
        try:
            bundle = _prepare_bundle(
                config, audit, run_root, protocol_hash, str(task["representation"])
            )
            result = run_task(config, audit, task, run_root, device, bundle, smoke_epochs)
            task["status"] = "complete"
            task["completed_at"] = result["completed_at"]
            task["elapsed_seconds"] = result["elapsed_seconds"]
            write_json(output / "status.json", {"status": "complete", "task": task})
        except BaseException as exc:
            task["status"] = "failed"
            task["failed_at"] = utc_now()
            task["error"] = f"{type(exc).__name__}: {exc}"
            write_json(output / "status.json", {
                "status": "failed",
                "task": task,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
            LOGGER.error("Task failed: %s: %s", task["task_id"], task["error"])
            if isinstance(exc, KeyboardInterrupt):
                manifest["updated_at"] = utc_now()
                write_json(run_root / "matrix_manifest.json", manifest)
                raise
        finally:
            if bundle is not None:
                bundle.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        manifest["updated_at"] = utc_now()
        write_json(run_root / "matrix_manifest.json", manifest)
    return matrix_status(run_root)


def matrix_status(run_root: Path) -> dict[str, Any]:
    path = run_root / "matrix_manifest.json"
    if not path.is_file():
        return {"status": "not_started", "family": FAMILY, "run_root": str(run_root), "declared": 0}
    manifest = read_json(path)
    tasks = list(manifest.get("tasks", {}).values())
    counts = {
        status: sum(task.get("status") == status for task in tasks)
        for status in ("pending", "running", "complete", "failed")
    }
    failed = [
        {"task_id": task["task_id"], "error": task.get("error")}
        for task in tasks if task.get("status") == "failed"
    ]
    expected_tasks = int(manifest.get("expected_tasks", len(tasks)))
    payload = {
        "status": (
            "complete"
            if tasks and len(tasks) == expected_tasks and counts["complete"] == expected_tasks
            else "in_progress"
        ),
        "family": FAMILY,
        "evidence_status": "exploratory_target_monitored",
        "run_root": str(run_root),
        "declared": len(tasks),
        "expected_tasks": expected_tasks,
        **counts,
        "failed_tasks": failed,
        "updated_at": manifest.get("updated_at"),
    }
    write_json(run_root / "progress.json", payload)
    _write_csv(run_root / "failed_tasks.csv", failed)
    return payload


def summarize(run_root: Path, allow_partial: bool = False) -> dict[str, Any]:
    status = matrix_status(run_root)
    expected_tasks = int(status.get("expected_tasks", status.get("declared", 0)))
    if not allow_partial and (
        status.get("declared") != expected_tasks
        or status.get("complete") != expected_tasks
    ):
        raise RuntimeError(
            f"Strict summary requires all {expected_tasks} protocol tasks; status={status}"
        )
    manifest = read_json(run_root / "matrix_manifest.json")
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for task in manifest["tasks"].values():
        if task.get("status") != "complete":
            continue
        result_path = run_root / str(task["result_path"])
        result = read_json(result_path)
        key = (str(result["dataset"]), str(result["condition"]))
        if key in seen:
            raise ValueError(f"Duplicate native compact result: {key}")
        seen.add(key)
        previous = hashes.setdefault(str(result["dataset"]), str(result["protocol_hash"]))
        if previous != result["protocol_hash"]:
            raise ValueError(f"Mixed protocol hashes for {result['dataset']}")
        metrics = result["final_target_test"]
        if not all(math.isfinite(float(metrics[name])) for name in ("accuracy", "balanced_accuracy", "macro_f1")):
            raise ValueError(f"Non-finite final metrics: {result_path}")
        rows.append({
            "dataset": result["dataset"],
            "condition": result["condition"],
            "representation": result["representation"],
            "architecture": result["architecture"],
            "training_method": result["training_method"],
            "fold": int(result["fold"]),
            "seed": int(result["seed"]),
            "max_epochs": int(result["max_epochs"]),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "parameter_count": int(result["parameter_count"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "protocol_hash": result["protocol_hash"],
        })
        curve = read_json(result_path.parent / "target_curve.json")["curve"]
        for point in curve:
            curve_rows.append({
                "dataset": result["dataset"],
                "condition": result["condition"],
                "fold": FOLD,
                "epoch": int(point["epoch"]),
                "accuracy": float(point["accuracy"]),
                "balanced_accuracy": float(point["balanced_accuracy"]),
                "macro_f1": float(point["macro_f1"]),
            })
    rows.sort(key=lambda row: (row["dataset"], EXPECTED_CONDITIONS.index(row["condition"])))
    curve_rows.sort(key=lambda row: (
        row["dataset"], EXPECTED_CONDITIONS.index(row["condition"]), row["epoch"]
    ))
    _write_csv(run_root / "fold1_results.csv", rows)
    _write_csv(run_root / "target_curves.csv", curve_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete" if len(rows) == expected_tasks else "partial",
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "seed": FIXED_SEED,
        "fold": FOLD,
        "results": rows,
        "protocol_hashes": hashes,
        "generated_at": utc_now(),
    }
    write_json(run_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fold-1 native-grid compact feature experiments with Base/v2 training"
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def config_command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--cache-parent")
        return child

    config_command("validate-cache")
    lock = config_command("lock")
    lock.add_argument("--run-root", required=True)
    prepare = config_command("prepare-features")
    prepare.add_argument("--run-root", required=True)
    for name in ("smoke", "matrix"):
        child = config_command(name)
        child.add_argument("--run-root", required=True)
        child.add_argument("--condition", action="append")
        child.add_argument("--resume", action="store_true")
        child.add_argument("--retry-failed", action="store_true")
        child.add_argument("--max-tasks", type=int)
        if name == "smoke":
            child.add_argument("--smoke-epochs", type=int, default=2)
    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--run-root", required=True)
    summary.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.command == "status":
        payload = matrix_status(Path(args.run_root).expanduser().resolve())
    elif args.command == "summarize":
        payload = summarize(Path(args.run_root).expanduser().resolve(), args.allow_partial)
    else:
        config = load_config(args.config)
        experiment_settings(config)
        cache_parent = Path(args.cache_parent).expanduser().resolve() if args.cache_parent else None
        if args.command == "validate-cache":
            payload = validate_native_sources(config, cache_parent)
        elif args.command == "lock":
            payload = lock_experiment(
                config, Path(args.run_root).expanduser().resolve(), cache_parent
            )
        elif args.command == "prepare-features":
            payload = prepare_native_feature_cache(
                config, Path(args.run_root).expanduser().resolve(), cache_parent
            )
        else:
            payload = run_matrix(
                config,
                Path(args.run_root).expanduser().resolve(),
                cache_parent,
                args.condition,
                args.resume,
                args.retry_failed,
                args.max_tasks,
                args.smoke_epochs if args.command == "smoke" else None,
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
