from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import logging
import math
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmrd.config import ExperimentConfig, load_config
from cmrd.fixed_protocol import (
    FIXED_SEED,
    MODEL_NAMES,
    REFERENCE_METHODS,
    REPRESENTATIONS,
    _forward_sequence_model,
    _loader,
    _valid_indices_on_device,
    build_model,
    evaluate_locked_checkpoint,
    evaluate_locked_classical_model,
    feature_cache_info,
    fit_locked_classical_model,
    fit_locked_source_model,
    fit_reference,
    load_representation_samples,
    pooled_vectors,
    reference_leave_one_subject_sensitivity,
    representation_uses_source_zscore,
    resolve_complete_cache,
    scaling_statistics,
)
from cmrd.io import read_json, write_json
from cmrd.training.runtime import select_device


LOGGER = logging.getLogger("cmrd.fixed_protocol")
SCHEMA_VERSION = 1
EXPECTED_DATASETS = ("seed", "seediv")
EXPECTED_FOLDS = tuple(range(1, 16))
CLASSICAL_MODELS = ("logistic_regression", "linear_svm")
NEURAL_MODELS = tuple(name for name in MODEL_NAMES if name not in CLASSICAL_MODELS)


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


def _matrix_settings(config: ExperimentConfig) -> dict[str, Any]:
    raw = copy.deepcopy(config.raw.get("matrix", {}))
    settings = {
        "representations": list(raw.get("representations", REPRESENTATIONS)),
        "models": list(raw.get("models", MODEL_NAMES)),
        "reference_method": str(raw.get("reference_method", "pooled_mean")),
        "seed": int(raw.get("seed", FIXED_SEED)),
        "fixed_epoch": int(raw.get("fixed_epoch", 0)),
        "histogram_batch_size": int(raw.get("histogram_batch_size", 1)),
        "default_batch_size": int(raw.get("default_batch_size", config.raw["training"]["batch_size"])),
        "model_batch_sizes": {
            str(key): int(value) for key, value in raw.get("model_batch_sizes", {}).items()
        },
        "precision": str(raw.get("precision", "float32")).lower(),
        "model_precisions": {
            str(key): str(value).lower() for key, value in raw.get("model_precisions", {}).items()
        },
        "matmul_precision": str(raw.get("matmul_precision", "high")).lower(),
        "dataloader_workers": int(raw.get("dataloader_workers", 0)),
        "persistent_workers": bool(raw.get("persistent_workers", False)),
        "prefetch_factor": int(raw.get("prefetch_factor", 1)),
        "classical_workers": int(raw.get("classical_workers", 1)),
        "classical_threads_per_worker": int(raw.get("classical_threads_per_worker", 1)),
        "linear_svm_tol": float(raw.get("linear_svm_tol", 1e-3)),
        "linear_svm_max_iter": int(raw.get("linear_svm_max_iter", 5000)),
    }
    if tuple(settings["representations"]) != REPRESENTATIONS:
        raise ValueError(f"Fixed matrix representations must be {REPRESENTATIONS}")
    if tuple(settings["models"]) != MODEL_NAMES:
        raise ValueError(f"Fixed matrix models must be {MODEL_NAMES}")
    if settings["reference_method"] != "pooled_mean":
        raise ValueError("The main matrix reference method is frozen to pooled_mean")
    if settings["seed"] != FIXED_SEED:
        raise ValueError(f"The first-stage matrix seed is frozen to {FIXED_SEED}")
    if settings["fixed_epoch"] < 1:
        raise ValueError("matrix.fixed_epoch must be locked to one positive dataset-level value")
    if settings["histogram_batch_size"] < 1 or settings["default_batch_size"] < 1:
        raise ValueError("Matrix batch sizes must be positive")
    if set(settings["model_batch_sizes"]) - set(NEURAL_MODELS):
        raise ValueError("matrix.model_batch_sizes contains an unknown or classical model")
    if any(value < 1 for value in settings["model_batch_sizes"].values()):
        raise ValueError("matrix.model_batch_sizes values must be positive")
    if settings["precision"] not in {"float32", "bfloat16", "float16"}:
        raise ValueError("matrix.precision must be float32, bfloat16, or float16")
    if set(settings["model_precisions"]) - set(NEURAL_MODELS):
        raise ValueError("matrix.model_precisions contains an unknown or classical model")
    if any(value not in {"float32", "bfloat16", "float16"} for value in settings["model_precisions"].values()):
        raise ValueError("matrix.model_precisions values must be float32, bfloat16, or float16")
    if settings["matmul_precision"] not in {"highest", "high", "medium"}:
        raise ValueError("matrix.matmul_precision must be highest, high, or medium")
    if settings["dataloader_workers"] not in {0, 1, 2}:
        raise ValueError("matrix.dataloader_workers is limited to 0..2 for Windows memory safety")
    if settings["persistent_workers"] and settings["dataloader_workers"] == 0:
        raise ValueError("matrix.persistent_workers requires dataloader_workers > 0")
    if settings["prefetch_factor"] < 1:
        raise ValueError("matrix.prefetch_factor must be positive")
    if settings["classical_workers"] < 1 or settings["classical_threads_per_worker"] < 1:
        raise ValueError("Classical worker and thread counts must be positive")
    if settings["linear_svm_tol"] <= 0 or settings["linear_svm_max_iter"] < 1:
        raise ValueError("Linear SVM tolerance and iteration limit must be positive")
    return settings


def default_cache_parent(config: ExperimentConfig) -> Path:
    return config.processed_root / config.dataset / "de_rjsd_ica_1s_hop1"


def validate_fixed_cache(config: ExperimentConfig, cache_parent: Path | None = None) -> dict[str, Any]:
    if config.dataset not in EXPECTED_DATASETS:
        raise ValueError(f"Unexpected dataset: {config.dataset}")
    root = resolve_complete_cache((cache_parent or default_cache_parent(config)).resolve(), config.dataset)
    pipeline = read_json(root / "pipeline_manifest.json")
    expected_dataset = "SEED-IV" if config.dataset == "seediv" else "SEED"
    if pipeline.get("dataset") != expected_dataset or not pipeline.get("all_15_folds_complete"):
        raise ValueError(f"Incomplete or wrong cache at {root}")
    signature = str(pipeline["preprocessing_signature"])
    fold_rows = []
    all_subjects = set(EXPECTED_FOLDS)
    for fold in EXPECTED_FOLDS:
        manifest_path = root / "folds" / f"fold-{fold:02d}" / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing fold manifest: {manifest_path}")
        manifest = read_json(manifest_path)
        if str(manifest.get("preprocessing_signature")) != signature:
            raise ValueError(f"Fold {fold:02d} has a mismatched preprocessing signature")
        groups = manifest["groups"]
        train_subjects = {int(item["subject"]) for item in groups["train"]}
        validation_subjects = {int(item["subject"]) for item in groups["validation"]}
        target_subjects = {int(item["subject"]) for item in groups["test"]}
        if len(train_subjects) != 12 or len(validation_subjects) != 2 or target_subjects != {fold}:
            raise ValueError(
                f"Fold {fold:02d} must contain 12 source-train, 2 source-validation, and target {fold}; "
                f"got {len(train_subjects)}, {len(validation_subjects)}, {sorted(target_subjects)}"
            )
        if train_subjects & validation_subjects or (train_subjects | validation_subjects) != all_subjects - {fold}:
            raise ValueError(f"Fold {fold:02d} violates 14-source/1-target LOSO isolation")
        for group_name in ("train", "validation", "test"):
            for entry in groups[group_name]:
                path = root / str(entry["de_phist_path"])
                if not path.is_file():
                    raise FileNotFoundError(f"Missing feature archive: {path}")
        fold_rows.append({
            "fold": fold,
            "source_train_subjects": sorted(train_subjects),
            "source_validation_subjects": sorted(validation_subjects),
            "target_subject": fold,
            "train_trials": len(groups["train"]),
            "validation_trials": len(groups["validation"]),
            "target_trials": len(groups["test"]),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.dataset,
        "cache_root": str(root),
        "preprocessing_signature": signature,
        "outer_protocol": "15-fold LOSO: 14 source subjects / 1 target subject",
        "cache_diagnostic_split": "12 source-train + 2 source-validation subjects; formal matrix reunites all 14",
        "folds": fold_rows,
        "validated_at": utc_now(),
    }


def protocol_payload(config: ExperimentConfig, audit: dict[str, Any]) -> dict[str, Any]:
    matrix = _matrix_settings(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.dataset,
        "seed": matrix["seed"],
        "window_seconds": 1.0,
        "hop_seconds": 1.0,
        "outer_folds": 15,
        "source_subjects_per_fold": 14,
        "formal_source_validation_subjects": 0,
        "cache_diagnostic_validation_subjects": 2,
        "fixed_epoch": matrix["fixed_epoch"],
        "epoch_policy": "one predeclared dataset-level epoch shared by all 15 folds",
        "target_monitoring": False,
        "representations": matrix["representations"],
        "models": matrix["models"],
        "reference_method": matrix["reference_method"],
        "preprocessing_signature": audit["preprocessing_signature"],
        "model": config.raw["model"],
        "training": config.raw["training"],
    }


def _model_config(config: ExperimentConfig, name: str) -> dict[str, Any]:
    if name not in MODEL_NAMES:
        raise ValueError(name)
    base = config.raw["model"]
    return {
        "name": name,
        "d_model": int(base["d_model"]),
        "heads": int(base.get("heads", base.get("nhead", 4))),
        "layers": int(base["layers"]),
        "feedforward": int(base["feedforward"]),
        "dropout": float(base["dropout"]),
        "hidden": int(base.get("hidden", 128)),
    }


def _training_config(config: ExperimentConfig, representation: str, epochs: int | None = None) -> dict[str, Any]:
    training = copy.deepcopy(config.raw["training"])
    matrix = _matrix_settings(config)
    training["batch_size"] = (
        matrix["histogram_batch_size"] if representation == "histogram" else matrix["default_batch_size"]
    )
    if epochs is not None:
        training["epochs"] = int(epochs)
        training["early_stopping_patience"] = max(1, min(int(training["early_stopping_patience"]), int(epochs)))
    return training


def _neural_loader_settings(
    matrix: dict[str, Any], representation: str, model_name: str,
) -> dict[str, Any]:
    """Choose a result-equivalent loader path that is safe on Windows.

    Histogram batches are hundreds of MB after padding. Sending them through
    multiprocessing queues creates additional Windows shared-file mappings and
    can exhaust the system commit limit (WinError 1455). Keeping their one-time
    normalization cache in the main process avoids those duplicate mappings.
    """
    if model_name == "small_mlp":
        return {
            "num_workers": 0,
            "persistent_workers": False,
            "normalization_cache": "trial_pool_once_source_zscore_v1",
            "loader_policy": "pooled_features_main_process",
        }
    if representation == "histogram":
        return {
            "num_workers": 0,
            "persistent_workers": False,
            "normalization_cache": "source_trial_once_contiguous_local_v1",
            "loader_policy": "large_histogram_main_process_no_shared_mapping",
        }
    return {
        "num_workers": int(matrix["dataloader_workers"]),
        "persistent_workers": bool(matrix["persistent_workers"]),
        "normalization_cache": "source_trial_once_contiguous_shared_v1",
        "loader_policy": "compact_sequence_persistent_workers",
    }


def task_identifier(dataset: str, representation: str, model: str, fold: int, seed: int = FIXED_SEED) -> str:
    return f"{dataset}__{representation}__{model}__fold-{fold:02d}__seed-{seed}"


def declared_tasks(
    config: ExperimentConfig,
    protocol_hash: str,
    representations: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    folds: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    selected_representations = tuple(representations or REPRESENTATIONS)
    requested_models = set(models or MODEL_NAMES)
    selected_models = tuple(name for name in (*CLASSICAL_MODELS, *NEURAL_MODELS) if name in requested_models)
    selected_folds = tuple(folds or EXPECTED_FOLDS)
    if any(item not in REPRESENTATIONS for item in selected_representations):
        raise ValueError("Unknown representation filter")
    if requested_models - set(MODEL_NAMES):
        raise ValueError("Unknown model filter")
    if any(fold not in EXPECTED_FOLDS for fold in selected_folds):
        raise ValueError("Fold filter must be in 1..15")
    tasks = []
    # Keep cheap classical baselines ahead of neural jobs across every
    # representation, matching the staged experiment plan.
    for model in selected_models:
        for representation in selected_representations:
            for fold in selected_folds:
                identifier = task_identifier(config.dataset, representation, model, fold)
                tasks.append({
                    "task_id": identifier,
                    "dataset": config.dataset,
                    "representation": representation,
                    "model": model,
                    "fold": fold,
                    "seed": FIXED_SEED,
                    "protocol_hash": protocol_hash,
                    "status": "pending",
                    "attempts": 0,
                    "result_path": f"{config.dataset}/{representation}/{model}/fold-{fold:02d}/seed-{FIXED_SEED}/result.json",
                })
    return tasks


def _load_or_merge_manifest(run_root: Path, tasks: Sequence[dict[str, Any]], audit: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    path = run_root / "matrix_manifest.json"
    manifest = read_json(path) if path.is_file() else {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "expected_full_matrix_tasks": 600,
        "protocols": {},
        "cache_audits": {},
        "tasks": {},
    }
    protocol_hash = _json_hash(protocol)
    manifest["protocols"][audit["dataset"]] = {"protocol_hash": protocol_hash, "payload": protocol}
    manifest["cache_audits"][audit["dataset"]] = audit
    for task in tasks:
        existing = manifest["tasks"].get(task["task_id"])
        if existing and existing["protocol_hash"] != task["protocol_hash"]:
            raise ValueError(f"Task {task['task_id']} already exists under a different protocol hash")
        if not existing:
            manifest["tasks"][task["task_id"]] = task
    manifest["updated_at"] = utc_now()
    write_json(path, manifest)
    return manifest


def lock_dataset_epoch(
    config: ExperimentConfig,
    run_root: Path,
    cache_parent: Path | None = None,
) -> dict[str, Any]:
    """Create the immutable dataset-level epoch declaration before formal runs."""
    audit = validate_fixed_cache(config, cache_parent)
    protocol = protocol_payload(config, audit)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.dataset,
        "fixed_epoch": int(_matrix_settings(config)["fixed_epoch"]),
        "epoch_policy": "one predeclared dataset-level epoch shared by all 15 folds",
        "selection_source": "configuration fixed before formal target evaluation",
        "target_metrics_used": False,
        "protocol_hash": _json_hash(protocol),
        "preprocessing_signature": audit["preprocessing_signature"],
        "locked_at": utc_now(),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / f"epoch_lock_{config.dataset}.json"
    if path.is_file():
        existing = read_json(path)
        comparable = {key: value for key, value in payload.items() if key != "locked_at"}
        existing_comparable = {key: value for key, value in existing.items() if key != "locked_at"}
        if existing_comparable != comparable:
            raise ValueError(f"Existing epoch lock conflicts with the active protocol: {path}")
        return existing
    manifest_path = run_root / "matrix_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        dataset_tasks = [task for task in manifest.get("tasks", {}).values() if task.get("dataset") == config.dataset]
        if any(task.get("status") in {"running", "complete"} for task in dataset_tasks):
            raise RuntimeError("Cannot create or change an epoch lock after formal tasks have started")
    write_json(path, payload)
    return payload


def require_epoch_lock(config: ExperimentConfig, run_root: Path, protocol_hash: str) -> dict[str, Any]:
    path = run_root / f"epoch_lock_{config.dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset epoch is not locked. Run lock-epoch before the formal matrix: {path}"
        )
    lock = read_json(path)
    expected_epoch = int(_matrix_settings(config)["fixed_epoch"])
    if int(lock.get("fixed_epoch", -1)) != expected_epoch or lock.get("protocol_hash") != protocol_hash:
        raise ValueError(f"Epoch lock does not match the active config/protocol: {path}")
    if lock.get("target_metrics_used") is not False:
        raise ValueError("Epoch lock must declare that no target metrics were used")
    return lock


def _task_output(run_root: Path, task: dict[str, Any]) -> Path:
    return run_root / task["dataset"] / task["representation"] / task["model"] / f"fold-{int(task['fold']):02d}" / f"seed-{int(task['seed'])}"


def _fold_source_entries(cache_root: Path, fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(cache_root / "folds" / f"fold-{fold:02d}" / "manifest.json")
    return list(manifest["groups"]["train"]), list(manifest["groups"]["validation"]), manifest["split"]


def _fold_target_entries(cache_root: Path, fold: int) -> list[dict[str, Any]]:
    # This function is deliberately separate and is called only after a locked
    # source-only model artifact exists.
    manifest = read_json(cache_root / "folds" / f"fold-{fold:02d}" / "manifest.json")
    return list(manifest["groups"]["test"])


def run_fold_task(
    config: ExperimentConfig,
    cache_audit: dict[str, Any],
    protocol_hash: str,
    task: dict[str, Any],
    run_root: Path,
    device: torch.device,
    fixed_epoch_override: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    cache_root = Path(cache_audit["cache_root"])
    fold = int(task["fold"])
    representation = str(task["representation"])
    model_name = str(task["model"])
    seed = int(task["seed"])
    output = _task_output(run_root, task)
    output.mkdir(parents=True, exist_ok=True)
    matrix = _matrix_settings(config)
    fixed_epoch = int(fixed_epoch_override or matrix["fixed_epoch"])
    diagnostic_smoke = fixed_epoch_override is not None
    audit_events: list[dict[str, Any]] = [{
        "event": "dataset_epoch_declared",
        "at": utc_now(),
        "target_arrays_loaded": False,
        "fixed_epoch": fixed_epoch,
        "diagnostic_smoke_override": diagnostic_smoke,
    }]
    train_entries, validation_entries, _ = _fold_source_entries(cache_root, fold)
    all_source_entries = train_entries + validation_entries
    source_subjects = sorted({int(item["subject"]) for item in all_source_entries})
    if len(source_subjects) != 14 or fold in source_subjects:
        raise ValueError("Formal LOSO training must use all 14 non-target source subjects")

    final_reference = None
    final_reference_provenance = None
    if representation == "rjsd_zscore":
        final_reference, final_reference_provenance = fit_reference(cache_root, all_source_entries, "pooled_mean")
        np.save(output / "locked_reference.npy", final_reference, allow_pickle=False)
    source_samples = load_representation_samples(cache_root, all_source_entries, representation, final_reference)
    model_config = _model_config(config, model_name)
    training = _training_config(config, representation)
    if model_name in NEURAL_MODELS:
        training["batch_size"] = matrix["model_batch_sizes"].get(
            model_name, training["batch_size"]
        )
        training["precision"] = matrix["model_precisions"].get(model_name, matrix["precision"])
        training["matmul_precision"] = matrix["matmul_precision"]
        loader_settings = _neural_loader_settings(matrix, representation, model_name)
        training.update(loader_settings)
        training["prefetch_factor"] = matrix["prefetch_factor"]
    training["epochs"] = fixed_epoch
    training["epoch_policy"] = "dataset_fixed"
    training.pop("early_stopping_patience", None)
    scale_inputs = representation_uses_source_zscore(representation)
    locked_context = {
        "dataset": config.dataset,
        "fold": fold,
        "representation": representation,
        "model": model_name,
        "protocol_hash": protocol_hash,
        "source_subjects": source_subjects,
        "target_subject": fold,
        "fixed_epoch": fixed_epoch,
        "epoch_policy": "dataset_fixed",
    }
    if model_name in CLASSICAL_MODELS:
        locked_path = output / "locked_source_model.pkl"
        estimator_options = (
            {
                "tol": matrix["linear_svm_tol"],
                "max_iter": matrix["linear_svm_max_iter"],
            }
            if model_name == "linear_svm" else {}
        )
        fit_locked_classical_model(
            source_samples, model_name, int(config.raw["dataset"]["classes"]), locked_path,
            seed=seed, scale_inputs=scale_inputs, context=locked_context,
            estimator_options=estimator_options,
        )
    else:
        locked_path = output / "locked_source_model.pt"
        locked_training = copy.deepcopy(training)
        locked_training["locked_epochs"] = fixed_epoch
        fit_locked_source_model(
            source_samples, model_config, locked_training, int(config.raw["dataset"]["classes"]),
            device, locked_path, seed=seed, scale_inputs=scale_inputs, context=locked_context,
        )
    del source_samples
    gc.collect()
    if not locked_path.is_file():
        raise RuntimeError("Locked source-only model was not created")
    audit_events.append({
        "event": "locked_source_model_complete",
        "at": utc_now(),
        "target_arrays_loaded": False,
        "locked_model": str(locked_path),
    })

    # Sole target-array access boundary.
    target_entries = _fold_target_entries(cache_root, fold)
    target_samples = load_representation_samples(cache_root, target_entries, representation, final_reference)
    audit_events.append({
        "event": "target_final_evaluation_started",
        "at": utc_now(),
        "target_arrays_loaded": True,
        "target_subject": fold,
        "target_trials": len(target_samples),
    })
    if model_name in CLASSICAL_MODELS:
        test_metrics, predictions = evaluate_locked_classical_model(locked_path, target_samples)
    else:
        test_metrics, predictions = evaluate_locked_checkpoint(locked_path, target_samples, device)
    del target_samples
    gc.collect()
    _write_csv(output / "predictions.csv", predictions)
    audit_events.append({
        "event": "target_final_evaluation_complete",
        "at": utc_now(),
        "target_arrays_loaded": True,
    })
    protocol_audit = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "protocol_hash": protocol_hash,
        "outer_protocol": "15-fold LOSO",
        "formal_source_subjects": source_subjects,
        "formal_source_subject_count": len(source_subjects),
        "formal_source_validation_subjects": [],
        "target_subject": fold,
        "epoch_policy": "one dataset-level fixed epoch shared by all folds",
        "fixed_epoch": fixed_epoch,
        "diagnostic_smoke": diagnostic_smoke,
        "model_input_adapter": (
            "trial_mean_std_source_zscore_v1" if model_name == "small_mlp" else "sequence_v1"
        ),
        "effective_batch_size": int(training["batch_size"]) if model_name in NEURAL_MODELS else None,
        "precision": training.get("precision") if model_name in NEURAL_MODELS else None,
        "dataloader_workers": training.get("num_workers") if model_name in NEURAL_MODELS else None,
        "persistent_workers": training.get("persistent_workers") if model_name in NEURAL_MODELS else None,
        "normalization_cache": training.get("normalization_cache") if model_name in NEURAL_MODELS else None,
        "loader_policy": training.get("loader_policy") if model_name in NEURAL_MODELS else None,
        "classical_estimator_options": estimator_options if model_name in CLASSICAL_MODELS else None,
        "target_monitoring_during_training": False,
        "events": audit_events,
    }
    write_json(output / "protocol_audit.json", protocol_audit)
    result = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "status": "complete",
        "dataset": config.dataset,
        "representation": representation,
        "model": model_name,
        "fold": fold,
        "seed": seed,
        "protocol_hash": protocol_hash,
        "preprocessing_signature": cache_audit["preprocessing_signature"],
        "source_zscore": scale_inputs,
        "reference_method": "pooled_mean" if representation == "rjsd_zscore" else None,
        "locked_reference_provenance": final_reference_provenance,
        "epoch_policy": "dataset_fixed",
        "fixed_epoch": fixed_epoch,
        "formal_source_subject_count": 14,
        "diagnostic_smoke": diagnostic_smoke,
        "model_input_adapter": (
            "trial_mean_std_source_zscore_v1" if model_name == "small_mlp" else "sequence_v1"
        ),
        "effective_batch_size": int(training["batch_size"]) if model_name in NEURAL_MODELS else None,
        "precision": training.get("precision") if model_name in NEURAL_MODELS else None,
        "dataloader_workers": training.get("num_workers") if model_name in NEURAL_MODELS else None,
        "persistent_workers": training.get("persistent_workers") if model_name in NEURAL_MODELS else None,
        "normalization_cache": training.get("normalization_cache") if model_name in NEURAL_MODELS else None,
        "loader_policy": training.get("loader_policy") if model_name in NEURAL_MODELS else None,
        "classical_estimator_options": estimator_options if model_name in CLASSICAL_MODELS else None,
        "target_test": test_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "feature_cache": feature_cache_info(),
        "completed_at": utc_now(),
    }
    write_json(output / "result.json", result)
    write_json(output / "COMPLETE.json", {
        "task_id": task["task_id"],
        "protocol_hash": protocol_hash,
        "completed_at": result["completed_at"],
    })
    return result


def run_matrix(
    config: ExperimentConfig,
    run_root: Path,
    cache_parent: Path | None,
    representations: Sequence[str] | None,
    models: Sequence[str] | None,
    folds: Sequence[int] | None,
    resume: bool,
    retry_failed: bool,
    max_tasks: int | None,
    fixed_epoch_override: int | None = None,
) -> dict[str, Any]:
    audit = validate_fixed_cache(config, cache_parent)
    protocol = protocol_payload(config, audit)
    protocol_hash = _json_hash(protocol)
    if fixed_epoch_override is None:
        require_epoch_lock(config, run_root, protocol_hash)
    tasks = declared_tasks(config, protocol_hash, representations, models, folds)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_merge_manifest(run_root, tasks, audit, protocol)
    write_json(run_root / f"protocol_audit_{config.dataset}.json", audit)
    device = select_device(str(config.raw["training"].get("device", "auto")))
    runnable: list[dict[str, Any]] = []
    for declared in tasks:
        task = manifest["tasks"][declared["task_id"]]
        output = _task_output(run_root, task)
        complete_marker = output / "COMPLETE.json"
        if task["status"] == "complete" or complete_marker.is_file():
            marker = read_json(complete_marker) if complete_marker.is_file() else {}
            if marker.get("protocol_hash", task["protocol_hash"]) != task["protocol_hash"]:
                raise ValueError(f"Completed artifact hash mismatch for {task['task_id']}")
            if not resume:
                raise FileExistsError(f"Task already complete; rerun with --resume: {task['task_id']}")
            task["status"] = "complete"
            continue
        if task["status"] == "failed" and not retry_failed:
            continue
        runnable.append(task)
    if max_tasks is not None:
        runnable = runnable[:max_tasks]

    def mark_running(task: dict[str, Any]) -> None:
        output = _task_output(run_root, task)
        task["status"] = "running"
        task["started_at"] = utc_now()
        task["attempts"] = int(task.get("attempts", 0)) + 1
        task.pop("error", None)
        write_json(output / "status.json", {"status": "running", "task": task})
        LOGGER.info("Running %s", task["task_id"])

    def finish_task(task: dict[str, Any], result: dict[str, Any] | None, exc: BaseException | None) -> None:
        output = _task_output(run_root, task)
        if exc is None and result is not None:
            task["status"] = "complete"
            task["completed_at"] = result["completed_at"]
            task["elapsed_seconds"] = result["elapsed_seconds"]
            write_json(output / "status.json", {"status": "complete", "task": task})
        else:
            assert exc is not None
            task["status"] = "failed"
            task["failed_at"] = utc_now()
            task["error"] = f"{type(exc).__name__}: {exc}"
            write_json(output / "status.json", {
                "status": "failed",
                "task": task,
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            })
            LOGGER.error("Task failed: %s: %s", task["task_id"], task["error"])
        manifest["updated_at"] = utc_now()
        write_json(run_root / "matrix_manifest.json", manifest)

    classical = [task for task in runnable if task["model"] in CLASSICAL_MODELS]
    neural = [task for task in runnable if task["model"] not in CLASSICAL_MODELS]
    workers = min(int(_matrix_settings(config)["classical_workers"]), max(len(classical), 1))
    threads_per_worker = int(_matrix_settings(config)["classical_threads_per_worker"])
    if classical:
        from threadpoolctl import threadpool_limits

        remaining = list(classical)
        warmed_families: set[str] = set()
        while remaining:
            first = remaining[0]
            family = "histogram" if first["representation"] in {"histogram", "rjsd_zscore"} else "de"
            cache = feature_cache_info()[family]
            if family not in warmed_families and int(cache["currsize"]) == 0:
                batch = [remaining.pop(0)]
                warmed_families.add(family)
            else:
                batch = remaining[:workers]
                del remaining[:workers]
            for task in batch:
                mark_running(task)
            manifest["updated_at"] = utc_now()
            write_json(run_root / "matrix_manifest.json", manifest)
            with threadpool_limits(limits=threads_per_worker):
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cmrd-classical") as executor:
                    futures = {
                        executor.submit(
                            run_fold_task, config, audit, protocol_hash, task, run_root,
                            torch.device("cpu"), fixed_epoch_override,
                        ): task
                        for task in batch
                    }
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            finish_task(task, future.result(), None)
                        except BaseException as exc:
                            finish_task(task, None, exc)

    for task in neural:
        mark_running(task)
        manifest["updated_at"] = utc_now()
        write_json(run_root / "matrix_manifest.json", manifest)
        try:
            result = run_fold_task(config, audit, protocol_hash, task, run_root, device, fixed_epoch_override)
            finish_task(task, result, None)
        except BaseException as exc:
            finish_task(task, None, exc)
    return matrix_status(run_root)


def invalidate_model_results(
    run_root: Path,
    dataset: str,
    model: str,
    reason: str,
) -> dict[str, Any]:
    """Remove invalid fold artifacts and reset only the selected model tasks."""
    if dataset not in EXPECTED_DATASETS:
        raise ValueError(f"Unexpected dataset: {dataset}")
    if model not in MODEL_NAMES:
        raise ValueError(f"Unexpected model: {model}")
    manifest_path = run_root / "matrix_manifest.json"
    manifest = read_json(manifest_path)
    selected = [
        task for task in manifest.get("tasks", {}).values()
        if task.get("dataset") == dataset and task.get("model") == model
    ]
    reset = []
    resolved_root = run_root.resolve()
    for task in selected:
        output = _task_output(run_root, task).resolve()
        if resolved_root not in output.parents:
            raise ValueError(f"Refusing to remove task output outside run root: {output}")
        if output.is_dir():
            shutil.rmtree(output)
        previous_status = task.get("status", "pending")
        for key in ("started_at", "completed_at", "failed_at", "elapsed_seconds", "error"):
            task.pop(key, None)
        task["status"] = "pending"
        task["attempts"] = 0
        reset.append({"task_id": task["task_id"], "previous_status": previous_status})
    manifest["updated_at"] = utc_now()
    write_json(manifest_path, manifest)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "model": model,
        "reason": reason,
        "reset_tasks": reset,
        "reset_at": utc_now(),
    }
    write_json(run_root / f"invalidation_{dataset}_{model}.json", audit)
    return audit


def matrix_status(run_root: Path) -> dict[str, Any]:
    path = run_root / "matrix_manifest.json"
    if not path.is_file():
        return {"status": "not_started", "run_root": str(run_root), "total": 0}
    manifest = read_json(path)
    tasks = list(manifest.get("tasks", {}).values())
    counts = {status: sum(task.get("status") == status for task in tasks) for status in ("pending", "running", "complete", "failed")}
    failed = [{"task_id": task["task_id"], "error": task.get("error")} for task in tasks if task.get("status") == "failed"]
    payload = {
        "status": "complete" if len(tasks) == 600 and counts["complete"] == 600 else "in_progress",
        "run_root": str(run_root),
        "declared": len(tasks),
        "expected_full_matrix_tasks": int(manifest.get("expected_full_matrix_tasks", 600)),
        **counts,
        "failed_tasks": failed,
        "updated_at": manifest.get("updated_at"),
    }
    write_json(run_root / "progress.json", payload)
    if failed:
        _write_csv(run_root / "failed_tasks.csv", failed)
    return payload


def _bootstrap_ci(values: np.ndarray, seed: int = FIXED_SEED, samples: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _holm_adjust(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    order = np.argsort(np.asarray(values))
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def summarize_matrix(run_root: Path, allow_partial: bool = False) -> dict[str, Any]:
    status = matrix_status(run_root)
    if not allow_partial and (status.get("declared") != 600 or status.get("complete") != 600):
        raise RuntimeError(f"Strict summary requires 600 complete tasks; status={status}")
    manifest = read_json(run_root / "matrix_manifest.json")
    rows: list[dict[str, Any]] = []
    protocol_hashes: dict[str, str] = {}
    fixed_epochs: dict[str, int] = {}
    seen: set[tuple[str, str, str, int, int]] = set()
    for task in manifest["tasks"].values():
        if task.get("status") != "complete":
            continue
        result_path = run_root / str(task["result_path"])
        result = read_json(result_path)
        key = (result["dataset"], result["representation"], result["model"], int(result["fold"]), int(result["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        previous = protocol_hashes.setdefault(result["dataset"], result["protocol_hash"])
        if previous != result["protocol_hash"]:
            raise ValueError(f"Mixed protocol hashes for {result['dataset']}")
        fixed_epoch = int(result["fixed_epoch"])
        previous_epoch = fixed_epochs.setdefault(result["dataset"], fixed_epoch)
        if previous_epoch != fixed_epoch:
            raise ValueError(f"Mixed fixed epochs for {result['dataset']}")
        if int(result.get("formal_source_subject_count", -1)) != 14 or result.get("diagnostic_smoke"):
            raise ValueError(f"Non-formal 14-source result in strict matrix: {result_path}")
        if (
            result.get("model") == "small_mlp"
            and result.get("model_input_adapter") != "trial_mean_std_source_zscore_v1"
        ):
            raise ValueError(f"Obsolete or unknown small MLP adapter in {result_path}")
        if result.get("model") == "linear_svm":
            options = result.get("classical_estimator_options", {})
            if float(options.get("tol", -1)) != 1e-3 or int(options.get("max_iter", -1)) != 5000:
                raise ValueError(f"Obsolete or non-convergent Linear SVM settings in {result_path}")
        metrics = result["target_test"]
        if not all(math.isfinite(float(metrics[name])) for name in ("accuracy", "balanced_accuracy", "macro_f1")):
            raise ValueError(f"Non-finite metrics in {result_path}")
        rows.append({
            "dataset": result["dataset"],
            "representation": result["representation"],
            "model": result["model"],
            "fold": int(result["fold"]),
            "seed": int(result["seed"]),
            "accuracy": float(metrics["accuracy"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
            "fixed_epoch": int(result["fixed_epoch"]),
            "formal_source_subject_count": int(result["formal_source_subject_count"]),
            "elapsed_seconds": float(result["elapsed_seconds"]),
            "protocol_hash": result["protocol_hash"],
        })
    _write_csv(run_root / "fold_results.csv", rows)
    grouped: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for representation in REPRESENTATIONS:
            for model in MODEL_NAMES:
                subset = [row for row in rows if row["dataset"] == dataset and row["representation"] == representation and row["model"] == model]
                if not subset:
                    continue
                grouped.append({
                    "dataset": dataset,
                    "representation": representation,
                    "model": model,
                    "folds": len(subset),
                    **{
                        f"{metric}_{suffix}": value
                        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                        for suffix, value in (
                            ("mean", float(np.mean([row[metric] for row in subset]))),
                            ("subject_std", float(np.std([row[metric] for row in subset], ddof=0))),
                        )
                    },
                })
    _write_csv(run_root / "condition_summary.csv", grouped)

    comparisons: list[dict[str, Any]] = []
    try:
        from scipy.stats import rankdata, wilcoxon
    except ModuleNotFoundError:
        rankdata = wilcoxon = None
    for dataset in sorted({row["dataset"] for row in rows}):
        for model in MODEL_NAMES:
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                rjsd = {row["fold"]: row[metric] for row in rows if row["dataset"] == dataset and row["representation"] == "rjsd_zscore" and row["model"] == model}
                de = {row["fold"]: row[metric] for row in rows if row["dataset"] == dataset and row["representation"] == "de_zscore" and row["model"] == model}
                common = sorted(set(rjsd) & set(de))
                if not common:
                    continue
                differences = np.asarray([rjsd[fold] - de[fold] for fold in common], dtype=np.float64)
                low, high = _bootstrap_ci(differences)
                nonzero = differences[differences != 0]
                if wilcoxon is not None and nonzero.size:
                    test = wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
                    p_value = float(test.pvalue)
                    ranks = rankdata(np.abs(nonzero))
                    positive = float(ranks[nonzero > 0].sum())
                    negative = float(ranks[nonzero < 0].sum())
                    rank_biserial = (positive - negative) / (positive + negative)
                else:
                    p_value = float("nan")
                    rank_biserial = float("nan")
                comparisons.append({
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "folds": len(common),
                    "rjsd_minus_de_mean": float(differences.mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "cohen_dz": float(differences.mean() / differences.std(ddof=1)) if differences.size > 1 and differences.std(ddof=1) > 0 else 0.0,
                    "rank_biserial": rank_biserial,
                    "wilcoxon_p": p_value,
                })
    families = sorted({(row["dataset"], row["metric"]) for row in comparisons})
    for dataset, metric in families:
        finite_indices = [
            index for index, row in enumerate(comparisons)
            if row["dataset"] == dataset
            and row["metric"] == metric
            and math.isfinite(float(row["wilcoxon_p"]))
        ]
        adjusted = _holm_adjust([float(comparisons[index]["wilcoxon_p"]) for index in finite_indices])
        for index, value in zip(finite_indices, adjusted):
            comparisons[index]["holm_p"] = value
    _write_csv(run_root / "paired_statistics.csv", comparisons)

    complexity = {name: index for index, name in enumerate(MODEL_NAMES)}
    shortlist = []
    for dataset in sorted({row["dataset"] for row in grouped}):
        rjsd_rows = [row for row in grouped if row["dataset"] == dataset and row["representation"] == "rjsd_zscore" and row["folds"] == 15]
        de_rows = [row for row in grouped if row["dataset"] == dataset and row["representation"] == "de_zscore" and row["folds"] == 15]
        if not rjsd_rows or not de_rows:
            continue
        best_rjsd = max(rjsd_rows, key=lambda row: row["accuracy_mean"])
        threshold = float(best_rjsd["accuracy_mean"]) - 0.01
        simplest = min((row for row in rjsd_rows if float(row["accuracy_mean"]) >= threshold), key=lambda row: complexity[row["model"]])
        best_de = max(de_rows, key=lambda row: row["accuracy_mean"])
        shortlist.append({
            "dataset": dataset,
            "status": "provisional_until_mechanism_evidence",
            "best_rjsd_model": best_rjsd["model"],
            "simplest_rjsd_within_one_point": simplest["model"],
            "de_zscore_comparator": best_de["model"],
            "seed": FIXED_SEED,
        })
    write_json(run_root / "candidate_shortlist.json", {"candidates": shortlist, "generated_at": utc_now()})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if status.get("complete") == 600 else "partial",
        "exploratory_single_seed": True,
        "seed": FIXED_SEED,
        "fold_results": len(rows),
        "protocol_hashes": protocol_hashes,
        "fixed_epochs": fixed_epochs,
        "conditions": grouped,
        "paired_statistics": comparisons,
        "candidate_shortlist": shortlist,
        "generated_at": utc_now(),
    }
    write_json(run_root / "summary.json", summary)
    return summary


def _scaled_samples(samples: Sequence[Any], enabled: bool) -> list[Any]:
    if not enabled:
        return list(samples)
    mean, std = scaling_statistics(samples, enabled)
    return [
        type(sample)(
            np.ascontiguousarray((sample.x - mean) / std, dtype=np.float32),
            sample.label,
            sample.subject,
            sample.session,
            sample.trial,
            sample.source_index,
        )
        for sample in samples
    ]


def _variance_and_separation(samples: Sequence[Any]) -> dict[str, float]:
    vectors = np.stack([sample.x.mean(axis=0, dtype=np.float32) for sample in samples]).astype(np.float64)
    subjects = np.asarray([sample.subject for sample in samples], dtype=np.int64)
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    grand = vectors.mean(axis=0)
    subject_means = np.stack([vectors[subjects == subject].mean(axis=0) for subject in np.unique(subjects)])
    between_subject = float(np.mean(np.square(subject_means - grand)))
    within_subject = float(np.mean([
        np.mean(np.square(vectors[subjects == subject] - vectors[subjects == subject].mean(axis=0)))
        for subject in np.unique(subjects)
    ]))
    class_means = np.stack([vectors[labels == label].mean(axis=0) for label in np.unique(labels)])
    between_class = float(np.mean(np.square(class_means - grand)))
    within_class = float(np.mean([
        np.mean(np.square(vectors[labels == label] - vectors[labels == label].mean(axis=0)))
        for label in np.unique(labels)
    ]))
    return {
        "between_subject_variance": between_subject,
        "within_subject_variance": within_subject,
        "subject_variance_ratio": between_subject / max(within_subject, 1e-12),
        "between_class_variance": between_class,
        "within_class_variance": within_class,
        "class_separation_ratio": between_class / max(within_class, 1e-12),
    }


def _source_probes(train_samples: Sequence[Any], validation_samples: Sequence[Any], classes: int) -> dict[str, Any]:
    train_vectors = pooled_vectors(train_samples)
    validation_vectors = pooled_vectors(validation_samples)
    train_labels = np.asarray([sample.label for sample in train_samples])
    validation_labels = np.asarray([sample.label for sample in validation_samples])
    emotion_probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=FIXED_SEED),
    )
    emotion_probe.fit(train_vectors, train_labels)
    emotion_predictions = emotion_probe.predict(validation_vectors)

    combined = list(train_samples) + list(validation_samples)
    subject_vectors = pooled_vectors(combined)
    subject_labels = np.asarray([sample.subject for sample in combined])
    indices = np.arange(len(combined))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=FIXED_SEED,
        stratify=subject_labels,
    )
    subject_probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=FIXED_SEED),
    )
    subject_probe.fit(subject_vectors[train_indices], subject_labels[train_indices])
    subject_predictions = subject_probe.predict(subject_vectors[test_indices])
    return {
        "emotion_probe_accuracy": float(accuracy_score(validation_labels, emotion_predictions)),
        "emotion_probe_chance": 1.0 / classes,
        "subject_probe_accuracy": float(accuracy_score(subject_labels[test_indices], subject_predictions)),
        "subject_probe_chance": 1.0 / len(np.unique(subject_labels)),
        "subject_probe_trials": int(test_indices.size),
    }


def _class_reference_sensitivity(cache_root: Path, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    references = {}
    for label in sorted({int(entry["label"]) for entry in entries}):
        reference, provenance = fit_reference(
            cache_root,
            [entry for entry in entries if int(entry["label"]) == label],
            "pooled_mean",
        )
        references[label] = {"reference": reference, "provenance": provenance}
    pairwise = []
    labels = sorted(references)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            difference = np.abs(references[left]["reference"] - references[right]["reference"])
            pairwise.append({
                "left_class": left,
                "right_class": right,
                "mean_absolute_difference": float(difference.mean()),
                "max_absolute_difference": float(difference.max()),
            })
    return {
        "construction": "source-label-conditioned reference bank; target label is never used",
        "classes": labels,
        "pairwise": pairwise,
    }


def _signed_histogram_rows(
    cache_root: Path,
    entries: Sequence[dict[str, Any]],
    dataset: str,
    fold: int,
) -> list[dict[str, Any]]:
    # Class-conditioned signed P-Q differences complement the unsigned RJSD.
    environment = read_json(cache_root / "environment.json")
    bands = list(environment["signature_payload"]["bands_hz"].items())
    overall, _ = fit_reference(cache_root, entries, "pooled_mean")
    rows: list[dict[str, Any]] = []
    for label in sorted({int(entry["label"]) for entry in entries}):
        class_reference, _ = fit_reference(
            cache_root,
            [entry for entry in entries if int(entry["label"]) == label],
            "pooled_mean",
        )
        signed = class_reference - overall
        for channel in range(signed.shape[0]):
            for band in range(signed.shape[1]):
                vector = signed[channel, band]
                band_name, limits = bands[band]
                low_hz, high_hz = map(float, limits)
                increase_bin = int(np.argmax(vector))
                decrease_bin = int(np.argmin(vector))
                rows.append({
                    "dataset": dataset,
                    "fold": fold,
                    "class": label,
                    "channel_index": channel,
                    "band_index": band,
                    "band_name": band_name,
                    "band_low_hz": low_hz,
                    "band_high_hz": high_hz,
                    "positive_mass": float(np.clip(vector, 0, None).sum()),
                    "negative_mass": float(np.clip(-vector, 0, None).sum()),
                    "largest_increase_bin": increase_bin,
                    "largest_increase_hz": low_hz + (increase_bin + 0.5) * (high_hz - low_hz) / vector.size,
                    "largest_increase": float(np.max(vector)),
                    "largest_decrease_bin": decrease_bin,
                    "largest_decrease_hz": low_hz + (decrease_bin + 0.5) * (high_hz - low_hz) / vector.size,
                    "largest_decrease": float(np.min(vector)),
                })
    return rows


@torch.no_grad()
def _attention_candidates(checkpoint_path: Path, samples: Sequence[Any], device: torch.device, top_k: int = 20) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(
        checkpoint["model"],
        int(checkpoint["input_dim"]),
        int(checkpoint["classes"]),
        max(int(checkpoint["max_length"]), max(sample.x.shape[0] for sample in samples)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = _loader(
        samples,
        np.asarray(checkpoint["normalization_mean"]),
        np.asarray(checkpoint["normalization_std"]),
        int(checkpoint["training"]["batch_size"]),
        False,
        int(checkpoint["seed"]),
    )
    model.eval()
    score_sum = np.zeros((62, 5), dtype=np.float64)
    weight_sum = 0.0
    for data, mask, _ in loader:
        valid_indices = _valid_indices_on_device(mask, device)
        _, attention = _forward_sequence_model(
            model,
            data.to(device),
            mask.to(device),
            valid_indices,
            return_attention=True,
        )
        band = attention["band"].detach().cpu().numpy()
        channel = attention["channel"].detach().cpu().numpy()
        valid = mask.numpy()[:, :, None, None]
        score_sum += (band * channel[..., None] * valid).sum(axis=(0, 1))
        weight_sum += float(valid.sum())
    scores = score_sum / max(weight_sum, 1.0)
    order = np.argsort(scores.reshape(-1))[::-1][:top_k]
    return [{
        "channel_index": int(index // 5),
        "band_index": int(index % 5),
        "attention_score": float(scores.reshape(-1)[index]),
    } for index in order]


def _perturbed_samples(
    samples: Sequence[Any],
    feature_index: int,
    method: str,
    seed: int,
    occlusion_value: float,
) -> list[Any]:
    rng = np.random.default_rng(seed)
    output = []
    for sample in samples:
        value = np.array(sample.x, dtype=np.float32, copy=True)
        if method == "occlusion":
            value[:, feature_index] = occlusion_value
        elif method == "permutation":
            value[:, feature_index] = value[rng.permutation(value.shape[0]), feature_index]
        else:
            raise ValueError(method)
        output.append(type(sample)(value, sample.label, sample.subject, sample.session, sample.trial, sample.source_index))
    return output


def _source_only_importance(
    checkpoint_path: Path,
    validation_samples: Sequence[Any],
    device: torch.device,
    dataset: str,
    fold: int,
) -> list[dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    normalization_mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    baseline, _ = evaluate_locked_checkpoint(checkpoint_path, validation_samples, device)
    candidates = _attention_candidates(checkpoint_path, validation_samples, device)
    rows = []
    for rank, candidate in enumerate(candidates, 1):
        feature_index = int(candidate["channel_index"]) * 5 + int(candidate["band_index"])
        row = {"dataset": dataset, "fold": fold, "rank": rank, **candidate}
        for method in ("occlusion", "permutation"):
            perturbed = _perturbed_samples(
                validation_samples,
                feature_index,
                method,
                FIXED_SEED + rank,
                float(normalization_mean[feature_index]),
            )
            metrics, _ = evaluate_locked_checkpoint(checkpoint_path, perturbed, device)
            row[f"{method}_accuracy_drop"] = float(baseline["accuracy"] - metrics["accuracy"])
            row[f"{method}_macro_f1_drop"] = float(baseline["macro_f1"] - metrics["macro_f1"])
        rows.append(row)
    return rows


def run_mechanism(
    config: ExperimentConfig,
    run_root: Path,
    cache_parent: Path | None,
    folds: Sequence[int] | None,
    representations: Sequence[str] | None,
) -> dict[str, Any]:
    audit = validate_fixed_cache(config, cache_parent)
    protocol_hash = _json_hash(protocol_payload(config, audit))
    cache_root = Path(audit["cache_root"])
    selected_folds = tuple(folds or EXPECTED_FOLDS)
    selected_representations = tuple(representations or REPRESENTATIONS)
    output_root = run_root / "mechanism" / config.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    signed_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    device = select_device(str(config.raw["training"].get("device", "auto")))
    for fold in selected_folds:
        train_entries, validation_entries, _ = _fold_source_entries(cache_root, fold)
        for representation in selected_representations:
            reference = None
            if representation == "rjsd_zscore":
                reference, _ = fit_reference(cache_root, train_entries, "pooled_mean")
            train_samples = load_representation_samples(cache_root, train_entries, representation, reference)
            validation_samples = load_representation_samples(cache_root, validation_entries, representation, reference)
            enabled = representation_uses_source_zscore(representation)
            scaled_train = _scaled_samples(train_samples, enabled)
            mean, std = scaling_statistics(train_samples, enabled)
            scaled_validation = (
                [
                    type(sample)(np.ascontiguousarray((sample.x - mean) / std, dtype=np.float32), sample.label, sample.subject, sample.session, sample.trial, sample.source_index)
                    for sample in validation_samples
                ]
                if enabled else list(validation_samples)
            )
            metrics = _variance_and_separation(scaled_train + scaled_validation)
            metrics.update(_source_probes(scaled_train, scaled_validation, int(config.raw["dataset"]["classes"])))
            rows.append({
                "dataset": config.dataset,
                "fold": fold,
                "representation": representation,
                "protocol_hash": protocol_hash,
                **metrics,
            })
            del train_samples, validation_samples, scaled_train, scaled_validation
            gc.collect()

        for method in REFERENCE_METHODS:
            sensitivity = reference_leave_one_subject_sensitivity(cache_root, train_entries, method)
            reference_rows.append({
                "dataset": config.dataset,
                "fold": fold,
                "reference_method": method,
                "mean_absolute_shift": sensitivity["mean_absolute_shift"],
                "max_absolute_shift": sensitivity["max_absolute_shift"],
            })
        class_bank = _class_reference_sensitivity(cache_root, train_entries)
        write_json(output_root / f"class_reference_bank_fold-{fold:02d}.json", class_bank)
        signed_rows.extend(_signed_histogram_rows(cache_root, train_entries, config.dataset, fold))

        matrix_output = run_root / config.dataset / "rjsd_zscore" / "hierarchical_attention" / f"fold-{fold:02d}" / f"seed-{FIXED_SEED}"
        checkpoint = matrix_output / "locked_source_model.pt"
        locked_reference_path = matrix_output / "locked_reference.npy"
        if not checkpoint.is_file() or not locked_reference_path.is_file():
            raise FileNotFoundError(
                f"Mechanism importance requires the completed RJSD hierarchical matrix task: {matrix_output}"
            )
        reference = np.load(locked_reference_path, allow_pickle=False)
        validation_samples = load_representation_samples(cache_root, validation_entries, "rjsd_zscore", reference)
        importance_rows.extend(_source_only_importance(checkpoint, validation_samples, device, config.dataset, fold))
        del validation_samples
        gc.collect()
    _write_csv(output_root / "mechanism_metrics.csv", rows)
    _write_csv(output_root / "reference_sensitivity.csv", reference_rows)
    _write_csv(output_root / "signed_spectrum.csv", signed_rows)
    _write_csv(output_root / "channel_band_importance.csv", importance_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset": config.dataset,
        "protocol_hash": protocol_hash,
        "source_only": True,
        "folds": list(selected_folds),
        "representations": list(selected_representations),
        "mechanism_rows": len(rows),
        "reference_rows": len(reference_rows),
        "signed_spectrum_rows": len(signed_rows),
        "importance_rows": len(importance_rows),
        "generated_at": utc_now(),
    }
    write_json(output_root / "mechanism_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict 15-fold source-only RJSD experiment runner")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def config_command(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--config", required=True)
        child.add_argument("--cache-parent")
        return child

    validate = config_command("validate-cache")
    validate.add_argument("--output")

    lock_epoch = config_command("lock-epoch")
    lock_epoch.add_argument("--run-root")

    for name in ("smoke", "matrix"):
        child = config_command(name)
        child.add_argument("--run-root")
        child.add_argument("--representation", action="append", choices=REPRESENTATIONS)
        child.add_argument("--model", action="append", choices=MODEL_NAMES)
        child.add_argument("--fold", action="append", type=int)
        child.add_argument("--resume", action="store_true")
        child.add_argument("--retry-failed", action="store_true")
        child.add_argument("--max-tasks", type=int)
        if name == "smoke":
            child.add_argument("--smoke-epoch", type=int, default=1)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-root", required=True)
    summarize.add_argument("--allow-partial", action="store_true")

    mechanism = config_command("mechanism")
    mechanism.add_argument("--run-root")
    mechanism.add_argument("--fold", action="append", type=int)
    mechanism.add_argument("--representation", action="append", choices=REPRESENTATIONS)

    invalidate = subparsers.add_parser("invalidate-model")
    invalidate.add_argument("--run-root", required=True)
    invalidate.add_argument("--dataset", required=True, choices=EXPECTED_DATASETS)
    invalidate.add_argument("--model", required=True, choices=MODEL_NAMES)
    invalidate.add_argument("--reason", required=True)
    return parser


def _resolve_run_root(config: ExperimentConfig | None, value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if config is None:
        raise ValueError("--run-root is required")
    return config.run_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.command == "status":
        payload = matrix_status(Path(args.run_root).expanduser().resolve())
    elif args.command == "summarize":
        payload = summarize_matrix(Path(args.run_root).expanduser().resolve(), args.allow_partial)
    elif args.command == "invalidate-model":
        payload = invalidate_model_results(
            Path(args.run_root).expanduser().resolve(), args.dataset, args.model, args.reason
        )
    else:
        config = load_config(args.config)
        cache_parent = Path(args.cache_parent).expanduser().resolve() if args.cache_parent else None
        if args.command == "validate-cache":
            payload = validate_fixed_cache(config, cache_parent)
            if args.output:
                write_json(Path(args.output).expanduser().resolve(), payload)
        elif args.command == "lock-epoch":
            payload = lock_dataset_epoch(config, _resolve_run_root(config, args.run_root), cache_parent)
        elif args.command in {"smoke", "matrix"}:
            run_root = _resolve_run_root(config, args.run_root)
            if args.command == "smoke":
                payload = run_matrix(
                    config,
                    run_root,
                    cache_parent,
                    args.representation or ["rjsd_zscore"],
                    args.model or ["small_mlp"],
                    args.fold or [1],
                    args.resume,
                    args.retry_failed,
                    args.max_tasks or 1,
                    args.smoke_epoch,
                )
            else:
                payload = run_matrix(
                    config,
                    run_root,
                    cache_parent,
                    args.representation,
                    args.model,
                    args.fold,
                    args.resume,
                    args.retry_failed,
                    args.max_tasks,
                    None,
                )
        else:
            payload = run_mechanism(
                config,
                _resolve_run_root(config, args.run_root),
                cache_parent,
                args.fold,
                args.representation,
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
