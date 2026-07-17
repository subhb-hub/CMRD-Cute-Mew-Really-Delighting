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
from cmrd.fixed_protocol import (
    FIXED_SEED,
    clear_feature_cache,
    evaluate_locked_checkpoint,
    fit_exploratory_monitored_source_model,
    fit_reference,
    load_representation_samples,
    representation_uses_source_zscore,
    scaling_statistics,
)
from cmrd.fixed_protocol_runner import validate_fixed_cache
from cmrd.io import read_json, write_json
from cmrd.training.runtime import select_device


LOGGER = logging.getLogger("cmrd.srjsd_large")
SCHEMA_VERSION = 1
EXPECTED_DATASETS = ("seed", "seediv")
EXPECTED_FOLDS = tuple(range(1, 16))
EXPECTED_CONDITIONS = (
    "a2_rjsd_base_v2",
    "b_srjsd_base_v2",
    "c_rjsd_large_v2",
    "d_srjsd_large_v2",
    "e_de_zscore_large_v2",
)


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
    raw = copy.deepcopy(config.raw.get("srjsd_large", {}))
    conditions = raw.get("conditions", {})
    architectures = raw.get("architectures", {})
    settings = {
        "seed": int(raw.get("seed", FIXED_SEED)),
        "max_epochs": int(raw.get("max_epochs", 200)),
        "target_monitor_interval": int(raw.get("target_monitor_interval", 10)),
        "conditions": conditions,
        "architectures": architectures,
        "expected_total_tasks": int(raw.get("expected_total_tasks", 150)),
    }
    if settings["seed"] != FIXED_SEED:
        raise ValueError("sRJSD-Large-v1 is frozen to seed 42")
    if settings["max_epochs"] < 1 or settings["target_monitor_interval"] < 1:
        raise ValueError("max_epochs and target_monitor_interval must be positive")
    if tuple(conditions) != EXPECTED_CONDITIONS:
        raise ValueError(f"Conditions must be declared in this order: {EXPECTED_CONDITIONS}")
    if set(architectures) != {"base", "large"}:
        raise ValueError("Architectures must contain exactly base and large")
    expected = {
        "a2_rjsd_base_v2": ("rjsd_zscore", "base"),
        "b_srjsd_base_v2": ("srjsd_zscore", "base"),
        "c_rjsd_large_v2": ("rjsd_zscore", "large"),
        "d_srjsd_large_v2": ("srjsd_zscore", "large"),
        "e_de_zscore_large_v2": ("de_zscore", "large"),
    }
    for name, pair in expected.items():
        active = conditions[name]
        if (str(active.get("representation")), str(active.get("architecture"))) != pair:
            raise ValueError(f"Condition {name} must be representation={pair[0]}, architecture={pair[1]}")
    for name, architecture in architectures.items():
        required = ("d_model", "heads", "layers", "feedforward", "dropout")
        if any(key not in architecture for key in required):
            raise KeyError(f"Architecture {name} is missing one of {required}")
        if int(architecture["d_model"]) % int(architecture["heads"]):
            raise ValueError(f"Architecture {name} d_model must be divisible by heads")
    if settings["expected_total_tasks"] != 150:
        raise ValueError("The frozen two-dataset experiment must contain 150 tasks")
    training = config.raw["training"]
    if float(training.get("warmup_fraction", -1)) != 0.1:
        raise ValueError("training.warmup_fraction is frozen to 0.1")
    if int(training.get("gradient_accumulation_steps", 0)) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    return settings


def protocol_payload(config: ExperimentConfig, cache_audit: dict[str, Any]) -> dict[str, Any]:
    settings = experiment_settings(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "family": "sRJSD-Large-v1",
        "dataset": config.dataset,
        "seed": settings["seed"],
        "window_seconds": 1.0,
        "hop_seconds": 1.0,
        "outer_protocol": "15-fold LOSO: all 14 non-target subjects train / one target subject",
        "reference_method": "pooled_mean from all 14 source subjects",
        "source_zscore": "fit once per fold and representation from all 14 sources",
        "signed_feature": "sign(centroid(P)-centroid(Q))*sqrt(JSD(P||Q)); ties use +1",
        "conditions": settings["conditions"],
        "architectures": settings["architectures"],
        "training": config.raw["training"],
        "max_epochs": settings["max_epochs"],
        "target_monitor_interval": settings["target_monitor_interval"],
        "target_monitoring": True,
        "target_metrics_affect_training": False,
        "checkpoint_selection": "predeclared final epoch only",
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "preprocessing_signature": cache_audit["preprocessing_signature"],
    }


def lock_experiment(config: ExperimentConfig, run_root: Path, cache_parent: Path | None) -> dict[str, Any]:
    audit = validate_fixed_cache(config, cache_parent)
    protocol = protocol_payload(config, audit)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": "sRJSD-Large-v1",
        "dataset": config.dataset,
        "max_epochs": experiment_settings(config)["max_epochs"],
        "target_monitor_interval": experiment_settings(config)["target_monitor_interval"],
        "target_metrics_affect_training": False,
        "checkpoint_selection": "predeclared final epoch only",
        "protocol_hash": _json_hash(protocol),
        "preprocessing_signature": audit["preprocessing_signature"],
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
        raise FileNotFoundError(f"Run the Lock stage before the matrix: {path}")
    lock = read_json(path)
    if lock.get("protocol_hash") != protocol_hash:
        raise ValueError(f"Experiment lock does not match the active protocol: {path}")
    if lock.get("target_metrics_affect_training") is not False:
        raise ValueError("The lock must forbid target-driven training decisions")


def task_identifier(dataset: str, condition: str, fold: int) -> str:
    return f"{dataset}__{condition}__fold-{fold:02d}__seed-{FIXED_SEED}"


def declared_tasks(
    config: ExperimentConfig,
    protocol_hash: str,
    conditions: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    settings = experiment_settings(config)
    selected_conditions = tuple(conditions or EXPECTED_CONDITIONS)
    selected_folds = tuple(folds or EXPECTED_FOLDS)
    if set(selected_conditions) - set(EXPECTED_CONDITIONS):
        raise ValueError("Unknown condition filter")
    if any(fold not in EXPECTED_FOLDS for fold in selected_folds):
        raise ValueError("Fold filter must be within 1..15")
    tasks = []
    # Fold-major order allows compact transformed features and normalization
    # statistics to be shared by Base and Large without retaining a whole
    # dataset of decompressed histograms beyond the fold.
    for fold in selected_folds:
        for condition in EXPECTED_CONDITIONS:
            if condition not in selected_conditions:
                continue
            definition = settings["conditions"][condition]
            representation = str(definition["representation"])
            identifier = task_identifier(config.dataset, condition, fold)
            tasks.append({
                "task_id": identifier,
                "dataset": config.dataset,
                "condition": condition,
                "representation": representation,
                "architecture": str(definition["architecture"]),
                "fold": fold,
                "seed": FIXED_SEED,
                "protocol_hash": protocol_hash,
                "status": "pending",
                "attempts": 0,
                "result_path": f"{config.dataset}/{condition}/fold-{fold:02d}/seed-{FIXED_SEED}/result.json",
            })
    return tasks


def _task_output(run_root: Path, task: dict[str, Any]) -> Path:
    return run_root / str(task["dataset"]) / str(task["condition"]) / f"fold-{int(task['fold']):02d}" / f"seed-{int(task['seed'])}"


def _fold_entries(cache_root: Path, fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(cache_root / "folds" / f"fold-{fold:02d}" / "manifest.json")
    source = list(manifest["groups"]["train"]) + list(manifest["groups"]["validation"])
    target = list(manifest["groups"]["test"])
    source_subjects = {int(entry["subject"]) for entry in source}
    target_subjects = {int(entry["subject"]) for entry in target}
    if len(source_subjects) != 14 or target_subjects != {fold} or fold in source_subjects:
        raise ValueError(f"Fold {fold} is not a valid 14-source/1-target LOSO fold")
    return source, target


def _prepare_bundle(
    config: ExperimentConfig,
    cache_root: Path,
    fold: int,
    representation: str,
) -> dict[str, Any]:
    source_entries, target_entries = _fold_entries(cache_root, fold)
    reference = None
    provenance = None
    if representation in {"rjsd_zscore", "srjsd_zscore"}:
        reference, provenance = fit_reference(cache_root, source_entries, "pooled_mean")
    bands_hz = [list(map(float, limits)) for limits in config.raw["signal"]["bands_hz"].values()]
    source_samples = load_representation_samples(
        cache_root, source_entries, representation, reference, bands_hz=bands_hz
    )
    scale_inputs = representation_uses_source_zscore(representation)
    normalization = scaling_statistics(source_samples, scale_inputs)
    source_locked_at = utc_now()
    # Deliberate exploratory target access. It occurs only after the source
    # reference is frozen, but before training so the fixed 10-epoch curve can
    # be evaluated. Labels never affect gradients or checkpoint selection.
    target_samples = load_representation_samples(
        cache_root, target_entries, representation, reference, bands_hz=bands_hz
    )
    return {
        "source_samples": source_samples,
        "target_samples": target_samples,
        "source_subjects": sorted({int(entry["subject"]) for entry in source_entries}),
        "reference": reference,
        "reference_provenance": provenance,
        "scale_inputs": scale_inputs,
        "normalization": normalization,
        "source_locked_at": source_locked_at,
        "target_loaded_at": utc_now(),
    }


def _model_config(config: ExperimentConfig, architecture_name: str) -> dict[str, Any]:
    architecture = experiment_settings(config)["architectures"][architecture_name]
    return {
        "name": "hierarchical_attention",
        "d_model": int(architecture["d_model"]),
        "heads": int(architecture["heads"]),
        "layers": int(architecture["layers"]),
        "feedforward": int(architecture["feedforward"]),
        "dropout": float(architecture["dropout"]),
        "architecture_label": architecture_name,
    }


def _training_config(config: ExperimentConfig, smoke_epochs: int | None = None) -> dict[str, Any]:
    training = copy.deepcopy(config.raw["training"])
    settings = experiment_settings(config)
    training["locked_epochs"] = int(smoke_epochs or settings["max_epochs"])
    training["target_monitor_interval"] = settings["target_monitor_interval"]
    training["epoch_policy"] = "predeclared_exploratory_final_epoch"
    training["checkpoint_selection"] = "fixed_final_epoch_only"
    training.pop("early_stopping_patience", None)
    return training


def run_task(
    config: ExperimentConfig,
    cache_audit: dict[str, Any],
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
    model_config = _model_config(config, str(task["architecture"]))
    checkpoint_path = output / "final_epoch_model.pt"
    if bundle["reference"] is not None:
        np.save(output / "locked_source_reference.npy", bundle["reference"], allow_pickle=False)
    events = [
        {
            "event": "source_reference_and_normalization_locked",
            "at": bundle["source_locked_at"],
            "source_subject_count": 14,
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
            task["task_id"],
            int(latest["epoch"]),
            float(latest["accuracy"]),
            float(latest["balanced_accuracy"]),
            float(latest["macro_f1"]),
        )

    training_result = fit_exploratory_monitored_source_model(
        bundle["source_samples"],
        bundle["target_samples"],
        model_config,
        training,
        int(config.raw["dataset"]["classes"]),
        device,
        checkpoint_path,
        seed=int(task["seed"]),
        scale_inputs=bool(bundle["scale_inputs"]),
        context={
            "family": "sRJSD-Large-v1",
            "dataset": config.dataset,
            "condition": task["condition"],
            "fold": task["fold"],
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
    write_json(output / "target_curve.json", {"curve": training_result["target_curve"], "updated_at": utc_now()})
    _write_csv(output / "predictions.csv", predictions)
    events.append({
        "event": "fixed_final_epoch_checkpoint_complete",
        "at": utc_now(),
        "final_epoch": int(training_result["final_epoch"]),
        "target_metrics_used_for_selection": False,
    })
    protocol_audit = {
        "schema_version": SCHEMA_VERSION,
        "family": "sRJSD-Large-v1",
        "task_id": task["task_id"],
        "protocol_hash": task["protocol_hash"],
        "outer_protocol": "15-fold LOSO",
        "formal_source_subjects": bundle["source_subjects"],
        "formal_source_subject_count": 14,
        "target_subject": int(task["fold"]),
        "target_monitoring_during_training": True,
        "target_monitor_interval": int(training["target_monitor_interval"]),
        "target_metrics_affect_training": False,
        "early_stopping": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "normalization_cache": "fold_representation_source_zscore_in_memory_v1",
        "events": events,
    }
    write_json(output / "protocol_audit.json", protocol_audit)
    result = {
        "schema_version": SCHEMA_VERSION,
        "family": "sRJSD-Large-v1",
        "status": "complete",
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "task_id": task["task_id"],
        "dataset": config.dataset,
        "condition": task["condition"],
        "representation": task["representation"],
        "architecture": task["architecture"],
        "fold": int(task["fold"]),
        "seed": int(task["seed"]),
        "protocol_hash": task["protocol_hash"],
        "preprocessing_signature": cache_audit["preprocessing_signature"],
        "source_zscore": bool(bundle["scale_inputs"]),
        "reference_method": "pooled_mean" if bundle["reference"] is not None else None,
        "locked_reference_provenance": bundle["reference_provenance"],
        "formal_source_subject_count": 14,
        "max_epochs": int(training["locked_epochs"]),
        "target_monitor_interval": int(training["target_monitor_interval"]),
        "target_metrics_affect_training": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "parameter_count": int(training_result["parameter_count"]),
        "effective_batch_size": int(training["batch_size"]) * int(training.get("gradient_accumulation_steps", 1)),
        "precision": training.get("precision", "float32"),
        "final_target_test": final_metrics,
        "target_curve_points": len(curve_rows),
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
        "family": "sRJSD-Large-v1",
        "created_at": utc_now(),
        "expected_full_matrix_tasks": 150,
        "protocols": {},
        "cache_audits": {},
        "tasks": {},
    }
    protocol_hash = _json_hash(protocol)
    manifest["protocols"][audit["dataset"]] = {"protocol_hash": protocol_hash, "payload": protocol}
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
    folds: Sequence[int] | None,
    resume: bool,
    retry_failed: bool,
    max_tasks: int | None = None,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    audit = validate_fixed_cache(config, cache_parent)
    protocol = protocol_payload(config, audit)
    protocol_hash = _json_hash(protocol)
    if smoke_epochs is None:
        _require_lock(config, run_root, protocol_hash)
    tasks = declared_tasks(config, protocol_hash, conditions, folds)
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_merge_manifest(run_root, tasks, audit, protocol)
    device = select_device(str(config.raw["training"].get("device", "auto")))
    runnable = []
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

    for fold in dict.fromkeys(int(task["fold"]) for task in runnable):
        fold_tasks = [task for task in runnable if int(task["fold"]) == fold]
        representations = tuple(dict.fromkeys(str(task["representation"]) for task in fold_tasks))
        # Materialize each compact representation once, then immediately drop
        # every decompressed 32-bin histogram before the long GPU training
        # starts.  Base/Large share these compact arrays and source statistics.
        bundles = {
            representation: _prepare_bundle(
                config, Path(audit["cache_root"]), fold, representation
            )
            for representation in representations
        }
        clear_feature_cache()
        gc.collect()
        for task in fold_tasks:
            output = _task_output(run_root, task)
            task["status"] = "running"
            task["started_at"] = utc_now()
            task["attempts"] = int(task.get("attempts", 0)) + 1
            task.pop("error", None)
            write_json(output / "status.json", {"status": "running", "task": task})
            manifest["updated_at"] = utc_now()
            write_json(run_root / "matrix_manifest.json", manifest)
            LOGGER.info("Running %s", task["task_id"])
            try:
                result = run_task(
                    config, audit, task, run_root, device,
                    bundles[str(task["representation"])], smoke_epochs,
                )
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
            manifest["updated_at"] = utc_now()
            write_json(run_root / "matrix_manifest.json", manifest)
        bundles.clear()
        gc.collect()
    return matrix_status(run_root)


def matrix_status(run_root: Path) -> dict[str, Any]:
    path = run_root / "matrix_manifest.json"
    if not path.is_file():
        return {"status": "not_started", "run_root": str(run_root), "declared": 0}
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
    payload = {
        "status": "complete" if len(tasks) == 150 and counts["complete"] == 150 else "in_progress",
        "family": "sRJSD-Large-v1",
        "evidence_status": "exploratory_target_monitored",
        "run_root": str(run_root),
        "declared": len(tasks),
        "expected_full_matrix_tasks": 150,
        **counts,
        "failed_tasks": failed,
        "updated_at": manifest.get("updated_at"),
    }
    write_json(run_root / "progress.json", payload)
    _write_csv(run_root / "failed_tasks.csv", failed)
    return payload


def _bootstrap_ci(values: np.ndarray, samples: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(FIXED_SEED)
    means = np.asarray([rng.choice(values, values.size, replace=True).mean() for _ in range(samples)])
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def _holm_adjust(values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def summarize(run_root: Path, allow_partial: bool = False) -> dict[str, Any]:
    status = matrix_status(run_root)
    if not allow_partial and (status.get("declared") != 150 or status.get("complete") != 150):
        raise RuntimeError(f"Strict summary requires 150 complete tasks; status={status}")
    manifest = read_json(run_root / "matrix_manifest.json")
    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    hashes: dict[str, str] = {}
    for task in manifest["tasks"].values():
        if task.get("status") != "complete":
            continue
        result_path = run_root / str(task["result_path"])
        result = read_json(result_path)
        key = (str(result["dataset"]), str(result["condition"]), int(result["fold"]))
        if key in seen:
            raise ValueError(f"Duplicate result: {key}")
        seen.add(key)
        previous = hashes.setdefault(str(result["dataset"]), str(result["protocol_hash"]))
        if previous != result["protocol_hash"]:
            raise ValueError(f"Mixed protocol hashes for {result['dataset']}")
        if int(result.get("formal_source_subject_count", -1)) != 14:
            raise ValueError(f"Invalid source subject count: {result_path}")
        metrics = result["final_target_test"]
        if not all(math.isfinite(float(metrics[name])) for name in ("accuracy", "balanced_accuracy", "macro_f1")):
            raise ValueError(f"Non-finite final metrics: {result_path}")
        rows.append({
            "dataset": result["dataset"],
            "condition": result["condition"],
            "representation": result["representation"],
            "architecture": result["architecture"],
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
                "fold": int(result["fold"]),
                "epoch": int(point["epoch"]),
                "accuracy": float(point["accuracy"]),
                "balanced_accuracy": float(point["balanced_accuracy"]),
                "macro_f1": float(point["macro_f1"]),
            })
    _write_csv(run_root / "fold_results.csv", rows)
    _write_csv(run_root / "target_curves.csv", curve_rows)
    curve_summary = []
    for dataset in sorted({row["dataset"] for row in curve_rows}):
        for condition in EXPECTED_CONDITIONS:
            epochs = sorted({
                row["epoch"] for row in curve_rows
                if row["dataset"] == dataset and row["condition"] == condition
            })
            for epoch in epochs:
                subset = [
                    row for row in curve_rows
                    if row["dataset"] == dataset
                    and row["condition"] == condition
                    and row["epoch"] == epoch
                ]
                curve_summary.append({
                    "dataset": dataset,
                    "condition": condition,
                    "epoch": epoch,
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
    _write_csv(run_root / "target_curve_summary.csv", curve_summary)
    grouped = []
    for dataset in sorted({row["dataset"] for row in rows}):
        for condition in EXPECTED_CONDITIONS:
            subset = [row for row in rows if row["dataset"] == dataset and row["condition"] == condition]
            if not subset:
                continue
            grouped.append({
                "dataset": dataset,
                "condition": condition,
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

    comparisons = []
    pairs = (
        ("a2_rjsd_base_v2", "b_srjsd_base_v2", "signed_transform_base"),
        ("a2_rjsd_base_v2", "c_rjsd_large_v2", "capacity_rjsd"),
        ("b_srjsd_base_v2", "d_srjsd_large_v2", "capacity_srjsd"),
        ("c_rjsd_large_v2", "d_srjsd_large_v2", "signed_transform_large"),
        ("e_de_zscore_large_v2", "d_srjsd_large_v2", "srjsd_vs_de_large"),
    )
    try:
        from scipy.stats import rankdata, wilcoxon
    except ModuleNotFoundError:
        rankdata = wilcoxon = None
    for dataset in sorted({row["dataset"] for row in rows}):
        for left, right, comparison in pairs:
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                left_values = {row["fold"]: row[metric] for row in rows if row["dataset"] == dataset and row["condition"] == left}
                right_values = {row["fold"]: row[metric] for row in rows if row["dataset"] == dataset and row["condition"] == right}
                common = sorted(set(left_values) & set(right_values))
                if not common:
                    continue
                differences = np.asarray([right_values[fold] - left_values[fold] for fold in common], dtype=np.float64)
                low, high = _bootstrap_ci(differences)
                p_value = float(wilcoxon(differences).pvalue) if wilcoxon is not None and np.any(differences != 0) else float("nan")
                nonzero = differences[differences != 0]
                if rankdata is not None and nonzero.size:
                    ranks = rankdata(np.abs(nonzero))
                    positive = float(ranks[nonzero > 0].sum())
                    negative = float(ranks[nonzero < 0].sum())
                    rank_biserial = (positive - negative) / (positive + negative)
                else:
                    rank_biserial = float("nan")
                comparisons.append({
                    "dataset": dataset,
                    "comparison": comparison,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "folds": len(common),
                    "right_minus_left_mean": float(differences.mean()),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wins": int(np.sum(differences > 0)),
                    "ties": int(np.sum(differences == 0)),
                    "losses": int(np.sum(differences < 0)),
                    "cohen_dz": float(differences.mean() / differences.std(ddof=1)) if differences.size > 1 and differences.std(ddof=1) > 0 else 0.0,
                    "rank_biserial": rank_biserial,
                    "wilcoxon_p": p_value,
                })
    for dataset, metric in sorted({(row["dataset"], row["metric"]) for row in comparisons}):
        indices = [
            index for index, row in enumerate(comparisons)
            if row["dataset"] == dataset
            and row["metric"] == metric
            and math.isfinite(float(row["wilcoxon_p"]))
        ]
        adjusted = _holm_adjust([float(comparisons[index]["wilcoxon_p"]) for index in indices])
        for index, value in zip(indices, adjusted):
            comparisons[index]["holm_p"] = value
    _write_csv(run_root / "paired_statistics.csv", comparisons)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "family": "sRJSD-Large-v1",
        "status": "complete" if status.get("complete") == 150 else "partial",
        "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence",
        "seed": FIXED_SEED,
        "fold_results": len(rows),
        "protocol_hashes": hashes,
        "conditions": grouped,
        "paired_statistics": comparisons,
        "generated_at": utc_now(),
    }
    write_json(run_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exploratory signed-sqrt RJSD Large experiment")
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
    for name in ("smoke", "matrix"):
        child = config_command(name)
        child.add_argument("--run-root", required=True)
        child.add_argument("--condition", action="append")
        child.add_argument("--fold", action="append", type=int)
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
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    if args.command == "status":
        payload = matrix_status(Path(args.run_root).expanduser().resolve())
    elif args.command == "summarize":
        payload = summarize(Path(args.run_root).expanduser().resolve(), args.allow_partial)
    else:
        config = load_config(args.config)
        experiment_settings(config)
        cache_parent = Path(args.cache_parent).expanduser().resolve() if args.cache_parent else None
        if args.command == "validate-cache":
            payload = validate_fixed_cache(config, cache_parent)
        elif args.command == "lock":
            payload = lock_experiment(config, Path(args.run_root).expanduser().resolve(), cache_parent)
        else:
            payload = run_matrix(
                config,
                Path(args.run_root).expanduser().resolve(),
                cache_parent,
                args.condition,
                args.fold,
                args.resume,
                args.retry_failed,
                args.max_tasks,
                args.smoke_epochs if args.command == "smoke" else None,
            )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
