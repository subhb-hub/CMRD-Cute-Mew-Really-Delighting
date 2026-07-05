from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrd.data.records import TrialSample
from cmrd.io import read_json, write_json
from cmrd.training.engine import train_once
from cmrd.training.runtime import environment_manifest, select_device


LOGGER = logging.getLogger("seediv.cmrd_ica.train")
FEATURES = ("cmrd", "de", "rjsd", "fusion")
EXPECTED_TRIALS = 15 * 3 * 24
EXPECTED_CHANNELS = 62
EXPECTED_BANDS = 5
EXPECTED_CLASSES = 4
DEFAULT_DATA_PARENT = (
    ROOT.parent / "Dataset" / "Processed" / "CMRD" / "seediv" / "de_rjsd_ica_1s_hop05"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _settings_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:12]


def _resolve_data_root(value: str | None) -> Path:
    requested = Path(value).expanduser().resolve() if value else DEFAULT_DATA_PARENT.resolve()
    if (requested / "pipeline_manifest.json").is_file():
        candidates = [requested]
    elif requested.is_dir():
        candidates = sorted(
            (
                path
                for path in requested.iterdir()
                if path.is_dir() and (path / "pipeline_manifest.json").is_file()
            ),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    else:
        candidates = []
    complete = [
        path
        for path in candidates
        if read_json(path / "pipeline_manifest.json").get("all_15_folds_complete") is True
    ]
    if not complete:
        raise FileNotFoundError(
            f"No complete SEED-IV DE+RJSD cache found under {requested}. "
            "Pass --data-root as the signature directory or its parent."
        )
    if len(complete) > 1 and value is None:
        LOGGER.warning("Multiple complete caches found; using newest: %s", complete[0])
    return complete[0]


def _metadata_value(archive: np.lib.npyio.NpzFile, key: str) -> int:
    return int(np.asarray(archive[key]).item())


def _check_entry_archive(
    archive: np.lib.npyio.NpzFile,
    entry: dict[str, Any],
    signature: str,
    feature_key: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    if feature_key not in archive.files:
        raise KeyError(f"Missing {feature_key!r} for {entry['trial_id']}")
    value = np.asarray(archive[feature_key], dtype=np.float32)
    if value.shape != expected_shape:
        raise ValueError(
            f"{entry['trial_id']} {feature_key} shape {value.shape} != {expected_shape}"
        )
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{entry['trial_id']} {feature_key} contains NaN/Inf")
    for key in ("label", "subject", "session", "trial", "source_index"):
        if _metadata_value(archive, key) != int(entry[key]):
            raise ValueError(f"{entry['trial_id']} has inconsistent {key}")
    stored_signature = str(np.asarray(archive["preprocessing_signature"]).item())
    if stored_signature != signature:
        raise ValueError(f"{entry['trial_id']} has preprocessing signature {stored_signature}")
    return value


def validate_cache(root: Path, deep: bool = False) -> dict[str, Any]:
    pipeline = read_json(root / "pipeline_manifest.json")
    signature = str(pipeline.get("preprocessing_signature", ""))
    if pipeline.get("dataset") != "SEED-IV" or pipeline.get("features") != ["de", "rjsd"]:
        raise ValueError("pipeline_manifest.json is not the expected SEED-IV DE+RJSD cache")
    if not pipeline.get("all_15_folds_complete"):
        raise RuntimeError("The preprocessing pipeline does not contain all 15 completed folds")

    trial_manifest = read_json(root / "trials_manifest.json")
    trials = list(trial_manifest.get("trials", []))
    if not trial_manifest.get("complete") or len(trials) != EXPECTED_TRIALS:
        raise RuntimeError(f"Expected {EXPECTED_TRIALS} complete trials, found {len(trials)}")
    if trial_manifest.get("failures"):
        raise RuntimeError(f"Preprocessing recorded failures: {trial_manifest['failures'][:3]}")
    if trial_manifest.get("preprocessing_signature") != signature:
        raise ValueError("Trial and pipeline preprocessing signatures differ")

    by_id = {str(entry["trial_id"]): entry for entry in trials}
    if len(by_id) != EXPECTED_TRIALS:
        raise ValueError("Duplicate trial IDs are present")
    expected_subject_counts = Counter(range(1, 16))
    actual_subject_counts = Counter(int(entry["subject"]) for entry in trials)
    if set(actual_subject_counts) != set(expected_subject_counts):
        raise ValueError(f"Unexpected subject IDs: {sorted(actual_subject_counts)}")
    if any(count != 72 for count in actual_subject_counts.values()):
        raise ValueError(f"Each subject must have 72 trials: {actual_subject_counts}")
    if Counter(int(entry["label"]) for entry in trials) != Counter({0: 270, 1: 270, 2: 270, 3: 270}):
        raise ValueError("Global SEED-IV labels are not balanced 270/270/270/270")

    window_counts = []
    for entry in trials:
        shape = tuple(map(int, entry["de_shape"]))
        p_shape = tuple(map(int, entry["p_hist_shape"]))
        if len(shape) != 3 or shape[1:] != (EXPECTED_CHANNELS, EXPECTED_BANDS):
            raise ValueError(f"Invalid DE shape in manifest for {entry['trial_id']}: {shape}")
        if p_shape != (*shape, 32):
            raise ValueError(f"Invalid p_hist shape in manifest for {entry['trial_id']}: {p_shape}")
        window_counts.append(shape[0])

    fold_summaries: dict[str, Any] = {}
    for target in range(1, 16):
        fold_name = f"fold-{target:02d}"
        fold_root = root / "folds" / fold_name
        manifest = read_json(fold_root / "manifest.json")
        if not manifest.get("complete") or manifest.get("preprocessing_signature") != signature:
            raise RuntimeError(f"{fold_name} is incomplete or has the wrong signature")
        if int(manifest.get("target_subject", -1)) != target:
            raise ValueError(f"{fold_name} has the wrong target subject")
        groups = manifest["groups"]
        expected_sizes = {"train": 864, "validation": 144, "test": 72}
        group_ids: dict[str, set[str]] = {}
        group_subjects: dict[str, set[int]] = {}
        for group_name, expected_size in expected_sizes.items():
            entries = list(groups[group_name])
            if len(entries) != expected_size:
                raise ValueError(f"{fold_name}/{group_name}: {len(entries)} != {expected_size}")
            group_ids[group_name] = {str(entry["trial_id"]) for entry in entries}
            group_subjects[group_name] = {int(entry["subject"]) for entry in entries}
            if Counter(int(entry["label"]) for entry in entries) != Counter(
                {label: expected_size // EXPECTED_CLASSES for label in range(EXPECTED_CLASSES)}
            ):
                raise ValueError(f"{fold_name}/{group_name} is not class-balanced")
        if group_ids["train"] & group_ids["validation"] or group_ids["train"] & group_ids["test"] or group_ids["validation"] & group_ids["test"]:
            raise ValueError(f"Trial leakage exists in {fold_name}")
        if set().union(*group_ids.values()) != set(by_id):
            raise ValueError(f"{fold_name} does not partition all trials")
        if group_subjects["test"] != {target}:
            raise ValueError(f"{fold_name} test group is not subject {target}")
        if len(group_subjects["train"]) != 12 or len(group_subjects["validation"]) != 2:
            raise ValueError(f"{fold_name} does not use a 12/2/1 subject split")
        if group_subjects["train"] & group_subjects["validation"] or target in group_subjects["train"] | group_subjects["validation"]:
            raise ValueError(f"Subject leakage exists in {fold_name}")

        provenance = manifest["provenance"]
        if provenance.get("reference_source") != "source_train_only":
            raise ValueError(f"{fold_name} RJSD reference is not source-training-only")
        train_indices = {int(entry["source_index"]) for entry in groups["train"]}
        if set(map(int, provenance["source_train_trial_indices"])) != train_indices:
            raise ValueError(f"{fold_name} reference provenance indices are inconsistent")

        with np.load(fold_root / "rjsd_reference.npz", allow_pickle=False) as archive:
            reference = np.asarray(archive["Q"], dtype=np.float32)
            if reference.shape != (EXPECTED_CHANNELS, EXPECTED_BANDS, 32):
                raise ValueError(f"{fold_name} reference shape is {reference.shape}")
            if not np.isfinite(reference).all() or np.any(reference < 0):
                raise FloatingPointError(f"{fold_name} reference is invalid")
            reference_error = float(np.max(np.abs(reference.sum(axis=-1) - 1.0)))
            if reference_error > 1e-5:
                raise ValueError(f"{fold_name} reference normalization error is {reference_error}")
            stored_subjects = set(map(int, np.asarray(archive["source_train_subjects"])))
            if stored_subjects != group_subjects["train"]:
                raise ValueError(f"{fold_name} reference subject list is inconsistent")

        fold_summaries[fold_name] = {
            "train_subjects": sorted(group_subjects["train"]),
            "validation_subjects": sorted(group_subjects["validation"]),
            "test_subject": target,
            "reference_windows": int(provenance["reference_window_count"]),
        }

    deep_stats: dict[str, Any] | None = None
    if deep:
        de_min, de_max = math.inf, -math.inf
        rjsd_min, rjsd_max = math.inf, -math.inf
        histogram_sum_error = 0.0
        for index, entry in enumerate(trials, 1):
            shape = tuple(map(int, entry["de_shape"]))
            with np.load(root / entry["path"], allow_pickle=False) as archive:
                de = _check_entry_archive(archive, entry, signature, "de", shape)
                histogram = np.asarray(archive["p_hist"], dtype=np.float32)
                if histogram.shape != (*shape, 32):
                    raise ValueError(f"{entry['trial_id']} p_hist shape is {histogram.shape}")
                if not np.isfinite(histogram).all() or np.any(histogram < 0):
                    raise FloatingPointError(f"{entry['trial_id']} p_hist is invalid")
                histogram_sum_error = max(
                    histogram_sum_error,
                    float(np.max(np.abs(histogram.sum(axis=-1) - 1.0))),
                )
                de_min = min(de_min, float(de.min()))
                de_max = max(de_max, float(de.max()))
            if index % 100 == 0:
                LOGGER.info("Deep validation: trial features %d/%d", index, len(trials))
        # Float16 p_hist values accumulate with a small quantization error.
        if histogram_sum_error > 5e-3:
            raise ValueError(f"p_hist normalization error is too large: {histogram_sum_error}")

        checked_rjsd = 0
        for target in range(1, 16):
            manifest = read_json(root / "folds" / f"fold-{target:02d}" / "manifest.json")
            for group_name in ("train", "validation", "test"):
                for entry in manifest["groups"][group_name]:
                    base = by_id[str(entry["trial_id"])]
                    shape = tuple(map(int, base["de_shape"]))
                    with np.load(root / entry["rjsd_path"], allow_pickle=False) as archive:
                        value = _check_entry_archive(archive, entry, signature, "rjsd", shape)
                        rjsd_min = min(rjsd_min, float(value.min()))
                        rjsd_max = max(rjsd_max, float(value.max()))
                        if np.any(value < -1e-7) or np.any(value > math.log(2.0) + 1e-5):
                            raise ValueError(f"{entry['trial_id']} RJSD falls outside [0, ln(2)]")
                    checked_rjsd += 1
            LOGGER.info("Deep validation: RJSD folds %d/15", target)
        deep_stats = {
            "checked_trial_archives": len(trials),
            "checked_rjsd_archives": checked_rjsd,
            "de_range": [de_min, de_max],
            "rjsd_range": [rjsd_min, rjsd_max],
            "max_p_hist_sum_error": histogram_sum_error,
        }

    metadata_files = list((root / "trial_metadata").glob("*.json"))
    fit_fallbacks = 0
    detection_errors = 0
    zero_exclusions = 0
    cleaned_std_over_100 = 0
    for path in metadata_files:
        metadata = read_json(path)["ica"]
        fit_fallbacks += bool(metadata.get("fit_errors"))
        detection_errors += bool(metadata.get("detection_errors"))
        zero_exclusions += not bool(metadata.get("excluded_components"))
        cleaned_std_over_100 += float(metadata["cleaned_std_microvolt"]) > 100.0

    return {
        "root": str(root),
        "preprocessing_signature": signature,
        "trials": len(trials),
        "folds": len(fold_summaries),
        "labels": dict(sorted(Counter(int(entry["label"]) for entry in trials).items())),
        "windows": {
            "min": min(window_counts),
            "max": max(window_counts),
            "mean": float(np.mean(window_counts)),
        },
        "ica": {
            "metadata_files": len(metadata_files),
            "primary_fit_fallbacks": fit_fallbacks,
            "detection_errors": detection_errors,
            "trials_with_zero_exclusions": zero_exclusions,
            "trials_with_cleaned_std_over_100_microvolt": cleaned_std_over_100,
        },
        "deep": deep_stats,
    }


def _fit_de_stats(root: Path, entries: Iterable[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros((EXPECTED_CHANNELS, EXPECTED_BANDS), dtype=np.float64)
    total_sq = np.zeros_like(total)
    count = 0
    for entry in entries:
        with np.load(root / entry["de_phist_path"], allow_pickle=False) as archive:
            value = np.asarray(archive["de"], dtype=np.float64)
        if value.ndim != 3 or value.shape[1:] != total.shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid DE tensor for {entry['trial_id']}: {value.shape}")
        total += value.sum(axis=0)
        total_sq += np.square(value).sum(axis=0)
        count += value.shape[0]
    if count == 0:
        raise ValueError("Cannot fit zDE statistics from an empty source-training split")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32), count


def _load_samples(
    root: Path,
    entries: Iterable[dict[str, Any]],
    feature: str,
    de_mean: np.ndarray | None,
    de_std: np.ndarray | None,
    alpha: float,
) -> list[TrialSample]:
    samples = []
    for entry in entries:
        de: np.ndarray | None = None
        rjsd: np.ndarray | None = None
        if feature in {"de", "cmrd", "fusion"}:
            with np.load(root / entry["de_phist_path"], allow_pickle=False) as archive:
                de = np.asarray(archive["de"], dtype=np.float32)
        if feature in {"rjsd", "cmrd", "fusion"}:
            with np.load(root / entry["rjsd_path"], allow_pickle=False) as archive:
                rjsd = np.asarray(archive["rjsd"], dtype=np.float32)
        if de is not None and rjsd is not None and de.shape != rjsd.shape:
            raise ValueError(f"DE/RJSD shape mismatch for {entry['trial_id']}: {de.shape} vs {rjsd.shape}")

        if feature == "de":
            assert de is not None
            value = de
        elif feature == "rjsd":
            assert rjsd is not None
            value = rjsd
        else:
            assert de is not None and rjsd is not None and de_mean is not None and de_std is not None
            zde = (de - de_mean[None]) / de_std[None]
            gated = np.tanh(float(alpha) * zde) * rjsd
            value = gated if feature == "cmrd" else np.concatenate((rjsd, gated, zde), axis=-1)
        if value.ndim != 3 or value.shape[1] != EXPECTED_CHANNELS or not np.isfinite(value).all():
            raise ValueError(f"Invalid {feature} tensor for {entry['trial_id']}: {value.shape}")
        samples.append(
            TrialSample(
                np.ascontiguousarray(value.reshape(value.shape[0], -1), dtype=np.float32),
                int(entry["label"]),
                int(entry["subject"]),
                int(entry["session"]),
                int(entry["trial"]),
                int(entry["source_index"]),
            )
        )
    return samples


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    test = result["test"]
    validation = result["validation"]
    return {
        "target_subject": int(result["target_subject"]),
        "seed": int(result["seed"]),
        "feature": result["feature"],
        "best_epoch": int(result["best_epoch"]),
        "validation_accuracy": float(validation["accuracy"]),
        "validation_macro_f1": float(validation["macro_f1"]),
        "test_accuracy": float(test["accuracy"]),
        "test_balanced_accuracy": float(test["balanced_accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_confusion_matrix": json.dumps(test["confusion_matrix"], separators=(",", ":")),
        "train_trials": int(result["train_trials"]),
        "validation_trials": int(result["validation_trials"]),
        "test_trials": int(result["test_trials"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = np.asarray([row["test_accuracy"] for row in rows], dtype=np.float64)
    macro_f1 = np.asarray([row["test_macro_f1"] for row in rows], dtype=np.float64)
    by_seed: dict[str, Any] = {}
    for seed in sorted({int(row["seed"]) for row in rows}):
        selected = [row for row in rows if int(row["seed"]) == seed]
        seed_accuracy = np.asarray([row["test_accuracy"] for row in selected], dtype=np.float64)
        seed_f1 = np.asarray([row["test_macro_f1"] for row in selected], dtype=np.float64)
        by_seed[str(seed)] = {
            "folds": len(selected),
            "test_accuracy_mean": float(seed_accuracy.mean()),
            "test_accuracy_subject_std": float(seed_accuracy.std(ddof=0)),
            "test_macro_f1_mean": float(seed_f1.mean()),
            "test_macro_f1_subject_std": float(seed_f1.std(ddof=0)),
        }
    return {
        "jobs": len(rows),
        "completed_subjects": sorted({int(row["target_subject"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "test_accuracy_mean_over_jobs": float(accuracies.mean()),
        "test_accuracy_std_over_jobs": float(accuracies.std(ddof=0)),
        "test_macro_f1_mean_over_jobs": float(macro_f1.mean()),
        "test_macro_f1_std_over_jobs": float(macro_f1.std(ddof=0)),
        "by_seed": by_seed,
    }


def train(args: argparse.Namespace, root: Path, validation: dict[str, Any]) -> Path:
    folds = sorted(set(args.fold or range(1, 16)))
    seeds = list(dict.fromkeys(args.seed))
    if any(not 1 <= fold <= 15 for fold in folds):
        raise ValueError("--fold values must be in 1..15")
    if args.d_model % args.nhead:
        raise ValueError("--d-model must be divisible by --nhead")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive")

    settings = {
        "schema_version": 1,
        "dataset": "SEED-IV",
        "data_root": str(root),
        "preprocessing_signature": validation["preprocessing_signature"],
        "feature": args.feature,
        "cmrd_definition": "tanh(alpha * source_train_zDE) * RJSD",
        "fusion_definition": "concat(RJSD, CMRD, source_train_zDE) along bands",
        "alpha": args.alpha,
        "folds": folds,
        "seeds": seeds,
        "model": {
            "name": "plain_masked_transformer",
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "feedforward": args.feedforward,
            "dropout": args.dropout,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "early_stopping_patience": args.early_stopping_patience,
            "gradient_clip_norm": args.gradient_clip_norm,
            "num_workers": args.num_workers,
            "pin_memory": not args.no_pin_memory,
            "deterministic": not args.non_deterministic,
            "device": args.device,
        },
        "selection_protocol": "best epoch by source-validation Macro-F1, accuracy tie-break; target evaluated once",
    }
    run_name = args.run_name or f"{args.feature}_{_settings_hash(settings)}"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else ROOT / "runs" / "seediv" / "de_rjsd_ica"
    run_root = output_root / run_name
    settings_path = run_root / "settings.json"
    if settings_path.is_file():
        if read_json(settings_path) != settings:
            raise RuntimeError(f"Existing run {run_root} has different settings; choose another --run-name")
        if not args.resume:
            raise FileExistsError(f"Run already exists: {run_root}. Pass --resume or choose --run-name.")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
        write_json(settings_path, settings)
        write_json(run_root / "environment.json", environment_manifest(sys.argv))
        write_json(run_root / "data_validation.json", validation)

    device = select_device(args.device)
    LOGGER.info("Device: %s", device)
    training = dict(settings["training"])
    model_config = dict(settings["model"])
    model_config.pop("name")
    completed_rows: list[dict[str, Any]] = []

    for target in folds:
        fold_name = f"fold-{target:02d}"
        manifest = read_json(root / "folds" / fold_name / "manifest.json")
        groups = manifest["groups"]
        de_mean: np.ndarray | None = None
        de_std: np.ndarray | None = None
        if args.feature in {"cmrd", "fusion"}:
            stats_path = run_root / "folds" / fold_name / "source_train_de_stats.npz"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            if stats_path.is_file():
                with np.load(stats_path, allow_pickle=False) as archive:
                    de_mean = np.asarray(archive["mean"], dtype=np.float32)
                    de_std = np.asarray(archive["std"], dtype=np.float32)
            else:
                LOGGER.info("Fold %02d: fitting source-training zDE statistics", target)
                de_mean, de_std, count = _fit_de_stats(root, groups["train"])
                np.savez_compressed(stats_path, mean=de_mean, std=de_std, windows=np.int64(count))

        pending_seeds = []
        for seed in seeds:
            result_path = run_root / "folds" / fold_name / f"seed-{seed}" / "result.json"
            if args.resume and result_path.is_file():
                LOGGER.info("Fold %02d seed %d: reusing completed result", target, seed)
                completed_rows.append(_flatten_result(read_json(result_path)))
            else:
                pending_seeds.append(seed)
        if not pending_seeds:
            continue

        LOGGER.info("Fold %02d: loading train/validation/test trials", target)
        train_samples = _load_samples(root, groups["train"], args.feature, de_mean, de_std, args.alpha)
        validation_samples = _load_samples(root, groups["validation"], args.feature, de_mean, de_std, args.alpha)
        test_samples = _load_samples(root, groups["test"], args.feature, de_mean, de_std, args.alpha)
        LOGGER.info(
            "Fold %02d: trials=%d/%d/%d input_dim=%d windows=%d..%d",
            target,
            len(train_samples),
            len(validation_samples),
            len(test_samples),
            train_samples[0].x.shape[1],
            min(sample.x.shape[0] for sample in train_samples + validation_samples + test_samples),
            max(sample.x.shape[0] for sample in train_samples + validation_samples + test_samples),
        )
        for seed in pending_seeds:
            output_dir = run_root / "folds" / fold_name / f"seed-{seed}"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            LOGGER.info("Fold %02d seed %d: training", target, seed)
            result = train_once(
                train_samples=train_samples,
                validation_samples=validation_samples,
                test_samples=test_samples,
                model_config=model_config,
                training=training,
                classes=EXPECTED_CLASSES,
                seed=seed,
                device=device,
                output_dir=output_dir,
                context={
                    "dataset": "SEED-IV",
                    "feature": args.feature,
                    "target_subject": target,
                    "preprocessing_signature": validation["preprocessing_signature"],
                    "alpha": args.alpha,
                    "split": manifest["split"],
                    "target_selection_used": False,
                },
            )
            row = _flatten_result(result)
            completed_rows.append(row)
            LOGGER.info(
                "Fold %02d seed %d complete: test ACC=%.4f Macro-F1=%.4f",
                target,
                seed,
                row["test_accuracy"],
                row["test_macro_f1"],
            )
            ordered = sorted(completed_rows, key=lambda item: (item["target_subject"], item["seed"]))
            _write_csv(run_root / "fold_results.csv", ordered)
            write_json(run_root / "summary.json", _aggregate(ordered))

    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a leakage-safe SEED-IV LOSO Transformer from the ICA-cleaned DE+RJSD cache. "
            "The default CMRD feature is tanh(alpha * source-train zDE) * RJSD."
        )
    )
    parser.add_argument("--data-root", help="Signature cache directory, or its de_rjsd_ica_1s_hop05 parent")
    parser.add_argument("--output-root", help="Training output parent (default: runs/seediv/de_rjsd_ica)")
    parser.add_argument("--run-name", help="Stable output directory name; defaults to a settings hash")
    parser.add_argument("--feature", choices=FEATURES, default="cmrd")
    parser.add_argument("--alpha", type=float, default=0.5, help="CMRD tanh gate strength")
    parser.add_argument("--fold", type=int, nargs="+", help="Target subject(s), default: all 1..15")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument("--resume", action="store_true", help="Reuse completed fold/seed jobs in the same run")
    parser.add_argument("--validate-only", action="store_true", help="Validate the cache and exit")
    parser.add_argument("--deep-validate", action="store_true", help="Read and numerically check every NPZ archive")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0, help="0 is the safest choice on Windows")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true", help="Allow faster non-deterministic CUDA kernels")

    parser.add_argument("--d-model", type=int, default=600)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = _resolve_data_root(args.data_root)
    LOGGER.info("Data root: %s", root)
    validation = validate_cache(root, deep=args.deep_validate)
    LOGGER.info(
        "Cache valid: trials=%d folds=%d labels=%s windows=%d..%d",
        validation["trials"],
        validation["folds"],
        validation["labels"],
        validation["windows"]["min"],
        validation["windows"]["max"],
    )
    LOGGER.info("ICA audit: %s", validation["ica"])
    if args.validate_only:
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return
    output = train(args, root, validation)
    LOGGER.info("Training outputs: %s", output)


if __name__ == "__main__":
    main()
