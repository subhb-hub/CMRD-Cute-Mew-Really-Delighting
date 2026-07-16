from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrd.data.records import TrialSample
from cmrd.features.rd import normalize_histograms, transform_rd
from cmrd.io import read_json, write_json
from cmrd.models import HierarchicalChannelBandTransformer, PlainTransformer
from cmrd.training.engine import SequenceDataset, collate_sequences, evaluate, fit_normalizer
from cmrd.training.runtime import environment_manifest, seed_everything, select_device


LOGGER = logging.getLogger("seediv.cmrd_ica.train")
FEATURES = ("cmrd", "de", "rjsd", "fusion")
MODELS = ("plain_transformer", "hierarchical_attention")
DATASET_NAME = "SEED-IV"
EXPECTED_SUBJECTS = 15
EXPECTED_CLASSES = 4
EXPECTED_TRIALS_PER_SUBJECT = 72
EXPECTED_TRIALS = EXPECTED_SUBJECTS * EXPECTED_TRIALS_PER_SUBJECT
EXPECTED_GROUP_SIZES = {"train": 864, "validation": 144, "test": 72}
EXPECTED_SOURCE_TRIALS = 1008
EXPECTED_TARGET_TRIALS = 72
EXPECTED_CHANNELS = 62
EXPECTED_BANDS = 5
DEFAULT_DATA_PARENT = (
    ROOT.parent / "Dataset" / "Processed" / "CMRD" / "seediv" / "de_rjsd_ica_1s_hop05"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "seediv" / "de_rjsd_ica"
ALL_SOURCE_STATS_ROOT = (
    ROOT / "runs" / "diagnostics" / "seediv_feature_tuning" / "_all_source_statistics"
)

CONFIG_FIELDS = {
    "paths": {
        "data_root": "data_root",
        "output_root": "output_root",
    },
    "experiment": {
        "run_name": "run_name",
        "feature": "feature",
        "alpha": "alpha",
        "folds": "fold",
        "seeds": "seed",
    },
    "model": {
        "name": "model",
        "d_model": "d_model",
        "nhead": "nhead",
        "layers": "layers",
        "channel_heads": "channel_heads",
        "temporal_heads": "temporal_heads",
        "temporal_layers": "temporal_layers",
        "feedforward": "feedforward",
        "dropout": "dropout",
    },
    "training": {
        "epochs": "epochs",
        "evaluation_interval": "evaluation_interval",
        "batch_size": "batch_size",
        "learning_rate": "learning_rate",
        "minimum_learning_rate": "minimum_learning_rate",
        "weight_decay": "weight_decay",
        "label_smoothing": "label_smoothing",
        "gradient_clip_norm": "gradient_clip_norm",
    },
    "runtime": {
        "device": "device",
        "num_workers": "num_workers",
        "resume": "resume",
        "validate_only": "validate_only",
        "deep_validate": "deep_validate",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _settings_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:12]


def _feature_name(value: str) -> str:
    feature = value.strip().lower()
    aliases = {"jsd": "rjsd", "de+rjsd": "fusion", "rjsd+de": "fusion"}
    return aliases.get(feature, feature)


def _config_defaults(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    unknown_sections = set(payload) - set(CONFIG_FIELDS)
    if unknown_sections:
        raise ValueError(f"Unknown config sections: {sorted(unknown_sections)}")

    defaults: dict[str, Any] = {}
    for section, values in payload.items():
        if not isinstance(values, dict):
            raise ValueError(f"Config section {section!r} must be a mapping")
        special_fields = {"pin_memory", "deterministic"} if section == "runtime" else set()
        unknown_fields = set(values) - set(CONFIG_FIELDS[section]) - special_fields
        if unknown_fields:
            raise ValueError(f"Unknown fields in config section {section!r}: {sorted(unknown_fields)}")
        for key, value in values.items():
            if value is None:
                continue
            if section == "runtime" and key == "pin_memory":
                defaults["no_pin_memory"] = not bool(value)
            elif section == "runtime" and key == "deterministic":
                defaults["non_deterministic"] = not bool(value)
            else:
                defaults[CONFIG_FIELDS[section][key]] = value

    if "feature" in defaults:
        defaults["feature"] = _feature_name(str(defaults["feature"]))
        if defaults["feature"] not in FEATURES:
            raise ValueError(f"Unknown feature in config: {defaults['feature']!r}")
    if "model" in defaults and defaults["model"] not in MODELS:
        raise ValueError(f"Unknown model in config: {defaults['model']!r}")
    for key in ("fold", "seed"):
        if key in defaults and not isinstance(defaults[key], list):
            raise ValueError(f"Config value {key!r} must be a YAML list")
    return defaults


def parse_args_with_config(
    argv: list[str] | None = None,
    parser: argparse.ArgumentParser | None = None,
) -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config")
    selected, _ = preliminary.parse_known_args(argv)
    parser = parser or build_parser()
    config_path = None
    if selected.config:
        config_path = Path(selected.config).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config does not exist: {config_path}")
        parser.set_defaults(**_config_defaults(config_path))
    args = parser.parse_args(argv)
    args.config = str(config_path) if config_path is not None else None
    return args


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
            f"No complete {DATASET_NAME} DE+RJSD cache found under {requested}. "
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
    if pipeline.get("dataset") != DATASET_NAME or pipeline.get("features") != ["de", "rjsd"]:
        raise ValueError(f"pipeline_manifest.json is not the expected {DATASET_NAME} DE+RJSD cache")
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
    expected_subject_counts = Counter(range(1, EXPECTED_SUBJECTS + 1))
    actual_subject_counts = Counter(int(entry["subject"]) for entry in trials)
    if set(actual_subject_counts) != set(expected_subject_counts):
        raise ValueError(f"Unexpected subject IDs: {sorted(actual_subject_counts)}")
    if any(count != EXPECTED_TRIALS_PER_SUBJECT for count in actual_subject_counts.values()):
        raise ValueError(
            f"Each subject must have {EXPECTED_TRIALS_PER_SUBJECT} trials: {actual_subject_counts}"
        )
    expected_labels = Counter({
        label: EXPECTED_TRIALS // EXPECTED_CLASSES for label in range(EXPECTED_CLASSES)
    })
    if Counter(int(entry["label"]) for entry in trials) != expected_labels:
        raise ValueError(f"Global {DATASET_NAME} labels are not balanced: expected {expected_labels}")

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
    for target in range(1, EXPECTED_SUBJECTS + 1):
        fold_name = f"fold-{target:02d}"
        fold_root = root / "folds" / fold_name
        manifest = read_json(fold_root / "manifest.json")
        if not manifest.get("complete") or manifest.get("preprocessing_signature") != signature:
            raise RuntimeError(f"{fold_name} is incomplete or has the wrong signature")
        if int(manifest.get("target_subject", -1)) != target:
            raise ValueError(f"{fold_name} has the wrong target subject")
        groups = manifest["groups"]
        expected_sizes = EXPECTED_GROUP_SIZES
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
        for target in range(1, EXPECTED_SUBJECTS + 1):
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
            LOGGER.info("Deep validation: RJSD folds %d/%d", target, EXPECTED_SUBJECTS)
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


def _fit_all_source_statistics(
    root: Path,
    entries: Iterable[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    de_total = np.zeros((EXPECTED_CHANNELS, EXPECTED_BANDS), dtype=np.float64)
    de_total_sq = np.zeros_like(de_total)
    reference_total = np.zeros((EXPECTED_CHANNELS, EXPECTED_BANDS, 32), dtype=np.float64)
    count = 0
    for entry in entries:
        with np.load(root / entry["de_phist_path"], allow_pickle=False) as archive:
            de = np.asarray(archive["de"], dtype=np.float64)
            histogram = normalize_histograms(np.asarray(archive["p_hist"], dtype=np.float32))
        if de.ndim != 3 or de.shape[1:] != de_total.shape or histogram.shape != (*de.shape, 32):
            raise ValueError(f"Invalid DE/p_hist tensors for {entry['trial_id']}")
        if not np.isfinite(de).all() or not np.isfinite(histogram).all():
            raise FloatingPointError(f"Non-finite all-source statistics input for {entry['trial_id']}")
        de_total += de.sum(axis=0)
        de_total_sq += np.square(de).sum(axis=0)
        reference_total += histogram.sum(axis=0, dtype=np.float64)
        count += de.shape[0]
    if count == 0:
        raise ValueError("Cannot fit all-source statistics from zero windows")
    de_mean = de_total / count
    variance = np.maximum(de_total_sq / count - np.square(de_mean), 0.0)
    de_std = np.sqrt(variance)
    de_std[de_std < 1e-6] = 1.0
    reference = normalize_histograms(reference_total / count)
    return de_mean.astype(np.float32), de_std.astype(np.float32), reference, count


def _load_samples(
    root: Path,
    entries: Iterable[dict[str, Any]],
    feature: str,
    de_mean: np.ndarray | None,
    de_std: np.ndarray | None,
    alpha: float,
    reference: np.ndarray | None = None,
) -> list[TrialSample]:
    samples = []
    for entry in entries:
        de: np.ndarray | None = None
        rjsd: np.ndarray | None = None
        if feature in {"de", "cmrd", "fusion"}:
            with np.load(root / entry["de_phist_path"], allow_pickle=False) as archive:
                de = np.asarray(archive["de"], dtype=np.float32)
        if feature in {"rjsd", "cmrd", "fusion"}:
            if reference is None:
                with np.load(root / entry["rjsd_path"], allow_pickle=False) as archive:
                    rjsd = np.asarray(archive["rjsd"], dtype=np.float32)
            else:
                with np.load(root / entry["de_phist_path"], allow_pickle=False) as archive:
                    histogram = np.asarray(archive["p_hist"], dtype=np.float32)
                flat = transform_rd(histogram, reference)
                rjsd = flat.reshape(flat.shape[0], EXPECTED_CHANNELS, EXPECTED_BANDS)
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


def _make_loader(
    samples: list[TrialSample],
    mean: np.ndarray,
    std: np.ndarray,
    training: dict[str, Any],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        SequenceDataset(samples, mean, std),
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True) and torch.cuda.is_available()),
        collate_fn=collate_sequences,
        generator=torch.Generator().manual_seed(seed),
    )


def _metrics_log(label: str, metrics: dict[str, Any]) -> str:
    per_class = ", ".join(
        f"c{row['class']}:P={row['precision']:.4f}/R={row['recall']:.4f}/F1={row['f1']:.4f}/N={row['support']}"
        for row in metrics["per_class"]
    )
    return (
        f"{label} ACC={metrics['accuracy']:.4f} BACC={metrics['balanced_accuracy']:.4f} "
        f"Macro-F1={metrics['macro_f1']:.4f} | {per_class} | CM={metrics['confusion_matrix']}"
    )


def _build_model(
    input_dim: int,
    max_length: int,
    model_config: dict[str, Any],
) -> nn.Module:
    name = str(model_config["name"])
    shared = {
        "input_dim": input_dim,
        "classes": EXPECTED_CLASSES,
        "max_length": max_length,
        "d_model": int(model_config["d_model"]),
        "feedforward": int(model_config["feedforward"]),
        "dropout": float(model_config["dropout"]),
    }
    if name == "plain_transformer":
        return PlainTransformer(
            **shared,
            nhead=int(model_config["nhead"]),
            layers=int(model_config["layers"]),
        )
    if name == "hierarchical_attention":
        return HierarchicalChannelBandTransformer(
            **shared,
            channels=int(model_config.get("channels", EXPECTED_CHANNELS)),
            channel_heads=int(model_config["channel_heads"]),
            temporal_heads=int(model_config["temporal_heads"]),
            temporal_layers=int(model_config["temporal_layers"]),
        )
    raise ValueError(f"Unknown model: {name!r}")


def _train_all_source_once(
    train_samples: list[TrialSample],
    test_samples: list[TrialSample],
    model_config: dict[str, Any],
    training: dict[str, Any],
    seed: int,
    device: torch.device,
    output_dir: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(seed, bool(training.get("deterministic", True)))
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = fit_normalizer(train_samples)
    input_dim = train_samples[0].x.shape[1]
    max_length = max(sample.x.shape[0] for sample in train_samples + test_samples)
    model = _build_model(input_dim, max_length, model_config).to(device)
    train_loader = _make_loader(train_samples, mean, std, training, True, seed)
    source_eval_loader = _make_loader(train_samples, mean, std, training, False, seed)
    test_loader = _make_loader(test_samples, mean, std, training, False, seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    evaluation_interval = int(training.get("evaluation_interval", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training.get("minimum_learning_rate", 1e-6)),
    )
    history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    source_metrics = None
    test_metrics = None
    started = time.perf_counter()
    log_path = output_dir / "train.log"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with log_path.open("w", encoding="utf-8") as log:
        header = {
            **context,
            "seed": seed,
            "device": str(device),
            "input_dim": input_dim,
            "max_length": max_length,
            "model_parameters": parameter_count,
            "train_trials": len(train_samples),
            "test_trials": len(test_samples),
        }
        log.write(json.dumps(header, ensure_ascii=False) + "\n")
        LOGGER.info(
            "Fold %02d seed %d model parameters=%d input_dim=%d max_length=%d",
            int(context["target_subject"]), seed, parameter_count, input_dim, max_length,
        )
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            for data, mask, labels in train_loader:
                data = data.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(data, mask), labels)
                loss.backward()
                clip = float(training.get("gradient_clip_norm", 0.0))
                if clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                loss_sum += float(loss.item()) * labels.shape[0]
                seen += labels.shape[0]
            scheduler.step()
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            LOGGER.info(
                "Fold %02d seed %d epoch %03d/%03d loss=%.6f lr=%.8g",
                int(context["target_subject"]), seed, epoch, epochs,
                row["train_loss"], row["learning_rate"],
            )

            should_evaluate = epoch == epochs or (
                evaluation_interval > 0 and epoch % evaluation_interval == 0
            )
            if should_evaluate:
                source_metrics = evaluate(model, source_eval_loader, device, EXPECTED_CLASSES)
                test_metrics = evaluate(model, test_loader, device, EXPECTED_CLASSES)
                evaluation_row = {
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "source_accuracy": source_metrics["accuracy"],
                    "source_balanced_accuracy": source_metrics["balanced_accuracy"],
                    "source_macro_f1": source_metrics["macro_f1"],
                    "target_accuracy": test_metrics["accuracy"],
                    "target_balanced_accuracy": test_metrics["balanced_accuracy"],
                    "target_macro_f1": test_metrics["macro_f1"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
                evaluation_history.append(evaluation_row)
                source_line = _metrics_log(f"SOURCE epoch={epoch:03d}", source_metrics)
                test_line = _metrics_log(f"TARGET epoch={epoch:03d}", test_metrics)
                log.write(json.dumps(evaluation_row) + "\n")
                log.write(source_line + "\n" + test_line + "\n")
                log.flush()
                LOGGER.info(source_line)
                LOGGER.info(test_line)

    if source_metrics is None or test_metrics is None:
        raise RuntimeError("Final source/target evaluation was not produced")

    _write_csv(output_dir / "epochs.csv", history)
    _write_csv(output_dir / "evaluations.csv", evaluation_history)
    checkpoint = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalization_mean": mean,
        "normalization_std": std,
        "model": model_config,
        "training": training,
        "context": context,
        "seed": seed,
        "final_epoch": epochs,
        "input_dim": input_dim,
        "max_length": max_length,
        "model_parameters": parameter_count,
    }
    torch.save(checkpoint, output_dir / "final.pt")
    result = {
        **context,
        "seed": seed,
        "model": model_config,
        "model_parameters": parameter_count,
        "training": training,
        "final_epoch": epochs,
        "source": source_metrics,
        "test": test_metrics,
        "train_trials": len(train_samples),
        "test_trials": len(test_samples),
        "elapsed_seconds": time.perf_counter() - started,
        "target_evaluated_during_training": any(
            int(row["epoch"]) < epochs for row in evaluation_history
        ),
    }
    write_json(output_dir / "result.json", result)
    return result


def _flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    source = result["source"]
    test = result["test"]
    return {
        "target_subject": int(result["target_subject"]),
        "seed": int(result["seed"]),
        "feature": result["feature"],
        "final_epoch": int(result["final_epoch"]),
        "model_parameters": int(result["model_parameters"]),
        "source_accuracy": float(source["accuracy"]),
        "source_balanced_accuracy": float(source["balanced_accuracy"]),
        "source_macro_f1": float(source["macro_f1"]),
        "source_confusion_matrix": json.dumps(source["confusion_matrix"], separators=(",", ":")),
        "test_accuracy": float(test["accuracy"]),
        "test_balanced_accuracy": float(test["balanced_accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_confusion_matrix": json.dumps(test["confusion_matrix"], separators=(",", ":")),
        "train_trials": int(result["train_trials"]),
        "test_trials": int(result["test_trials"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
    }


def _per_class_from_confusion(confusion: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for label in range(confusion.shape[0]):
        true_positive = float(confusion[label, label])
        support = float(confusion[label].sum())
        predicted = float(confusion[:, label].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append({
            "class": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support),
        })
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_accuracies = np.asarray([row["source_accuracy"] for row in rows], dtype=np.float64)
    source_macro_f1 = np.asarray([row["source_macro_f1"] for row in rows], dtype=np.float64)
    accuracies = np.asarray([row["test_accuracy"] for row in rows], dtype=np.float64)
    balanced_accuracies = np.asarray([row["test_balanced_accuracy"] for row in rows], dtype=np.float64)
    macro_f1 = np.asarray([row["test_macro_f1"] for row in rows], dtype=np.float64)
    source_confusion = sum(
        (np.asarray(json.loads(row["source_confusion_matrix"]), dtype=np.int64) for row in rows),
        start=np.zeros((EXPECTED_CLASSES, EXPECTED_CLASSES), dtype=np.int64),
    )
    test_confusion = sum(
        (np.asarray(json.loads(row["test_confusion_matrix"]), dtype=np.int64) for row in rows),
        start=np.zeros((EXPECTED_CLASSES, EXPECTED_CLASSES), dtype=np.int64),
    )
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
    by_target: dict[str, Any] = {}
    for target in sorted({int(row["target_subject"]) for row in rows}):
        selected = [row for row in rows if int(row["target_subject"]) == target]
        target_accuracy = np.asarray([row["test_accuracy"] for row in selected], dtype=np.float64)
        target_f1 = np.asarray([row["test_macro_f1"] for row in selected], dtype=np.float64)
        by_target[f"{target:02d}"] = {
            "seeds": len(selected),
            "test_accuracy_mean": float(target_accuracy.mean()),
            "test_accuracy_seed_std": float(target_accuracy.std(ddof=0)),
            "test_macro_f1_mean": float(target_f1.mean()),
            "test_macro_f1_seed_std": float(target_f1.std(ddof=0)),
        }
    return {
        "jobs": len(rows),
        "completed_subjects": sorted({int(row["target_subject"]) for row in rows}),
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "source_accuracy_mean_over_jobs": float(source_accuracies.mean()),
        "source_accuracy_std_over_jobs": float(source_accuracies.std(ddof=0)),
        "source_macro_f1_mean_over_jobs": float(source_macro_f1.mean()),
        "source_macro_f1_std_over_jobs": float(source_macro_f1.std(ddof=0)),
        "test_accuracy_mean_over_jobs": float(accuracies.mean()),
        "test_accuracy_std_over_jobs": float(accuracies.std(ddof=0)),
        "test_balanced_accuracy_mean_over_jobs": float(balanced_accuracies.mean()),
        "test_balanced_accuracy_std_over_jobs": float(balanced_accuracies.std(ddof=0)),
        "test_macro_f1_mean_over_jobs": float(macro_f1.mean()),
        "test_macro_f1_std_over_jobs": float(macro_f1.std(ddof=0)),
        "aggregate_source_confusion_matrix": source_confusion.tolist(),
        "aggregate_source_per_class": _per_class_from_confusion(source_confusion),
        "aggregate_test_confusion_matrix": test_confusion.tolist(),
        "aggregate_test_per_class": _per_class_from_confusion(test_confusion),
        "by_seed": by_seed,
        "by_target_subject": by_target,
    }


def _write_run_results(run_root: Path, results: list[dict[str, Any]]) -> None:
    ordered_results = sorted(results, key=lambda item: (int(item["target_subject"]), int(item["seed"])))
    rows = [_flatten_result(result) for result in ordered_results]
    _write_csv(run_root / "fold_results.csv", rows)
    write_json(run_root / "detailed_results.json", ordered_results)
    write_json(run_root / "summary.json", _aggregate(rows))


def train(args: argparse.Namespace, root: Path, validation: dict[str, Any]) -> Path:
    folds = sorted(set(args.fold or range(1, EXPECTED_SUBJECTS + 1)))
    seeds = list(dict.fromkeys(args.seed))
    if any(not 1 <= fold <= EXPECTED_SUBJECTS for fold in folds):
        raise ValueError(f"--fold values must be in 1..{EXPECTED_SUBJECTS}")
    if args.model == "plain_transformer" and args.d_model % args.nhead:
        raise ValueError("--d-model must be divisible by --nhead")
    if args.model == "hierarchical_attention":
        if args.d_model % args.channel_heads:
            raise ValueError("--d-model must be divisible by --channel-heads")
        if args.d_model % args.temporal_heads:
            raise ValueError("--d-model must be divisible by --temporal-heads")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive")
    if args.evaluation_interval < 0:
        raise ValueError("--evaluation-interval must be >= 0")

    settings = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "data_root": str(root),
        "preprocessing_signature": validation["preprocessing_signature"],
        "feature": args.feature,
        "cmrd_definition": "tanh(alpha * source_train_zDE) * RJSD",
        "fusion_definition": "concat(RJSD, CMRD, source_train_zDE) along bands",
        "alpha": args.alpha,
        "folds": folds,
        "seeds": seeds,
        "model": (
            {
                "name": "plain_transformer",
                "d_model": args.d_model,
                "nhead": args.nhead,
                "layers": args.layers,
                "feedforward": args.feedforward,
                "dropout": args.dropout,
            }
            if args.model == "plain_transformer"
            else {
                "name": "hierarchical_attention",
                "channels": EXPECTED_CHANNELS,
                "d_model": args.d_model,
                "channel_heads": args.channel_heads,
                "temporal_heads": args.temporal_heads,
                "temporal_layers": args.temporal_layers,
                "feedforward": args.feedforward,
                "dropout": args.dropout,
            }
        ),
        "training": {
            "epochs": args.epochs,
            "evaluation_interval": args.evaluation_interval,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "gradient_clip_norm": args.gradient_clip_norm,
            "num_workers": args.num_workers,
            "pin_memory": not args.no_pin_memory,
            "deterministic": not args.non_deterministic,
            "device": args.device,
        },
        "split_protocol": "14-source-subject train / 1-target-subject test",
        "selection_protocol": (
            "fixed epoch count; periodic source/target monitoring does not alter training or model selection"
        ),
    }
    run_name = args.run_name or f"{args.feature}_{_settings_hash(settings)}"
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else DEFAULT_OUTPUT_ROOT
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
        if args.config:
            shutil.copyfile(Path(args.config), run_root / "input_config.yaml")

    run_log = run_root / "run.log"
    if not any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == run_log
        for handler in LOGGER.handlers
    ):
        file_handler = logging.FileHandler(run_log, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        LOGGER.addHandler(file_handler)

    device = select_device(args.device)
    LOGGER.info("Device: %s", device)
    training = dict(settings["training"])
    model_config = dict(settings["model"])
    completed_results: list[dict[str, Any]] = []

    for target in folds:
        fold_name = f"fold-{target:02d}"
        manifest = read_json(root / "folds" / fold_name / "manifest.json")
        groups = manifest["groups"]
        source_entries = sorted(
            list(groups["train"]) + list(groups["validation"]),
            key=lambda entry: int(entry["source_index"]),
        )
        test_entries = list(groups["test"])
        source_subjects = sorted({int(entry["subject"]) for entry in source_entries})
        if len(source_entries) != EXPECTED_SOURCE_TRIALS or len(source_subjects) != 14:
            raise RuntimeError(f"Fold {target:02d} does not contain 14 complete source subjects")
        if len(test_entries) != EXPECTED_TARGET_TRIALS or {int(entry["subject"]) for entry in test_entries} != {target}:
            raise RuntimeError(f"Fold {target:02d} target split is invalid")

        de_mean: np.ndarray | None = None
        de_std: np.ndarray | None = None
        reference: np.ndarray | None = None
        source_window_count = 0
        if args.feature in {"rjsd", "cmrd", "fusion"}:
            stats_path = run_root / "folds" / fold_name / "all_source_statistics.npz"
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            if stats_path.is_file():
                with np.load(stats_path, allow_pickle=False) as archive:
                    de_mean = np.asarray(archive["de_mean"], dtype=np.float32)
                    de_std = np.asarray(archive["de_std"], dtype=np.float32)
                    reference = np.asarray(archive["rjsd_reference"], dtype=np.float32)
                    source_window_count = int(archive["source_window_count"])
                    cached_subjects = list(map(int, archive["source_subjects"]))
                if cached_subjects != source_subjects:
                    raise RuntimeError(f"Fold {target:02d} all-source statistics subject mismatch")
                LOGGER.info("Fold %02d: reusing all-source DE/RJSD statistics", target)
            else:
                shared_path = (
                    ALL_SOURCE_STATS_ROOT
                    / validation["preprocessing_signature"]
                    / f"fold-{target:02d}.npz"
                )
                if shared_path.is_file():
                    with np.load(shared_path, allow_pickle=False) as archive:
                        de_mean = np.asarray(archive["de_mean"], dtype=np.float32)
                        de_std = np.asarray(archive["de_std"], dtype=np.float32)
                        reference = np.asarray(archive["rjsd_reference"], dtype=np.float32)
                        source_window_count = int(archive["source_window_count"])
                        cached_subjects = list(map(int, archive["source_subjects"]))
                    if cached_subjects != source_subjects:
                        raise RuntimeError(f"Shared fold {target:02d} statistics subject mismatch")
                    LOGGER.info("Fold %02d: importing verified all-source statistics cache", target)
                else:
                    LOGGER.info("Fold %02d: fitting all-source DE/RJSD statistics", target)
                    de_mean, de_std, reference, source_window_count = _fit_all_source_statistics(root, source_entries)
                np.savez_compressed(
                    stats_path,
                    de_mean=de_mean,
                    de_std=de_std,
                    rjsd_reference=reference,
                    source_window_count=np.int64(source_window_count),
                    source_subjects=np.asarray(source_subjects, dtype=np.int64),
                    preprocessing_signature=np.asarray(validation["preprocessing_signature"]),
                )
            if args.feature == "rjsd":
                de_mean = de_std = None

        pending_seeds = []
        for seed in seeds:
            result_path = run_root / "folds" / fold_name / f"seed-{seed}" / "result.json"
            if args.resume and result_path.is_file():
                LOGGER.info("Fold %02d seed %d: reusing completed result", target, seed)
                completed_results.append(read_json(result_path))
            else:
                pending_seeds.append(seed)
        if not pending_seeds:
            continue

        LOGGER.info("Fold %02d: loading 14-source train and 1-target test trials", target)
        train_samples = _load_samples(
            root, source_entries, args.feature, de_mean, de_std, args.alpha, reference=reference
        )
        test_samples = _load_samples(
            root, test_entries, args.feature, de_mean, de_std, args.alpha, reference=reference
        )
        LOGGER.info(
            "Fold %02d: train/test trials=%d/%d subjects=14/1 input_dim=%d windows=%d..%d",
            target,
            len(train_samples),
            len(test_samples),
            train_samples[0].x.shape[1],
            min(sample.x.shape[0] for sample in train_samples + test_samples),
            max(sample.x.shape[0] for sample in train_samples + test_samples),
        )
        for seed in pending_seeds:
            output_dir = run_root / "folds" / fold_name / f"seed-{seed}"
            if output_dir.exists():
                shutil.rmtree(output_dir)
            LOGGER.info("Fold %02d seed %d: training", target, seed)
            result = _train_all_source_once(
                train_samples=train_samples,
                test_samples=test_samples,
                model_config=model_config,
                training=training,
                seed=seed,
                device=device,
                output_dir=output_dir,
                context={
                    "dataset": DATASET_NAME,
                    "feature": args.feature,
                    "target_subject": target,
                    "preprocessing_signature": validation["preprocessing_signature"],
                    "alpha": args.alpha,
                    "split": {
                        "protocol": "14-source-train/1-target-test",
                        "train_subjects": source_subjects,
                        "target_subject": target,
                    },
                    "source_train_windows": source_window_count,
                    "target_selection_used": False,
                },
            )
            completed_results.append(result)
            row = _flatten_result(result)
            LOGGER.info(
                "Fold %02d seed %d complete: test ACC=%.4f Macro-F1=%.4f",
                target,
                seed,
                row["test_accuracy"],
                row["test_macro_f1"],
            )
            _write_run_results(run_root, completed_results)

    if completed_results:
        _write_run_results(run_root, completed_results)
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Train {DATASET_NAME} LOSO with all 14 non-target subjects as source training data and one target test subject. "
            "The target is evaluated only after the fixed epoch count completes."
        )
    )
    parser.add_argument("--config", help="YAML experiment config; explicit CLI arguments override it")
    parser.add_argument("--data-root", help="Signature cache directory, or its de_rjsd_ica_1s_hop05 parent")
    parser.add_argument("--output-root", help="Training output parent (default: runs/seediv/de_rjsd_ica)")
    parser.add_argument("--run-name", help="Stable output directory name; defaults to a settings hash")
    parser.add_argument(
        "--feature",
        type=_feature_name,
        choices=FEATURES,
        default="cmrd",
        metavar="{CMRD,RJSD,DE,FUSION}",
        help=(
            "Training feature (case-insensitive): CMRD=tanh(alpha*zDE)*RJSD; "
            "RJSD=fold-specific divergence; DE=band log-power; "
            "FUSION=concat(RJSD, CMRD, zDE). Default: CMRD"
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="CMRD tanh gate strength")
    parser.add_argument("--fold", type=int, nargs="+", help="Target subject(s), default: all 1..15")
    parser.add_argument("--seed", type=int, nargs="+", default=[42])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:0")
    parser.add_argument("--resume", action="store_true", help="Reuse completed fold/seed jobs in the same run")
    parser.add_argument("--validate-only", action="store_true", help="Validate the cache and exit")
    parser.add_argument("--deep-validate", action="store_true", help="Read and numerically check every NPZ archive")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=10,
        help="Evaluate source and target every N epochs; 0 means final epoch only",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0, help="0 is the safest choice on Windows")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true", help="Allow faster non-deterministic CUDA kernels")

    parser.add_argument(
        "--model",
        choices=MODELS,
        default="plain_transformer",
        help="Model architecture; use hierarchical_attention for band/channel/temporal attention",
    )
    parser.add_argument("--d-model", type=int, default=600)
    parser.add_argument("--nhead", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--channel-heads", type=int, default=4)
    parser.add_argument("--temporal-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser


def main() -> None:
    args = parse_args_with_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    root = _resolve_data_root(args.data_root)
    LOGGER.info("Data root: %s", root)
    validation = validate_cache(root, deep=args.deep_validate)
    LOGGER.info("Selected training feature: %s", args.feature.upper())
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
