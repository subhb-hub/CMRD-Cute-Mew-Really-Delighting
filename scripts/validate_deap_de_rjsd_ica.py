from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrd.config import load_config
from cmrd.io import read_json, write_json


EXPECTED_TRIALS = 1280
EXPECTED_SUBJECTS = 32
EXPECTED_TRIALS_PER_SUBJECT = 40
EXPECTED_LABELS = {0: 260, 1: 296, 2: 266, 3: 458}
EXPECTED_DE_SHAPE = (60, 32, 5)
EXPECTED_PHIST_SHAPE = (60, 32, 5, 32)
EXPECTED_REFERENCE_SHAPE = (32, 5, 32)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _default_cache_root() -> Path:
    config = load_config("configs/deap/rd.yaml", expected_feature="rd")
    parent = config.processed_root / "deap" / "de_rjsd_ica_1s_hop1"
    candidates = [
        path
        for path in parent.iterdir()
        if path.is_dir() and (path / "pipeline_manifest.json").is_file()
    ]
    complete = [
        path
        for path in candidates
        if read_json(path / "pipeline_manifest.json").get("all_folds_complete")
    ]
    if len(complete) != 1:
        raise RuntimeError(f"Expected one complete DEAP cache under {parent}, found {complete}")
    return complete[0]


def validate_cache(root: Path, deep: bool) -> dict[str, Any]:
    root = root.resolve()
    trials_manifest = read_json(root / "trials_manifest.json")
    pipeline_manifest = read_json(root / "pipeline_manifest.json")
    environment = read_json(root / "environment.json")
    signature = str(trials_manifest["preprocessing_signature"])
    _require(environment["preprocessing_signature"] == signature, "Environment signature mismatch")
    _require(pipeline_manifest["preprocessing_signature"] == signature, "Pipeline signature mismatch")
    _require(bool(trials_manifest["complete"]), "Trial manifest is incomplete")
    _require(not trials_manifest["failures"], "Trial manifest contains failures")
    _require(bool(pipeline_manifest["all_folds_complete"]), "Pipeline folds are incomplete")
    _require(int(pipeline_manifest["expected_folds"]) == EXPECTED_SUBJECTS, "Expected fold count mismatch")

    entries = list(trials_manifest["trials"])
    _require(len(entries) == EXPECTED_TRIALS, f"Expected 1280 trials, found {len(entries)}")
    _require(len({entry["trial_id"] for entry in entries}) == EXPECTED_TRIALS, "Duplicate trial IDs")
    _require(
        Counter(int(entry["subject"]) for entry in entries)
        == Counter({subject: EXPECTED_TRIALS_PER_SUBJECT for subject in range(1, 33)}),
        "Per-subject trial counts are invalid",
    )
    label_counts = Counter(int(entry["label"]) for entry in entries)
    _require(label_counts == Counter(EXPECTED_LABELS), f"Unexpected label counts: {label_counts}")
    _require(
        sorted(int(entry["source_index"]) for entry in entries) == list(range(EXPECTED_TRIALS)),
        "Source indices are not a complete 0..1279 sequence",
    )

    trial_files_checked = 0
    maximum_histogram_sum_error = 0.0
    for entry in entries:
        _require(tuple(entry["de_shape"]) == EXPECTED_DE_SHAPE, f"Bad DE manifest shape: {entry['trial_id']}")
        _require(tuple(entry["p_hist_shape"]) == EXPECTED_PHIST_SHAPE, f"Bad p_hist manifest shape: {entry['trial_id']}")
        path = root / str(entry["path"])
        _require(path.is_file(), f"Missing trial feature: {path}")
        if deep:
            with np.load(path, allow_pickle=False) as archive:
                de = np.asarray(archive["de"])
                histogram = np.asarray(archive["p_hist"], dtype=np.float32)
                _require(de.shape == EXPECTED_DE_SHAPE, f"Bad DE file shape: {path}")
                _require(histogram.shape == EXPECTED_PHIST_SHAPE, f"Bad p_hist file shape: {path}")
                _require(np.isfinite(de).all() and np.isfinite(histogram).all(), f"Non-finite trial feature: {path}")
                _require(int(archive["label"].item()) == int(entry["label"]), f"Trial label mismatch: {path}")
                error = float(np.max(np.abs(histogram.sum(axis=-1) - 1.0)))
                maximum_histogram_sum_error = max(maximum_histogram_sum_error, error)
                _require(error <= 0.002, f"Histogram normalization error {error}: {path}")
            trial_files_checked += 1

    ica_root = Path(environment["ica_cache_root"])
    ica_files = sorted((ica_root / "trials").glob("*.npz"))
    _require(len(ica_files) == EXPECTED_TRIALS, f"Expected 1280 ICA files, found {len(ica_files)}")
    ica_files_checked = 0
    detection_error_trials = 0
    fit_fallback_trials = 0
    excluded_components: list[int] = []
    interpolated_channel_trials = 0
    if deep:
        for path in ica_files:
            with np.load(path, allow_pickle=False) as archive:
                cleaned = np.asarray(archive["cleaned"])
                _require(cleaned.shape == (32, 12_000), f"Bad ICA shape: {path}")
                _require(np.isfinite(cleaned).all(), f"Non-finite ICA signal: {path}")
                metadata = json.loads(str(archive["ica_metadata_json"].item()))
            detection_error_trials += int(bool(metadata["detection_errors"]))
            fit_fallback_trials += int(bool(metadata["fit_errors"]))
            excluded_components.append(len(metadata["excluded_components"]))
            interpolated_channel_trials += int(bool(metadata["interpolated_bad_channels"]))
            ica_files_checked += 1
        _require(detection_error_trials == 0, f"Strict ICA detection errors in {detection_error_trials} trials")

    folds_checked = 0
    rjsd_files_checked = 0
    for target in range(1, EXPECTED_SUBJECTS + 1):
        fold_root = root / "folds" / f"fold-{target:02d}"
        manifest = read_json(fold_root / "manifest.json")
        provenance = read_json(fold_root / "provenance.json")
        _require(bool(manifest["complete"]), f"Fold {target} is incomplete")
        _require(manifest["preprocessing_signature"] == signature, f"Fold {target} signature mismatch")
        _require(int(manifest["target_subject"]) == target, f"Fold {target} target mismatch")
        train_subjects = set(map(int, provenance["train_subjects"]))
        validation_subjects = set(map(int, provenance["validation_subjects"]))
        _require(len(train_subjects) == 29 and len(validation_subjects) == 2, f"Fold {target} source split sizes")
        _require(not train_subjects & validation_subjects, f"Fold {target} train/validation overlap")
        _require(target not in train_subjects | validation_subjects, f"Fold {target} target leakage")
        _require(provenance["reference_source"] == "source_train_only", f"Fold {target} reference source")
        _require(int(provenance["reference_window_count"]) == 69_600, f"Fold {target} Q window count")
        groups = manifest["groups"]
        _require(len(groups["train"]) == 1160, f"Fold {target} train trial count")
        _require(len(groups["validation"]) == 80, f"Fold {target} validation trial count")
        _require(len(groups["test"]) == 40, f"Fold {target} test trial count")
        _require({int(row["subject"]) for row in groups["test"]} == {target}, f"Fold {target} test leakage")
        with np.load(fold_root / "rjsd_reference.npz", allow_pickle=False) as archive:
            reference = np.asarray(archive["Q"])
            _require(reference.shape == EXPECTED_REFERENCE_SHAPE, f"Fold {target} reference shape")
            _require(np.isfinite(reference).all(), f"Fold {target} non-finite reference")
            _require(set(map(int, archive["source_train_subjects"])) == train_subjects, f"Fold {target} Q subjects")
        if deep:
            for group in ("train", "validation", "test"):
                for row in groups[group]:
                    path = root / str(row["rjsd_path"])
                    _require(path.is_file(), f"Missing RJSD file: {path}")
                    with np.load(path, allow_pickle=False) as archive:
                        value = np.asarray(archive["rjsd"])
                        _require(value.shape == EXPECTED_DE_SHAPE, f"Bad RJSD shape: {path}")
                        _require(np.isfinite(value).all(), f"Non-finite RJSD: {path}")
                        _require(int(archive["subject"].item()) == int(row["subject"]), f"RJSD subject mismatch: {path}")
                    rjsd_files_checked += 1
        folds_checked += 1

    result = {
        "schema_version": 1,
        "status": "valid",
        "deep": deep,
        "cache_root": str(root),
        "preprocessing_signature": signature,
        "cleaning_signature": environment["cleaning_signature"],
        "trials": len(entries),
        "subjects": EXPECTED_SUBJECTS,
        "windows_per_trial": 60,
        "label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "trial_files_checked": trial_files_checked,
        "maximum_histogram_sum_error": maximum_histogram_sum_error,
        "ica_files": len(ica_files),
        "ica_files_checked": ica_files_checked,
        "strict_ica_detection_error_trials": detection_error_trials,
        "ica_fit_fallback_trials": fit_fallback_trials,
        "ica_interpolated_channel_trials": interpolated_channel_trials,
        "ica_excluded_components": {
            "min": min(excluded_components) if excluded_components else None,
            "max": max(excluded_components) if excluded_components else None,
            "mean": float(np.mean(excluded_components)) if excluded_components else None,
        },
        "folds_checked": folds_checked,
        "rjsd_files_checked": rjsd_files_checked,
        "reference_windows_per_fold": 69_600,
        "split_trials_per_fold": {"train": 1160, "validation": 80, "test": 40},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the completed DEAP DE/RJSD/ICA cache")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--write", action="store_true", help="Write validation.json into the cache root")
    args = parser.parse_args()
    root = args.cache_root.resolve() if args.cache_root else _default_cache_root()
    result = validate_cache(root, args.deep)
    if args.write:
        write_json(root / "validation.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
