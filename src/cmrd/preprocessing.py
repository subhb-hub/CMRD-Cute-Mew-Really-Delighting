from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from cmrd.config import ExperimentConfig
from cmrd.data.loaders import iter_trials, validate_dataset
from cmrd.data.records import TrialRecord
from cmrd.data.splits import subject_loso_split
from cmrd.features.de import extract_de
from cmrd.features.rd import extract_spectral_histograms, fit_reference, transform_rd
from cmrd.features.signal import preprocess_signal
from cmrd.io import read_json, write_json, write_npz

LOGGER = logging.getLogger("cmrd.preprocessing")


def cache_root(config: ExperimentConfig) -> Path:
    return config.processed_root / config.dataset / config.feature / config.preprocessing_signature()


def _trial_entry(record: TrialRecord, source_index: int, relative_path: str, shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "trial_id": record.trial_id,
        "path": relative_path,
        "label": record.label,
        "subject": record.subject,
        "session": record.session,
        "trial": record.trial,
        "source_index": source_index,
        "source_file": record.source_file,
        "source_key": record.source_key,
        "shape": list(shape),
    }


def _signal_parameters(config: ExperimentConfig) -> tuple[dict, float, float, int, float, float, dict]:
    signal = config.raw["signal"]
    return (
        signal,
        float(signal["original_rate"]),
        float(signal["target_rate"]),
        int(signal["filter_order"]),
        float(signal["window_seconds"]),
        float(signal["hop_seconds"]),
        signal["bands_hz"],
    )


def _prepare_signal(record: TrialRecord, config: ExperimentConfig) -> np.ndarray:
    signal = config.raw["signal"]
    return preprocess_signal(
        record.signal,
        float(signal["original_rate"]),
        float(signal["target_rate"]),
        signal["broad_band_hz"],
        int(signal["filter_order"]),
    )


def _valid_manifest(path: Path, signature: str, expected_trials: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = read_json(path)
        return value["preprocessing_signature"] == signature and len(value["trials"]) == expected_trials
    except (KeyError, TypeError, ValueError):
        return False


def preprocess_de(config: ExperimentConfig, force: bool = False, resume: bool = False) -> Path:
    if config.feature != "de":
        raise ValueError("preprocess_de requires a DE configuration")
    root = cache_root(config)
    manifest_path = root / "manifest.json"
    dataset = validate_dataset(config)
    expected = int(dataset["expected_trials"])
    if _valid_manifest(manifest_path, config.preprocessing_signature(), expected) and not force:
        LOGGER.info("Reusing complete DE cache: %s", root)
        return root
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    _, _, rate, order, window, hop, bands = _signal_parameters(config)
    trials: list[dict[str, object]] = []
    started = time.perf_counter()
    for source_index, record in enumerate(iter_trials(config)):
        relative = f"trials/{record.trial_id}.npz"
        destination = root / relative
        if destination.is_file() and resume:
            with np.load(destination, allow_pickle=False) as archive:
                x = archive["X"]
                shape = x.shape
        else:
            x = extract_de(_prepare_signal(record, config), rate, window, hop, bands, order)
            write_npz(
                destination,
                X=x.astype(np.float32), label=np.int64(record.label), subject=np.int64(record.subject),
                session=np.int64(record.session), trial=np.int64(record.trial), source_index=np.int64(source_index),
            )
            shape = x.shape
        trials.append(_trial_entry(record, source_index, relative, shape))
        if (source_index + 1) % 25 == 0:
            LOGGER.info("DE processed %d/%d trials", source_index + 1, expected)
    if len(trials) != expected:
        raise RuntimeError(f"Expected {expected} trials, produced {len(trials)}")
    write_json(manifest_path, {
        "schema_version": 1,
        "dataset": config.dataset,
        "feature": "de",
        "preprocessing_signature": config.preprocessing_signature(),
        "feature_dim": int(config.raw["dataset"]["channels"]) * len(bands),
        "dataset_manifest": dataset,
        "config": config.canonical(),
        "trials": trials,
        "elapsed_seconds": time.perf_counter() - started,
    })
    return root


def _build_phist(config: ExperimentConfig, force: bool, resume: bool) -> tuple[Path, dict[str, object]]:
    root = cache_root(config) / "phist"
    manifest_path = root / "manifest.json"
    dataset = validate_dataset(config)
    expected = int(dataset["expected_trials"])
    if _valid_manifest(manifest_path, config.preprocessing_signature(), expected) and not force:
        return root, read_json(manifest_path)
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    _, _, rate, _, window, hop, bands = _signal_parameters(config)
    feature = config.raw["feature"]
    dtype = np.dtype(str(feature["storage_dtype"]))
    if dtype not in (np.dtype("float16"), np.dtype("float32")):
        raise ValueError("RD storage_dtype must be float16 or float32")
    trials: list[dict[str, object]] = []
    started = time.perf_counter()
    for source_index, record in enumerate(iter_trials(config)):
        relative = f"trials/{record.trial_id}.npz"
        destination = root / relative
        if destination.is_file() and resume:
            with np.load(destination, allow_pickle=False) as archive:
                shape = archive["p_hist"].shape
        else:
            histogram = extract_spectral_histograms(
                _prepare_signal(record, config), rate, window, hop, bands,
                int(feature["hist_bins_per_band"]), int(feature["spectral_nfft"]),
            )
            write_npz(
                destination,
                p_hist=histogram.astype(dtype), label=np.int64(record.label), subject=np.int64(record.subject),
                session=np.int64(record.session), trial=np.int64(record.trial), source_index=np.int64(source_index),
            )
            shape = histogram.shape
        trials.append(_trial_entry(record, source_index, relative, shape))
        if (source_index + 1) % 25 == 0:
            LOGGER.info("RD p_hist processed %d/%d trials", source_index + 1, expected)
    manifest = {
        "schema_version": 1,
        "dataset": config.dataset,
        "feature": "p_hist",
        "preprocessing_signature": config.preprocessing_signature(),
        "dataset_manifest": dataset,
        "config": config.canonical(),
        "trials": trials,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    return root, manifest


def import_legacy_phist(config: ExperimentConfig, legacy_root: str | Path) -> Path:
    """Move a verified legacy p_hist cache into the v1 per-trial cache schema."""
    legacy = Path(legacy_root).resolve()
    old_manifest_path = legacy / "manifest.json"
    if not old_manifest_path.is_file():
        raise FileNotFoundError(f"Legacy p_hist manifest not found: {old_manifest_path}")
    old = read_json(old_manifest_path)
    old_signature = old.get("preprocessing_signature", {})
    signal = config.raw["signal"]
    feature = config.raw["feature"]
    expected_signature = {
        "dataset": "SEED" if config.dataset == "seed" else "SEED-IV",
        "channels": int(config.raw["dataset"]["channels"]),
        "original_sampling_rate": float(signal["original_rate"]),
        "target_sampling_rate": float(signal["target_rate"]),
        "broad_band_hz": list(map(float, signal["broad_band_hz"])),
        "filter_order": int(signal["filter_order"]),
        "window_seconds": float(signal["window_seconds"]),
        "hop_seconds": float(signal["hop_seconds"]),
        "bands_hz": {name: list(map(float, limits)) for name, limits in signal["bands_hz"].items()},
        "hist_bins_per_band": int(feature["hist_bins_per_band"]),
        "spectral_nfft": int(feature["spectral_nfft"]),
    }
    if old_signature != expected_signature:
        raise ValueError("Legacy p_hist scientific signature does not match the active configuration")
    if str(old.get("storage_dtype")) != str(feature["storage_dtype"]):
        raise ValueError("Legacy p_hist storage dtype does not match the active configuration")
    dataset = validate_dataset(config)
    expected_trials = int(dataset["expected_trials"])
    old_trials = list(old.get("trials", []))
    if len(old_trials) != expected_trials:
        raise ValueError(f"Legacy cache has {len(old_trials)} trials, expected {expected_trials}")
    destination_root = cache_root(config) / "phist"
    destination_root.mkdir(parents=True, exist_ok=True)
    migrated: list[dict[str, object]] = []
    for old_entry in old_trials:
        source_index = int(old_entry["index"])
        subject = int(old_entry["subject"])
        session = int(old_entry["session"])
        trial = int(old_entry["trial"])
        trial_id = f"sub-{subject:02d}_ses-{session:02d}_trial-{trial:02d}"
        relative = f"trials/{trial_id}.npz"
        destination = destination_root / relative
        source = legacy / f"trial_{source_index:04d}.npz"
        if not destination.is_file():
            if not source.is_file():
                raise FileNotFoundError(f"Neither legacy nor migrated p_hist exists for index {source_index}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
        with np.load(destination, allow_pickle=False) as archive:
            shape = archive["p_hist"].shape
        old_source = Path(str(old_entry["source_file"]))
        try:
            marker = [part.lower() for part in old_source.parts].index("dataset")
            current_source = config.data_root.joinpath(*old_source.parts[marker + 1:]).resolve()
        except ValueError:
            current_source = old_source
        migrated.append({
            "trial_id": trial_id,
            "path": relative,
            "label": int(old_entry["label"]),
            "subject": subject,
            "session": session,
            "trial": trial,
            "source_index": source_index,
            "source_file": str(current_source),
            "source_key": str(old_entry["source_key"]),
            "shape": list(shape),
        })
    manifest = {
        "schema_version": 1,
        "dataset": config.dataset,
        "feature": "p_hist",
        "preprocessing_signature": config.preprocessing_signature(),
        "dataset_manifest": dataset,
        "config": config.canonical(),
        "trials": migrated,
        "migration": {"source": str(legacy), "legacy_signature_sha256": old.get("preprocessing_signature_sha256")},
    }
    write_json(destination_root / "manifest.json", manifest)
    return destination_root


def _load_histograms(root: Path, entries: Iterable[dict[str, object]]) -> Iterable[np.ndarray]:
    for entry in entries:
        with np.load(root / str(entry["path"]), allow_pickle=False) as archive:
            yield np.asarray(archive["p_hist"], dtype=np.float32)


def _rd_fold(config: ExperimentConfig, phist_root: Path, phist_manifest: dict[str, object], target: int, force: bool, resume: bool) -> Path:
    fold_root = cache_root(config) / "folds" / f"fold-{target:02d}"
    manifest_path = fold_root / "manifest.json"
    signature = config.preprocessing_signature()
    if manifest_path.is_file() and not force:
        value = read_json(manifest_path)
        if value.get("preprocessing_signature") == signature and value.get("target_subject") == target:
            return fold_root
    if force and fold_root.exists():
        shutil.rmtree(fold_root)
    fold_root.mkdir(parents=True, exist_ok=True)
    entries = list(phist_manifest["trials"])
    subjects = np.asarray([entry["subject"] for entry in entries], dtype=np.int64)
    split = subject_loso_split(subjects, target, int(config.raw["split"]["validation_subjects"]), int(config.raw["split"]["seed"]))
    groups = {
        "train": [entry for entry in entries if int(entry["subject"]) in split.train_subjects],
        "validation": [entry for entry in entries if int(entry["subject"]) in split.validation_subjects],
        "test": [entry for entry in entries if int(entry["subject"]) == target],
    }
    reference, window_count = fit_reference(_load_histograms(phist_root, groups["train"]))
    write_npz(
        fold_root / "rd_reference.npz",
        Q=reference.astype(np.float32), source_train_subjects=np.asarray(split.train_subjects),
        source_train_indices=np.asarray([entry["source_index"] for entry in groups["train"]], dtype=np.int64),
    )
    output_groups: dict[str, list[dict[str, object]]] = {key: [] for key in groups}
    for group_name, group_entries in groups.items():
        for entry in group_entries:
            relative = f"features/{entry['trial_id']}.npz"
            destination = fold_root / relative
            if destination.is_file() and resume:
                with np.load(destination, allow_pickle=False) as archive:
                    shape = archive["X"].shape
            else:
                with np.load(phist_root / str(entry["path"]), allow_pickle=False) as archive:
                    x = transform_rd(np.asarray(archive["p_hist"], dtype=np.float32), reference)
                write_npz(
                    destination, X=x, label=np.int64(entry["label"]), subject=np.int64(entry["subject"]),
                    session=np.int64(entry["session"]), trial=np.int64(entry["trial"]), source_index=np.int64(entry["source_index"]),
                )
                shape = x.shape
            output_groups[group_name].append({**entry, "path": relative, "shape": list(shape)})
    provenance = {
        "feature": "rd",
        "old_name": "rjsd",
        "reference_source": "source_train_only",
        "preprocessing_signature": signature,
        **split.as_dict(),
        "source_train_trial_indices": [entry["source_index"] for entry in groups["train"]],
        "source_validation_trial_indices": [entry["source_index"] for entry in groups["validation"]],
        "target_test_trial_indices": [entry["source_index"] for entry in groups["test"]],
        "reference_window_count": window_count,
        "reference_shape": list(reference.shape),
    }
    write_json(fold_root / "provenance.json", provenance)
    write_json(manifest_path, {
        "schema_version": 1,
        "dataset": config.dataset,
        "feature": "rd",
        "preprocessing_signature": signature,
        "target_subject": target,
        "feature_dim": int(config.raw["dataset"]["channels"]) * len(config.raw["signal"]["bands_hz"]),
        "split": split.as_dict(),
        "reference": provenance,
        "groups": output_groups,
    })
    return fold_root


def preprocess_rd(config: ExperimentConfig, fold: int | None = None, force: bool = False, resume: bool = False) -> Path:
    if config.feature != "rd":
        raise ValueError("preprocess_rd requires an RD configuration")
    phist_root, manifest = _build_phist(config, force, resume)
    targets = [fold] if fold is not None else list(range(1, int(config.raw["dataset"]["subjects"]) + 1))
    for target in targets:
        if not 1 <= int(target) <= int(config.raw["dataset"]["subjects"]):
            raise ValueError(f"Invalid target fold: {target}")
        LOGGER.info("Building RD fold %02d", target)
        _rd_fold(config, phist_root, manifest, int(target), force, resume)
    return cache_root(config)
