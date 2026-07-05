from __future__ import annotations

from pathlib import Path

import numpy as np

from cmrd.config import ExperimentConfig
from cmrd.data.records import TrialSample
from cmrd.data.splits import SubjectSplit, subject_loso_split
from cmrd.io import read_json
from cmrd.preprocessing import cache_root


def _sample(path: Path, entry: dict[str, object]) -> TrialSample:
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError(f"Invalid processed trial {path}: {x.shape}")
    return TrialSample(
        x=x,
        label=int(entry["label"]),
        subject=int(entry["subject"]),
        session=int(entry["session"]),
        trial=int(entry["trial"]),
        source_index=int(entry["source_index"]),
    )


def load_de_split(config: ExperimentConfig, target: int, include_test: bool) -> tuple[list[TrialSample], list[TrialSample], list[TrialSample], SubjectSplit]:
    root = cache_root(config)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DE cache missing: {manifest_path}. Run preprocessing first.")
    manifest = read_json(manifest_path)
    if manifest.get("preprocessing_signature") != config.preprocessing_signature():
        raise ValueError("DE cache signature does not match the active configuration")
    entries = list(manifest["trials"])
    subjects = np.asarray([entry["subject"] for entry in entries], dtype=np.int64)
    split = subject_loso_split(subjects, target, int(config.raw["split"]["validation_subjects"]), int(config.raw["split"]["seed"]))
    train_entries = [entry for entry in entries if int(entry["subject"]) in split.train_subjects]
    validation_entries = [entry for entry in entries if int(entry["subject"]) in split.validation_subjects]
    test_entries = [entry for entry in entries if int(entry["subject"]) == target] if include_test else []
    return (
        [_sample(root / str(entry["path"]), entry) for entry in train_entries],
        [_sample(root / str(entry["path"]), entry) for entry in validation_entries],
        [_sample(root / str(entry["path"]), entry) for entry in test_entries],
        split,
    )


def load_rd_split(config: ExperimentConfig, target: int, include_test: bool) -> tuple[list[TrialSample], list[TrialSample], list[TrialSample], SubjectSplit]:
    root = cache_root(config) / "folds" / f"fold-{target:02d}"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RD fold cache missing: {manifest_path}. Run preprocessing first.")
    manifest = read_json(manifest_path)
    provenance = read_json(root / "provenance.json")
    if manifest.get("preprocessing_signature") != config.preprocessing_signature():
        raise ValueError("RD cache signature does not match the active configuration")
    if provenance.get("reference_source") != "source_train_only" or int(provenance["target_subject"]) != target:
        raise ValueError("Invalid RD reference provenance")
    split_data = manifest["split"]
    split = SubjectSplit(tuple(split_data["train_subjects"]), tuple(split_data["validation_subjects"]), int(split_data["target_subject"]))
    groups = manifest["groups"]
    train = [_sample(root / str(entry["path"]), entry) for entry in groups["train"]]
    validation = [_sample(root / str(entry["path"]), entry) for entry in groups["validation"]]
    test = [_sample(root / str(entry["path"]), entry) for entry in groups["test"]] if include_test else []
    train_subjects = {sample.subject for sample in train}
    validation_subjects = {sample.subject for sample in validation}
    test_subjects = {sample.subject for sample in test}
    if train_subjects != set(split.train_subjects) or validation_subjects != set(split.validation_subjects):
        raise ValueError("RD cached split disagrees with provenance")
    if include_test and test_subjects != {target}:
        raise ValueError("RD test cache contains non-target subjects")
    if train_subjects & validation_subjects or train_subjects & test_subjects or validation_subjects & test_subjects:
        raise ValueError("Subject leakage in RD cached fold")
    return train, validation, test, split


def load_split(config: ExperimentConfig, target: int, include_test: bool = False):
    if config.feature == "de":
        return load_de_split(config, target, include_test)
    return load_rd_split(config, target, include_test)

