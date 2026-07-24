from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import time
import traceback
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cmrd.config import ExperimentConfig, load_config
from cmrd.faced import EMOTION_NAMES, SUBJECTS, VIDEO_LABELS, VIDEOS, trial_entries
from cmrd.features.landmark_hilbert import HilbertAtlasState, fit_channel_band_atlas, streaming_moments
from cmrd.io import _replace_with_retry, read_json, write_json, write_npz
from cmrd.models.slst import ARCHITECTURES, FEATURE_MODES, JSDHilbertTokenizer, StructuredLandmarkSpectralTransformer
from cmrd.spectral_atlas_runner import (
    FACED_PROTOCOLS,
    _base_signature,
    _load_base,
    _native_grids,
    _seediv_context,
    prepare_base_cache,
    split_groups,
    validate_sources,
)
from cmrd.training.metrics import classification_metrics
from cmrd.training.runtime import environment_manifest, seed_everything, select_device


LOGGER = logging.getLogger("cmrd.slst")
SCHEMA_VERSION = 1
FAMILY = "SLST-v1"
LEARNABILITY_CONDITIONS = (
    "L0_fixed",
    "L1_lr3e5_freeze3",
    "L2_lr1e4_freeze3",
    "L3_lr3e4_freeze3",
    "L4_lr1e4_unfrozen",
    "L5_lr1e4_regularized",
)
CONDITIONS = (
    *FEATURE_MODES,
    "C1_random_learnable",
    "C2_learnable",
    "C3_anchor",
    "C4_regularized",
    *LEARNABILITY_CONDITIONS,
)
FACED_SLST_PROTOCOLS = (
    "conventional_subject_holdout",
    "subject_stimulus_rotation_0",
    "subject_stimulus_rotation_1",
    "subject_stimulus_rotation_2",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def settings(config: ExperimentConfig) -> dict[str, Any]:
    raw = dict(config.raw.get("slst", {}))
    value = {
        "folds": tuple(map(int, raw.get("folds", [1, 5, 10] if config.dataset == "faced" else [1, 8, 15]))),
        "protocols": tuple(map(str, raw.get("protocols", ["conventional_subject_holdout"] if config.dataset == "faced" else ["loso"]))),
        "seeds": tuple(map(int, raw.get("seeds", config.raw["training"]["seeds"]))),
        "landmarks": int(raw.get("landmarks", 8)),
        "candidate_windows_per_trial": int(raw.get("candidate_windows_per_trial", 2)),
        "maximum_candidates": int(raw.get("maximum_candidates", 8192)),
        "gram_ridge": float(raw.get("gram_ridge", 1e-4)),
        "direction_rank": int(raw.get("direction_rank", 4)),
        "eigenvalue_floor_ratio": float(raw.get("eigenvalue_floor_ratio", 1e-3)),
        "effective_rank_tolerance": float(raw.get("effective_rank_tolerance", 1e-6)),
        "monitor_interval": int(raw.get("monitor_interval", 10)),
        "conditions": tuple(map(str, raw.get("conditions", CONDITIONS))),
        "architectures": tuple(map(str, raw.get("architectures", ["B4_slst"]))),
        "anchor_weight": float(raw.get("anchor_weight", 0.01)),
        "diversity_weight": float(raw.get("diversity_weight", 0.01)),
        "coverage_weight": float(raw.get("coverage_weight", 0.01)),
        "diversity_margin": float(raw.get("diversity_margin", 0.1)),
        "coverage_temperature": float(raw.get("coverage_temperature", 0.05)),
        "freeze_landmark_epochs": int(raw.get("freeze_landmark_epochs", 12)),
        "landmark_learning_rate": float(raw.get("landmark_learning_rate", 3e-5)),
        "subjects_per_batch": int(raw.get("subjects_per_batch", 4)),
        "trials_per_subject": int(raw.get("trials_per_subject", 1)),
        "gradient_accumulation": int(raw.get("gradient_accumulation", 1)),
        "early_stopping_patience_monitors": int(raw.get("early_stopping_patience_monitors", 3)),
        "target_monitoring": bool(raw.get("target_monitoring", True)),
        "source_diagnostic_batches": int(raw.get("source_diagnostic_batches", 8)),
        "landmark_gradient_diagnostics": bool(raw.get("landmark_gradient_diagnostics", True)),
        "vrex_weight": float(raw.get("vrex_weight", 0.0)),
    }
    maximum_fold = 10 if config.dataset == "faced" else 15
    if not value["folds"] or any(not 1 <= fold <= maximum_fold for fold in value["folds"]):
        raise ValueError(f"Invalid folds: {value['folds']}")
    expected_protocols = FACED_SLST_PROTOCOLS if config.dataset == "faced" else ("loso",)
    if not value["protocols"] or any(protocol not in expected_protocols for protocol in value["protocols"]):
        raise ValueError(f"Invalid {config.dataset} protocols: {value['protocols']}")
    if not value["seeds"] or value["landmarks"] < 1 or value["candidate_windows_per_trial"] < 1:
        raise ValueError("seeds, landmarks, and candidate_windows_per_trial must be non-empty/positive")
    if value["maximum_candidates"] < 32 or value["monitor_interval"] != 10:
        raise ValueError("SLST requires >=32 candidates and 10-epoch monitoring")
    if any(condition not in CONDITIONS for condition in value["conditions"]):
        raise ValueError("slst.conditions contains an unknown condition")
    if any(model not in ARCHITECTURES for model in value["architectures"]):
        raise ValueError("slst.architectures contains an unknown architecture")
    training = config.raw["training"]
    if int(training["epochs"]) < 10 or int(training["epochs"]) % 10:
        raise ValueError("training.epochs must be a positive multiple of 10")
    if value["subjects_per_batch"] < 2 or value["trials_per_subject"] < 1 or value["gradient_accumulation"] < 1:
        raise ValueError("Subject-balanced batch settings are invalid")
    if value["direction_rank"] < 1 or value["direction_rank"] > value["landmarks"] or value["eigenvalue_floor_ratio"] <= 0 or value["effective_rank_tolerance"] <= 0 or value["source_diagnostic_batches"] < 1:
        raise ValueError("SLST coordinate diagnostic settings must be positive")
    return value


def _rotation_videos(rotation: int) -> dict[str, np.ndarray]:
    train: list[int] = []
    development: list[int] = []
    test: list[int] = []
    for label in range(len(EMOTION_NAMES)):
        videos = np.flatnonzero(VIDEO_LABELS == label).tolist()
        rotating = videos[:3]
        test_video = rotating[rotation]
        development_video = rotating[(rotation + 1) % 3]
        test.append(test_video)
        development.append(development_video)
        train.extend(video for video in videos if video not in {test_video, development_video})
    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "development": np.asarray(sorted(development), dtype=np.int64),
        "test": np.asarray(sorted(test), dtype=np.int64),
    }


def groups_for(config: ExperimentConfig, fold: int, protocol: str) -> dict[str, list[dict[str, Any]]]:
    if config.dataset != "faced" or protocol == "conventional_subject_holdout":
        compatible = "conventional_subject_holdout" if protocol == "conventional_subject_holdout" else protocol
        return split_groups(config, fold, compatible)
    if not protocol.startswith("subject_stimulus_rotation_"):
        raise ValueError(protocol)
    rotation = int(protocol.rsplit("_", 1)[1])
    base = split_groups(config, fold, "conventional_subject_holdout")
    videos = _rotation_videos(rotation)
    output: dict[str, list[dict[str, Any]]] = {}
    for group, entries in base.items():
        allowed = {int(value) + 1 for value in videos[group]}
        output[group] = [entry for entry in entries if int(entry["video"]) in allowed]
    return output


def _all_entries(config: ExperimentConfig) -> list[dict[str, Any]]:
    if config.dataset == "faced":
        return trial_entries(range(SUBJECTS))
    _, _, folds = _seediv_context(config)
    unique: dict[str, dict[str, Any]] = {}
    for fold in folds.values():
        for group in fold.values():
            for entry in group:
                unique[str(entry["trial_id"])] = dict(entry)
    return sorted(unique.values(), key=lambda item: int(item["source_index"]))


def _base_cache_run_root(config: ExperimentConfig, run_root: Path) -> Path:
    configured = config.raw.get("slst", {}).get("base_cache_run_root")
    return config.resolve_path(configured) if configured else run_root


def _pack_root(config: ExperimentConfig, run_root: Path) -> Path:
    return run_root / "cache" / "structured" / config.dataset / _hash({"family": FAMILY, "base": _base_signature(config)})


def prepare_structured_cache(config: ExperimentConfig, run_root: Path) -> dict[str, Any]:
    root = _pack_root(config, run_root)
    manifest_path = root / "manifest.json"
    required = [root / name for name in ("shape.npy", "magnitude.npy", "de.npy")]
    if manifest_path.is_file() and all(path.is_file() for path in required):
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete" and manifest.get("base_signature") == _base_signature(config):
            return manifest
    entries = _all_entries(config)
    metadata: list[dict[str, Any]] = []
    total_windows = 0
    for position, entry in enumerate(entries, 1):
        shapes, magnitude, _ = _load_base(config, _base_cache_run_root(config, run_root), entry)
        windows = int(magnitude.shape[0])
        metadata.append({
            "trial_id": str(entry["trial_id"]),
            "start": total_windows,
            "stop": total_windows + windows,
            "label": int(entry["label"]),
            "subject": int(entry["subject"]),
            "session": int(entry.get("session", 1)),
            "trial": int(entry["trial"]),
            "video": int(entry.get("video", entry["trial"])),
            "source_index": int(entry["source_index"]),
        })
        total_windows += windows
        if position % 250 == 0:
            LOGGER.info("Structured-cache sizing %d/%d", position, len(entries))
    grids = _native_grids(config)
    maximum_frequency = max(map(len, grids))
    frequency_mask = np.zeros((len(grids), maximum_frequency), dtype=bool)
    for band, grid in enumerate(grids):
        frequency_mask[band, : len(grid)] = True
    root.mkdir(parents=True, exist_ok=True)
    channels = int(config.raw["dataset"]["channels"])
    bands = len(grids)
    temporary = {name: root / f".{name}.partial.npy" for name in ("shape", "magnitude", "de")}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    shape_store = np.lib.format.open_memmap(temporary["shape"], mode="w+", dtype=np.float16, shape=(total_windows, channels, bands, maximum_frequency))
    magnitude_store = np.lib.format.open_memmap(temporary["magnitude"], mode="w+", dtype=np.float32, shape=(total_windows, channels, bands))
    de_store = np.lib.format.open_memmap(temporary["de"], mode="w+", dtype=np.float32, shape=(total_windows, channels, bands))
    try:
        for position, (entry, meta) in enumerate(zip(entries, metadata, strict=True), 1):
            shapes, magnitude, de = _load_base(config, _base_cache_run_root(config, run_root), entry)
            start, stop = int(meta["start"]), int(meta["stop"])
            shape_store[start:stop] = 0.0
            for band, values in enumerate(shapes):
                shape_store[start:stop, :, band, : values.shape[-1]] = values.astype(np.float16)
            magnitude_store[start:stop] = magnitude
            de_store[start:stop] = de
            if position % 100 == 0 or position == len(entries):
                LOGGER.info("Structured-cache write %d/%d", position, len(entries))
        shape_store.flush()
        magnitude_store.flush()
        de_store.flush()
    finally:
        del shape_store, magnitude_store, de_store
    for name, path in temporary.items():
        _replace_with_retry(path, root / f"{name}.npy")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "dataset": config.dataset,
        "base_signature": _base_signature(config),
        "root": str(root.resolve()),
        "total_trials": len(entries),
        "total_windows": total_windows,
        "shape": [total_windows, channels, bands, maximum_frequency],
        "frequency_mask": frequency_mask.tolist(),
        "entries": metadata,
        "storage": "window-concatenated float16 q and float32 magnitude/DE; mmap-compatible",
        "created_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    return manifest


class PackedTrialStore:
    def __init__(self, config: ExperimentConfig, run_root: Path) -> None:
        root = _pack_root(config, run_root)
        self.manifest = read_json(root / "manifest.json")
        self.shape = np.load(root / "shape.npy", mmap_mode="r")
        self.magnitude = np.load(root / "magnitude.npy", mmap_mode="r")
        self.de = np.load(root / "de.npy", mmap_mode="r")
        self.frequency_mask = np.asarray(self.manifest["frequency_mask"], dtype=bool)
        self.entries = {str(item["trial_id"]): item for item in self.manifest["entries"]}

    def arrays(self, entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        meta = self.entries[str(entry["trial_id"])]
        start, stop = int(meta["start"]), int(meta["stop"])
        return self.shape[start:stop], self.magnitude[start:stop], self.de[start:stop]


def _protocol_payload(config: ExperimentConfig, fold: int, protocol: str, groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    value = settings(config)
    return {
        "family": FAMILY,
        "dataset": config.dataset,
        "fold": int(fold),
        "protocol": protocol,
        "source_train_trial_ids": [str(entry["trial_id"]) for entry in groups["train"]],
        "source_development_trial_ids": [str(entry["trial_id"]) for entry in groups["development"]],
        "target_trial_ids": [str(entry["trial_id"]) for entry in groups["test"]],
        "landmarks": value["landmarks"],
        "candidate_windows_per_trial": value["candidate_windows_per_trial"],
        "maximum_candidates": value["maximum_candidates"],
        "state_fit_scope": "source-train only",
        "normalization_fit_scope": "source-train only",
    }


def fit_atlas(config: ExperimentConfig, run_root: Path, fold: int, protocol: str) -> tuple[HilbertAtlasState, dict[str, Any]]:
    groups = groups_for(config, fold, protocol)
    payload = _protocol_payload(config, fold, protocol, groups)
    protocol_hash = _hash(payload)
    root = run_root / "cache" / "atlases" / config.dataset / protocol / f"fold-{fold:02d}" / protocol_hash
    state_path, meta_path = root / "state.npz", root / "state.json"
    if state_path.is_file() and meta_path.is_file() and read_json(meta_path).get("status") == "complete":
        with np.load(state_path, allow_pickle=False) as archive:
            state = HilbertAtlasState(**{name: np.asarray(archive[name]) for name in HilbertAtlasState.__dataclass_fields__})
        return state, read_json(meta_path)
    store = PackedTrialStore(config, run_root)
    value = settings(config)
    candidates: list[np.ndarray] = []
    magnitude_chunks: list[np.ndarray] = []
    de_chunks: list[np.ndarray] = []
    for position, entry in enumerate(groups["train"], 1):
        shape, magnitude, de = store.arrays(entry)
        count = min(value["candidate_windows_per_trial"], shape.shape[0])
        indices = np.unique(np.linspace(0, shape.shape[0] - 1, count, dtype=np.int64))
        candidates.append(np.asarray(shape[indices], dtype=np.float32))
        magnitude_chunks.append(magnitude)
        de_chunks.append(de)
        if position % 250 == 0:
            LOGGER.info("Source-only atlas pass %d/%d", position, len(groups["train"]))
    candidate_array = np.concatenate(candidates, axis=0)
    if candidate_array.shape[0] > value["maximum_candidates"]:
        selected = np.unique(np.linspace(0, candidate_array.shape[0] - 1, value["maximum_candidates"], dtype=np.int64))
        candidate_array = candidate_array[selected]
    center, anchors = fit_channel_band_atlas(candidate_array, store.frequency_mask, value["landmarks"])
    magnitude_mean, magnitude_scale = streaming_moments(magnitude_chunks)
    de_mean, de_scale = streaming_moments(de_chunks)
    state = HilbertAtlasState(
        center=center,
        anchors=anchors,
        frequency_mask=store.frequency_mask,
        magnitude_mean=magnitude_mean,
        magnitude_scale=magnitude_scale,
        de_mean=de_mean,
        de_scale=de_scale,
    )
    write_npz(state_path, **{name: getattr(state, name) for name in HilbertAtlasState.__dataclass_fields__})
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "family": FAMILY,
        "status": "complete",
        "protocol_hash": protocol_hash,
        "protocol": payload,
        "candidate_count": int(candidate_array.shape[0]),
        "candidate_source_subjects": sorted({int(entry["subject"]) for entry in groups["train"]}),
        "target_data_used_to_fit_state": False,
        "state_path": str(state_path.resolve()),
        "created_at": utc_now(),
    }
    write_json(meta_path, metadata)
    return state, metadata


class TrialDataset(Dataset[dict[str, Any]]):
    def __init__(self, store: PackedTrialStore, entries: Sequence[dict[str, Any]]) -> None:
        self.store = store
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        shape, magnitude, de = self.store.arrays(entry)
        return {
            "shape": np.asarray(shape, dtype=np.float32),
            "magnitude": np.asarray(magnitude, dtype=np.float32),
            "de": np.asarray(de, dtype=np.float32),
            "label": int(entry["label"]),
            "subject": int(entry["subject"]),
            "video": int(entry.get("video", entry["trial"])),
            "trial_id": str(entry["trial_id"]),
        }


def collate_trials(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(item["shape"].shape[0] for item in items)
    batch = len(items)
    _, channels, bands, frequencies = items[0]["shape"].shape
    shape = np.zeros((batch, maximum, channels, bands, frequencies), dtype=np.float32)
    magnitude = np.zeros((batch, maximum, channels, bands), dtype=np.float32)
    de = np.zeros_like(magnitude)
    mask = np.zeros((batch, maximum), dtype=bool)
    for index, item in enumerate(items):
        length = item["shape"].shape[0]
        shape[index, :length] = item["shape"]
        magnitude[index, :length] = item["magnitude"]
        de[index, :length] = item["de"]
        mask[index, :length] = True
    return {
        "shape": torch.from_numpy(shape),
        "magnitude": torch.from_numpy(magnitude),
        "de": torch.from_numpy(de),
        "mask": torch.from_numpy(mask),
        "label": torch.tensor([item["label"] for item in items], dtype=torch.long),
        "subject": torch.tensor([item["subject"] for item in items], dtype=torch.long),
        "video": torch.tensor([item["video"] for item in items], dtype=torch.long),
        "trial_id": [item["trial_id"] for item in items],
    }


class SubjectBalancedBatchSampler(Sampler[list[int]]):
    def __init__(self, entries: Sequence[dict[str, Any]], subjects_per_batch: int, trials_per_subject: int, seed: int) -> None:
        self.by_subject: dict[int, list[int]] = defaultdict(list)
        for index, entry in enumerate(entries):
            self.by_subject[int(entry["subject"])].append(index)
        self.subjects_per_batch = int(subjects_per_batch)
        self.trials_per_subject = int(trials_per_subject)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        size = self.subjects_per_batch * self.trials_per_subject
        return max(1, math.ceil(sum(map(len, self.by_subject.values())) / size))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        queues = {subject: rng.permutation(indices).tolist() for subject, indices in self.by_subject.items()}
        while True:
            available = [subject for subject, queue in queues.items() if queue]
            if len(available) < 2:
                return
            chosen = rng.choice(available, size=min(self.subjects_per_batch, len(available)), replace=False)
            batch: list[int] = []
            for subject in chosen:
                queue = queues[int(subject)]
                for _ in range(min(self.trials_per_subject, len(queue))):
                    batch.append(queue.pop())
            if len(batch) >= 2:
                yield batch


def _condition(condition: str, value: dict[str, Any]) -> dict[str, Any]:
    if condition in FEATURE_MODES:
        return {"feature_mode": condition, "learnable": False, "random": False, "freeze": 0, "weights": (0.0, 0.0, 0.0), "landmark_learning_rate": 0.0}
    mapping = {
        "C1_random_learnable": (True, True, 0, (0.0, 0.0, 0.0)),
        "C2_learnable": (True, False, value["freeze_landmark_epochs"], (0.0, 0.0, 0.0)),
        "C3_anchor": (True, False, value["freeze_landmark_epochs"], (value["anchor_weight"], 0.0, 0.0)),
        "C4_regularized": (True, False, value["freeze_landmark_epochs"], (value["anchor_weight"], value["diversity_weight"], value["coverage_weight"])),
    }
    if condition in mapping:
        learnable, random, freeze, weights = mapping[condition]
        return {
            "feature_mode": "A6_hilbert_landmark",
            "learnable": learnable,
            "random": random,
            "freeze": freeze,
            "weights": weights,
            "landmark_learning_rate": value["landmark_learning_rate"],
        }
    learnability = {
        "L0_fixed": (False, 0, 0.0, (0.0, 0.0, 0.0)),
        "L1_lr3e5_freeze3": (True, 3, 3e-5, (0.0, 0.0, 0.0)),
        "L2_lr1e4_freeze3": (True, 3, 1e-4, (0.0, 0.0, 0.0)),
        "L3_lr3e4_freeze3": (True, 3, 3e-4, (0.0, 0.0, 0.0)),
        "L4_lr1e4_unfrozen": (True, 0, 1e-4, (0.0, 0.0, 0.0)),
        "L5_lr1e4_regularized": (
            True,
            3,
            1e-4,
            (value["anchor_weight"], value["diversity_weight"], value["coverage_weight"]),
        ),
    }
    learnable, freeze, landmark_lr, weights = learnability[condition]
    return {
        "feature_mode": "H4_stable_hilbert_lowrank_explicit",
        "learnable": learnable,
        "random": False,
        "freeze": freeze,
        "weights": weights,
        "landmark_learning_rate": landmark_lr,
    }


def build_model(config: ExperimentConfig, atlas: HilbertAtlasState, condition: str, architecture: str) -> StructuredLandmarkSpectralTransformer:
    value = settings(config)
    specification = _condition(condition, value)
    tokenizer = JSDHilbertTokenizer(
        torch.from_numpy(atlas.center),
        torch.from_numpy(atlas.anchors),
        torch.from_numpy(atlas.frequency_mask),
        torch.from_numpy(atlas.magnitude_mean),
        torch.from_numpy(atlas.magnitude_scale),
        torch.from_numpy(atlas.de_mean),
        torch.from_numpy(atlas.de_scale),
        feature_mode=specification["feature_mode"],
        learnable_landmarks=specification["learnable"],
        random_landmarks=specification["random"],
        gram_ridge=value["gram_ridge"],
        direction_rank=value["direction_rank"],
        eigenvalue_floor_ratio=value["eigenvalue_floor_ratio"],
        effective_rank_tolerance=value["effective_rank_tolerance"],
        diversity_margin=value["diversity_margin"],
        coverage_temperature=value["coverage_temperature"],
    )
    model = config.raw["model"]
    return StructuredLandmarkSpectralTransformer(
        tokenizer,
        channels=int(config.raw["dataset"]["channels"]),
        bands=len(config.raw["signal"]["bands_hz"]),
        classes=int(config.raw["dataset"]["classes"]),
        max_length=int(model["max_length"]),
        architecture=architecture,
        d_model=int(model["d_model"]),
        band_heads=int(model["band_heads"]),
        channel_heads=int(model["channel_heads"]),
        temporal_heads=int(model["temporal_heads"]),
        band_layers=int(model["band_layers"]),
        channel_layers=int(model["channel_layers"]),
        temporal_layers=int(model["temporal_layers"]),
        feedforward=int(model["feedforward"]),
        dropout=float(model["dropout"]),
    )


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _subject_metrics(targets: np.ndarray, predictions: np.ndarray, subjects: np.ndarray, classes: int) -> dict[str, Any]:
    rows = []
    for subject in sorted(np.unique(subjects)):
        selected = subjects == subject
        metrics = classification_metrics(targets[selected], predictions[selected], classes)
        rows.append({"subject": int(subject), "trials": int(selected.sum()), "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"]})
    macro = np.asarray([row["macro_f1"] for row in rows], dtype=float)
    quartile = max(1, math.ceil(len(rows) / 4))
    return {
        "per_subject": rows,
        "subject_averaged_macro_f1": float(macro.mean()),
        "worst_quartile_subject_macro_f1": float(np.sort(macro)[:quartile].mean()),
    }


@torch.no_grad()
def evaluate(model: StructuredLandmarkSpectralTransformer, loader: DataLoader, device: torch.device, classes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    subjects: list[int] = []
    rows: list[dict[str, Any]] = []
    for batch in loader:
        moved = _move(batch, device)
        logits = model(moved["shape"], moved["magnitude"], moved["de"], moved["mask"])
        predicted = logits.argmax(dim=-1).cpu().numpy()
        labels = batch["label"].numpy()
        for index, trial_id in enumerate(batch["trial_id"]):
            rows.append({"trial_id": trial_id, "subject": int(batch["subject"][index]), "video": int(batch["video"][index]), "target": int(labels[index]), "prediction": int(predicted[index])})
        targets.extend(labels.tolist())
        predictions.extend(predicted.tolist())
        subjects.extend(batch["subject"].numpy().tolist())
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    subject_array = np.asarray(subjects, dtype=np.int64)
    metrics = classification_metrics(target_array, prediction_array, classes)
    metrics.update(_subject_metrics(target_array, prediction_array, subject_array, classes))
    return metrics, rows


def _gradient_norm(loss: torch.Tensor, parameters: Sequence[nn.Parameter]) -> float:
    active = [parameter for parameter in parameters if parameter.requires_grad]
    if not active or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, active, retain_graph=True, allow_unused=True)
    total = sum(float(gradient.detach().square().sum().cpu()) for gradient in gradients if gradient is not None)
    return math.sqrt(total)


def _existing_gradient_norm(parameters: Sequence[nn.Parameter]) -> float:
    total = sum(
        float(parameter.grad.detach().square().sum().cpu())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(total)


@torch.no_grad()
def source_coordinate_diagnostics(
    model: StructuredLandmarkSpectralTransformer,
    loader: DataLoader,
    device: torch.device,
    band_names: Sequence[str],
    maximum_batches: int,
) -> dict[str, Any]:
    model.eval()
    sums: torch.Tensor | None = None
    sums_square: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    d0_sum = torch.zeros(len(band_names), dtype=torch.float64)
    residual_sum = torch.zeros(len(band_names), dtype=torch.float64)
    coordinate_energy = torch.zeros(len(band_names), dtype=torch.float64)
    eigenvalues: torch.Tensor | None = None
    observed_batches = 0
    for batch_index, batch in enumerate(loader, 1):
        if batch_index > maximum_batches:
            break
        moved = _move(batch, device)
        _, diagnostics = model.tokenizer(
            moved["shape"], moved["magnitude"], moved["de"], return_diagnostics=True
        )
        if "direction_coordinates" not in diagnostics:
            return {
                "status": "not_applicable",
                "feature_mode": model.tokenizer.feature_mode,
                "reason": "condition_has_no_direction_coordinates",
            }
        coordinates = diagnostics["direction_coordinates"].detach().cpu().to(torch.float64)
        d0 = diagnostics["distance_to_center"].detach().cpu().to(torch.float64)
        residual = diagnostics["orthogonal_residual"].detach().cpu().to(torch.float64)
        valid = moved["mask"].detach().cpu()[:, :, None, None]
        expanded = valid.expand(*coordinates.shape[:-1])
        if sums is None:
            sums = torch.zeros(coordinates.shape[-2:], dtype=torch.float64)
            sums_square = torch.zeros_like(sums)
            counts = torch.zeros_like(sums)
        assert sums_square is not None and counts is not None
        mask_axis = expanded.unsqueeze(-1).to(torch.float64)
        sums += (coordinates * mask_axis).sum(dim=(0, 1, 2))
        sums_square += (coordinates.square() * mask_axis).sum(dim=(0, 1, 2))
        counts += mask_axis.sum(dim=(0, 1, 2))
        valid_scalar = expanded.to(torch.float64)
        d0_sum += (d0 * valid_scalar).sum(dim=(0, 1, 2))
        residual_sum += (residual * valid_scalar).sum(dim=(0, 1, 2))
        coordinate_energy += (coordinates.square().sum(dim=-1) * valid_scalar).sum(dim=(0, 1, 2))
        if eigenvalues is None:
            eigenvalues = diagnostics["gram_eigenvalues"].detach().cpu().to(torch.float64)
        observed_batches += 1
    if sums is None or sums_square is None or counts is None or eigenvalues is None:
        raise RuntimeError("No source batches were available for coordinate diagnostics")
    variance = (sums_square / counts.clamp_min(1.0) - (sums / counts.clamp_min(1.0)).square()).clamp_min(0.0)
    standard_deviation = variance.sqrt()
    hilbert_geometry = model.tokenizer.feature_mode in {
        "H3_hilbert_lowrank_explicit",
        "H4_stable_hilbert_lowrank_explicit",
        "H5_hilbert_full_explicit",
        "H6_stable_hilbert_lowrank_residual",
    }
    rows: list[dict[str, Any]] = []
    tolerance = model.tokenizer.effective_rank_tolerance
    for band_index, band_name in enumerate(band_names):
        band_eigenvalues = eigenvalues[:, band_index]
        threshold = tolerance * band_eigenvalues[:, :1]
        effective_rank = (band_eigenvalues > threshold).sum(dim=-1).to(torch.float64)
        condition = band_eigenvalues[:, 0] / band_eigenvalues[:, -1].clamp_min(model.tokenizer.gram_ridge)
        rows.append({
            "band": str(band_name),
            "gram_eigenvalues_median": band_eigenvalues.median(dim=0).values.tolist(),
            "minimum_gram_eigenvalue": float(band_eigenvalues.min()),
            "median_condition_number": float(condition.median()),
            "maximum_condition_number": float(condition.max()),
            "mean_effective_rank": float(effective_rank.mean()),
            "coordinate_standard_deviation": standard_deviation[band_index].tolist(),
            "coordinate_energy_over_d0": float(coordinate_energy[band_index] / d0_sum[band_index].clamp_min(1e-12)) if hilbert_geometry else None,
            "orthogonal_residual_over_d0": float(residual_sum[band_index] / d0_sum[band_index].clamp_min(1e-12)) if hilbert_geometry else None,
            "gram_ridge_or_jitter": model.tokenizer.gram_ridge,
            "eigenvalue_floor_ratio": model.tokenizer.eigenvalue_floor_ratio,
        })
    return {
        "status": "complete",
        "feature_mode": model.tokenizer.feature_mode,
        "direction_rank": model.tokenizer.direction_rank,
        "source_batches": observed_batches,
        "bands": rows,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_confusion(output: Path, split: str, epoch: int | str, metrics: dict[str, Any]) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    label = f"{int(epoch):03d}" if isinstance(epoch, int) else str(epoch)
    rows = [{"target": row, **{f"pred_{column}": int(matrix[row, column]) for column in range(matrix.shape[1])}} for row in range(matrix.shape[0])]
    _write_csv(output / "confusions" / f"{split}_epoch_{label}.csv", rows)


def _loaders(config: ExperimentConfig, run_root: Path, groups: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, DataLoader]:
    store = PackedTrialStore(config, run_root)
    value = settings(config)
    datasets = {name: TrialDataset(store, entries) for name, entries in groups.items()}
    sampler = SubjectBalancedBatchSampler(groups["train"], value["subjects_per_batch"], value["trials_per_subject"], seed)
    evaluation_batch = int(config.raw["training"].get("evaluation_batch_size", value["subjects_per_batch"] * value["trials_per_subject"]))
    return {
        "train_fit": DataLoader(datasets["train"], batch_sampler=sampler, collate_fn=collate_trials, num_workers=0),
        "train": DataLoader(datasets["train"], batch_size=evaluation_batch, shuffle=False, collate_fn=collate_trials, num_workers=0),
        "development": DataLoader(datasets["development"], batch_size=evaluation_batch, shuffle=False, collate_fn=collate_trials, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=evaluation_batch, shuffle=False, collate_fn=collate_trials, num_workers=0),
    }


def _task_signature(config: ExperimentConfig, condition: str, architecture: str, seed: int) -> str:
    return _hash({
        "family": FAMILY,
        "condition": condition,
        "architecture": architecture,
        "seed": seed,
        "model": config.raw["model"],
        "training": config.raw["training"],
        "slst": config.raw["slst"],
    }, length=12)


def _task_root(run_root: Path, dataset: str, protocol: str, fold: int, condition: str, architecture: str, seed: int, landmarks: int, task_signature: str) -> Path:
    return run_root / "tasks" / dataset / protocol / f"fold-{fold:02d}" / f"k-{landmarks}" / condition / architecture / f"seed-{seed}" / task_signature


def train_task(config: ExperimentConfig, run_root: Path, fold: int, protocol: str, condition: str, architecture: str, seed: int, *, resume: bool = True) -> dict[str, Any]:
    value = settings(config)
    if condition not in value["conditions"] or architecture not in value["architectures"] or seed not in value["seeds"]:
        raise ValueError(f"Task is not enabled by config: {condition}/{architecture}/seed-{seed}")
    task_signature = _task_signature(config, condition, architecture, seed)
    output = _task_root(run_root, config.dataset, protocol, fold, condition, architecture, seed, value["landmarks"], task_signature)
    result_path = output / "result.json"
    if resume and result_path.is_file() and read_json(result_path).get("status") == "complete":
        LOGGER.info(
            "RESUME-SKIP dataset=%s fold=%02d condition=%s architecture=%s seed=%d",
            config.dataset, fold, condition, architecture, seed,
        )
        return read_json(result_path)
    output.mkdir(parents=True, exist_ok=True)
    task_started_at = utc_now()
    write_json(output / "status.json", {
        "status": "running", "started_at": task_started_at, "dataset": config.dataset,
        "fold": fold, "condition": condition, "architecture": architecture, "seed": seed,
        "epoch": 0, "epochs": int(config.raw["training"]["epochs"]),
    })
    started = time.perf_counter()
    try:
        LOGGER.info(
            "TASK-START dataset=%s protocol=%s fold=%02d condition=%s architecture=%s seed=%d output=%s",
            config.dataset, protocol, fold, condition, architecture, seed, output,
        )
        groups = groups_for(config, fold, protocol)
        atlas, atlas_metadata = fit_atlas(config, run_root, fold, protocol)
        seed_everything(seed, bool(config.raw["training"].get("deterministic", True)))
        device = select_device(str(config.raw["training"].get("device", "auto")))
        model = build_model(config, atlas, condition, architecture).to(device)
        loaders = _loaders(config, run_root, groups, seed)
        specification = _condition(condition, value)
        source_diagnostics = source_coordinate_diagnostics(
            model,
            loaders["train"],
            device,
            tuple(config.raw["signal"]["bands_hz"]),
            value["source_diagnostic_batches"],
        )
        write_json(output / "coordinate_diagnostics_source_train.json", source_diagnostics)
        LOGGER.info(
            "COORD-DIAG dataset=%s fold=%02d condition=%s status=%s feature=%s batches=%s",
            config.dataset, fold, condition, source_diagnostics["status"],
            source_diagnostics["feature_mode"], source_diagnostics.get("source_batches", 0),
        )
        landmark_parameters = model.tokenizer.landmark_parameters()
        landmark_ids = {id(parameter) for parameter in landmark_parameters}
        base_parameters = [parameter for parameter in model.parameters() if id(parameter) not in landmark_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_parameters, "lr": float(config.raw["training"]["learning_rate"])},
                {"params": landmark_parameters, "lr": specification["landmark_learning_rate"], "weight_decay": 0.0},
            ],
            weight_decay=float(config.raw["training"]["weight_decay"]),
        )
        epochs = int(config.raw["training"]["epochs"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(config.raw["training"].get("minimum_learning_rate", 1e-6)))
        counts = np.bincount([int(entry["label"]) for entry in groups["train"]], minlength=int(config.raw["dataset"]["classes"])).astype(float)
        class_weights = counts.sum() / np.maximum(counts * len(counts), 1.0)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device), label_smoothing=float(config.raw["training"].get("label_smoothing", 0.05)), reduction="none")
        monitoring: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        best_development = -math.inf
        best_epoch = 0
        stale_monitors = 0
        accumulation = value["gradient_accumulation"]
        training_started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.tokenizer.set_landmarks_trainable(specification["learnable"] and epoch > specification["freeze"])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_started = time.perf_counter()
            landmark_before = [parameter.detach().clone() for parameter in landmark_parameters]
            loss_sum = 0.0
            examples = 0
            steps = 0
            raw_regularization = {"anchor": 0.0, "diversity": 0.0, "coverage": 0.0}
            gradient_diagnostics = {
                "classification": 0.0,
                "anchor": 0.0,
                "diversity": 0.0,
                "coverage": 0.0,
                "combined_before_clip": 0.0,
            }
            for step, batch in enumerate(loaders["train_fit"], 1):
                steps = step
                moved = _move(batch, device)
                logits = model(moved["shape"], moved["magnitude"], moved["de"], moved["mask"])
                example_loss = criterion(logits, moved["label"])
                classification_loss = example_loss.mean()
                vrex = logits.new_zeros(())
                if value["vrex_weight"] > 0:
                    subject_risks = [example_loss[moved["subject"] == subject].mean() for subject in torch.unique(moved["subject"])]
                    if len(subject_risks) > 1:
                        vrex = torch.stack(subject_risks).var(unbiased=False)
                regularization = model.tokenizer.regularization(moved["shape"], moved["mask"])
                anchor_weight, diversity_weight, coverage_weight = specification["weights"]
                if step == 1 and value["landmark_gradient_diagnostics"] and landmark_parameters:
                    gradient_diagnostics["classification"] = _gradient_norm(classification_loss, landmark_parameters)
                    gradient_diagnostics["anchor"] = _gradient_norm(regularization["anchor"], landmark_parameters)
                    gradient_diagnostics["diversity"] = _gradient_norm(regularization["diversity"], landmark_parameters)
                    gradient_diagnostics["coverage"] = _gradient_norm(regularization["coverage"], landmark_parameters)
                loss = classification_loss + value["vrex_weight"] * vrex
                loss = loss + anchor_weight * regularization["anchor"] + diversity_weight * regularization["diversity"] + coverage_weight * regularization["coverage"]
                (loss / accumulation).backward()
                if step == 1:
                    gradient_diagnostics["combined_before_clip"] = _existing_gradient_norm(landmark_parameters)
                if step % accumulation == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(config.raw["training"].get("gradient_clip_norm", 1.0)))
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                loss_sum += float(loss.detach().cpu()) * moved["label"].shape[0]
                examples += int(moved["label"].shape[0])
                for name in raw_regularization:
                    raw_regularization[name] += float(regularization[name].detach().cpu()) * moved["label"].shape[0]
            if steps % accumulation:
                nn.utils.clip_grad_norm_(model.parameters(), float(config.raw["training"].get("gradient_clip_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_norm = math.sqrt(sum(
                float((parameter.detach() - before).square().sum().cpu())
                for parameter, before in zip(landmark_parameters, landmark_before, strict=True)
            ))
            atlas_epoch = model.tokenizer.atlas_diagnostics()
            train_loss = loss_sum / max(examples, 1)
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "landmark_learning_rate": optimizer.param_groups[1]["lr"] if landmark_parameters else 0.0,
                "landmarks_trainable": bool(landmark_parameters and landmark_parameters[0].requires_grad),
                "landmark_update_l2": update_norm,
                "landmark_grad_classification_l2": gradient_diagnostics["classification"],
                "landmark_grad_anchor_l2": gradient_diagnostics["anchor"],
                "landmark_grad_diversity_l2": gradient_diagnostics["diversity"],
                "landmark_grad_coverage_l2": gradient_diagnostics["coverage"],
                "landmark_grad_combined_before_clip_l2": gradient_diagnostics["combined_before_clip"],
                "regularization_anchor_raw": raw_regularization["anchor"] / max(examples, 1),
                "regularization_diversity_raw": raw_regularization["diversity"] / max(examples, 1),
                "regularization_coverage_raw": raw_regularization["coverage"] / max(examples, 1),
                "regularization_anchor_weighted": anchor_weight * raw_regularization["anchor"] / max(examples, 1),
                "regularization_diversity_weighted": diversity_weight * raw_regularization["diversity"] / max(examples, 1),
                "regularization_coverage_weighted": coverage_weight * raw_regularization["coverage"] / max(examples, 1),
                **{f"atlas_{name}": metric for name, metric in atlas_epoch.items()},
            }
            history.append(history_row)
            _write_csv(output / "training_history.csv", history)
            elapsed_training = time.perf_counter() - training_started
            eta_seconds = elapsed_training / epoch * (epochs - epoch)
            current_status: dict[str, Any] = {
                "status": "running",
                "started_at": task_started_at,
                "updated_at": utc_now(),
                "dataset": config.dataset,
                "fold": fold,
                "condition": condition,
                "architecture": architecture,
                "seed": seed,
                "epoch": epoch,
                "epochs": epochs,
                "train_loss": train_loss,
                "eta_seconds": eta_seconds,
                "best_epoch": best_epoch,
                "best_source_development_subject_averaged_macro_f1": best_development if math.isfinite(best_development) else None,
            }
            write_json(output / "status.json", current_status)
            LOGGER.info(
                "EPOCH dataset=%s fold=%02d condition=%s epoch=%03d/%03d loss=%.5f lr=%.3g landmark_lr=%.3g "
                "landmark_trainable=%s drift_jsd=%.3g update_l2=%.3g epoch_sec=%.1f eta_min=%.1f",
                config.dataset, fold, condition, epoch, epochs, train_loss,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"] if landmark_parameters else 0.0,
                history_row["landmarks_trainable"], atlas_epoch["mean_jsd_drift"], update_norm,
                time.perf_counter() - epoch_started, eta_seconds / 60.0,
            )
            if epoch % value["monitor_interval"] == 0 or epoch == epochs:
                row: dict[str, Any] = {"epoch": epoch, "selection_metric": "source_development.subject_averaged_macro_f1"}
                splits = ("train", "development", "test") if value["target_monitoring"] else ("train", "development")
                for split in splits:
                    metrics, predictions = evaluate(model, loaders[split], device, int(config.raw["dataset"]["classes"]))
                    row[split] = metrics
                    _save_confusion(output, split, epoch, metrics)
                    _write_csv(output / "predictions" / f"{split}_epoch_{epoch:03d}.csv", predictions)
                development_score = float(row["development"]["subject_averaged_macro_f1"])
                if development_score > best_development + 1e-12:
                    best_development = development_score
                    best_epoch = epoch
                    stale_monitors = 0
                    torch.save({"model_state_dict": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, "epoch": epoch, "source_development_subject_averaged_macro_f1": development_score}, output / "best_model.pt")
                else:
                    stale_monitors += 1
                row["checkpoint_improved"] = best_epoch == epoch
                row["target_metrics_used_for_selection"] = False
                monitoring.append(row)
                write_json(output / "monitoring.json", monitoring)
                target_text = ""
                if value["target_monitoring"]:
                    target_text = " target_mf1=%.4f target_subject_mf1=%.4f" % (
                        row["test"]["macro_f1"], row["test"]["subject_averaged_macro_f1"],
                    )
                LOGGER.info(
                    "MONITOR dataset=%s fold=%02d condition=%s epoch=%03d train_mf1=%.4f "
                    "dev_mf1=%.4f dev_subject_mf1=%.4f best_dev=%.4f best_epoch=%03d improved=%s stale=%d%s",
                    config.dataset, fold, condition, epoch,
                    row["train"]["macro_f1"], row["development"]["macro_f1"],
                    row["development"]["subject_averaged_macro_f1"], best_development, best_epoch,
                    row["checkpoint_improved"], stale_monitors, target_text,
                )
                current_status.update({
                    "best_epoch": best_epoch,
                    "best_source_development_subject_averaged_macro_f1": best_development,
                    "current_source_train_macro_f1": row["train"]["macro_f1"],
                    "current_source_development_macro_f1": row["development"]["macro_f1"],
                    "current_source_development_subject_averaged_macro_f1": row["development"]["subject_averaged_macro_f1"],
                    "current_target_macro_f1": row["test"]["macro_f1"] if value["target_monitoring"] else None,
                    "current_target_subject_averaged_macro_f1": row["test"]["subject_averaged_macro_f1"] if value["target_monitoring"] else None,
                    "target_metrics_used_for_selection": False,
                })
                write_json(output / "status.json", current_status)
                if stale_monitors >= value["early_stopping_patience_monitors"]:
                    LOGGER.info(
                        "EARLY-STOP dataset=%s fold=%02d condition=%s epoch=%03d best_epoch=%03d",
                        config.dataset, fold, condition, epoch, best_epoch,
                    )
                    break
        checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics: dict[str, Any] = {}
        for split in ("train", "development", "test"):
            metrics, predictions = evaluate(model, loaders[split], device, int(config.raw["dataset"]["classes"]))
            final_metrics[split] = metrics
            _save_confusion(output, split, "selected", metrics)
            _write_csv(output / "predictions" / f"{split}_selected.csv", predictions)
        result = {
            "schema_version": SCHEMA_VERSION,
            "family": FAMILY,
            "status": "complete",
            "evidence_status": "exploratory_target_monitored_not_unbiased_formal_evidence" if value["target_monitoring"] else "source_selected_target_evaluated_after_selection",
            "dataset": config.dataset,
            "fold": fold,
            "protocol": protocol,
            "protocol_hash": atlas_metadata["protocol_hash"],
            "condition": condition,
            "architecture": architecture,
            "seed": seed,
            "task_signature": task_signature,
            "model": model.configuration(),
            "condition_specification": {
                **specification,
                "weights": list(specification["weights"]),
            },
            "best_epoch": best_epoch,
            "checkpoint_selection": "source-development subject-averaged Macro-F1 only",
            "target_metrics_used_for_selection": False,
            "target_monitored_every_epochs": value["monitor_interval"] if value["target_monitoring"] else None,
            "final_metrics": final_metrics,
            "atlas_diagnostics": model.tokenizer.atlas_diagnostics(),
            "source_coordinate_diagnostics": source_diagnostics,
            "source_train_subjects": sorted({int(entry["subject"]) for entry in groups["train"]}),
            "source_development_subjects": sorted({int(entry["subject"]) for entry in groups["development"]}),
            "target_subjects": sorted({int(entry["subject"]) for entry in groups["test"]}),
            "environment": environment_manifest([]),
            "elapsed_seconds": time.perf_counter() - started,
            "completed_at": utc_now(),
        }
        write_json(result_path, result)
        write_json(output / "status.json", {"status": "complete", "completed_at": utc_now()})
        LOGGER.info(
            "TASK-COMPLETE dataset=%s fold=%02d condition=%s best_epoch=%03d dev_subject_mf1=%.4f "
            "target_mf1=%.4f target_subject_mf1=%.4f elapsed_min=%.1f",
            config.dataset, fold, condition, best_epoch,
            final_metrics["development"]["subject_averaged_macro_f1"], final_metrics["test"]["macro_f1"],
            final_metrics["test"]["subject_averaged_macro_f1"], (time.perf_counter() - started) / 60.0,
        )
        return result
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "failed_at": utc_now(), "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()})
        LOGGER.exception(
            "TASK-FAILED dataset=%s fold=%02d condition=%s architecture=%s seed=%d",
            config.dataset, fold, condition, architecture, seed,
        )
        raise


def synthetic_smoke(config: ExperimentConfig, run_root: Path) -> dict[str, Any]:
    seed_everything(42, False)
    rng = np.random.default_rng(42)
    channels, bands, frequencies, landmarks = 3, 5, 7, 4
    mask = np.zeros((bands, frequencies), dtype=bool)
    mask[:, :5] = True
    candidates = rng.random((32, channels, bands, frequencies), dtype=np.float32) * mask[None, None]
    candidates /= candidates.sum(axis=-1, keepdims=True)
    center, anchors = fit_channel_band_atlas(candidates, mask, landmarks)
    atlas = HilbertAtlasState(center, anchors, mask, np.zeros((channels, bands), np.float32), np.ones((channels, bands), np.float32), np.zeros((channels, bands), np.float32), np.ones((channels, bands), np.float32))
    results: dict[str, Any] = {}
    smoke_conditions = (
        "A0_magnitude", "A1_de", "A2_full_shape", "A3_scalar_rjsd", "A4_raw_landmark",
        "A5_centered_landmark", "A6_hilbert_landmark", "C4_regularized",
        "H0_scalar_explicit", "H1_raw_inner_explicit", "H2_pca_lowrank_explicit",
        "H3_hilbert_lowrank_explicit", "H4_stable_hilbert_lowrank_explicit",
        "H5_hilbert_full_explicit", "H6_stable_hilbert_lowrank_residual", "L5_lr1e4_regularized",
    )
    for condition in smoke_conditions:
        specification = _condition(condition, settings(config))
        tokenizer = JSDHilbertTokenizer(torch.from_numpy(center), torch.from_numpy(anchors), torch.from_numpy(mask), torch.zeros(channels, bands), torch.ones(channels, bands), torch.zeros(channels, bands), torch.ones(channels, bands), feature_mode=specification["feature_mode"], learnable_landmarks=specification["learnable"], gram_ridge=1e-4)
        model = StructuredLandmarkSpectralTransformer(tokenizer, channels=channels, bands=bands, classes=int(config.raw["dataset"]["classes"]), max_length=4, architecture="B4_slst", d_model=16, band_heads=4, channel_heads=4, temporal_heads=4, band_layers=1, channel_layers=1, temporal_layers=1, feedforward=32, dropout=0.0)
        q = torch.from_numpy(candidates[:8].reshape(2, 4, channels, bands, frequencies))
        magnitude = torch.randn(2, 4, channels, bands)
        de = torch.randn_like(magnitude)
        time_mask = torch.ones(2, 4, dtype=torch.bool)
        logits = model(q, magnitude, de, time_mask)
        loss = logits.square().mean() + sum(model.tokenizer.regularization(q, time_mask).values())
        loss.backward()
        results[condition] = {"logits": list(logits.shape), "finite": bool(torch.isfinite(logits).all()), "parameters": model.parameter_count()}
    payload = {"status": "complete", "dataset_config": config.dataset, "conditions": results, "created_at": utc_now()}
    write_json(run_root / f"smoke_slst_{config.dataset}.json", payload)
    return payload


def matrix_status(run_root: Path) -> dict[str, Any]:
    rows = []
    task_root = run_root / "tasks"
    for path in sorted(task_root.glob("**/status.json")) if task_root.is_dir() else []:
        value = read_json(path)
        rows.append({"path": str(path.parent.relative_to(run_root)), **value})
    return {"run_root": str(run_root.resolve()), "tasks": rows, "counts": {status: sum(row.get("status") == status for row in rows) for status in ("complete", "running", "failed")}}


def summarize(run_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "tasks").glob("**/result.json")) if (run_root / "tasks").is_dir() else []:
        result = read_json(path)
        development = result["final_metrics"]["development"]
        test = result["final_metrics"]["test"]
        model = result["model"]
        atlas = result.get("atlas_diagnostics", {})
        rows.append({
            "dataset": result["dataset"], "protocol": result["protocol"], "fold": result["fold"],
            "condition": result["condition"], "feature_mode": model.get("feature_mode"),
            "architecture": result["architecture"], "seed": result["seed"],
            "landmarks": model.get("landmarks"), "direction_rank": model.get("direction_rank"),
            "eigenvalue_floor_ratio": model.get("eigenvalue_floor_ratio"),
            "source_development_subject_averaged_macro_f1": development["subject_averaged_macro_f1"],
            "accuracy": test["accuracy"], "balanced_accuracy": test["balanced_accuracy"],
            "macro_f1": test["macro_f1"], "subject_averaged_macro_f1": test["subject_averaged_macro_f1"],
            "worst_quartile_subject_macro_f1": test["worst_quartile_subject_macro_f1"],
            "mean_landmark_jsd_drift": atlas.get("mean_jsd_drift"), "best_epoch": result["best_epoch"],
            "parameters": model["parameters"], "task_signature": result.get("task_signature"),
        })
    _write_csv(run_root / "summary.csv", rows)
    payload = {"status": "complete", "tasks": len(rows), "summary_csv": str((run_root / "summary.csv").resolve()), "evidence_status": "exploratory_target_monitored", "updated_at": utc_now()}
    write_json(run_root / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured Landmark Spectral Transformer experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name: str) -> argparse.ArgumentParser:
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--run-root")
        command.add_argument("--fold", type=int, action="append")
        command.add_argument("--protocol", action="append")
        command.add_argument("--set", action="append", dest="overrides")
        return command

    for name in ("validate", "prepare-base", "prepare-pack", "prepare-atlas", "smoke"):
        configured(name)
    run = configured("run")
    run.add_argument("--condition", required=True)
    run.add_argument("--architecture", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--no-resume", action="store_true")
    matrix = configured("matrix")
    matrix.add_argument("--condition", action="append")
    matrix.add_argument("--architecture", action="append")
    matrix.add_argument("--seed", type=int, action="append")
    matrix.add_argument("--no-resume", action="store_true")
    for name in ("status", "summarize"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_parser().parse_args(argv)
    if args.command in {"status", "summarize"}:
        payload = matrix_status(Path(args.run_root)) if args.command == "status" else summarize(Path(args.run_root))
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    config = load_config(args.config, args.overrides)
    value = settings(config)
    run_root = Path(args.run_root).resolve() if args.run_root else config.run_root
    folds = tuple(args.fold or value["folds"])
    protocols = tuple(args.protocol or value["protocols"])
    if args.command == "validate":
        payload = validate_sources(config)
        payload["slst_settings"] = {key: list(item) if isinstance(item, tuple) else item for key, item in value.items()}
    elif args.command == "prepare-base":
        payload = prepare_base_cache(config, _base_cache_run_root(config, run_root), folds)
    elif args.command == "prepare-pack":
        prepare_base_cache(config, _base_cache_run_root(config, run_root), folds)
        payload = prepare_structured_cache(config, run_root)
    elif args.command == "prepare-atlas":
        prepare_base_cache(config, _base_cache_run_root(config, run_root), folds)
        prepare_structured_cache(config, run_root)
        payload = {f"{protocol}/fold-{fold:02d}": fit_atlas(config, run_root, fold, protocol)[1] for fold in folds for protocol in protocols}
    elif args.command == "smoke":
        payload = synthetic_smoke(config, run_root)
    elif args.command == "run":
        prepare_base_cache(config, _base_cache_run_root(config, run_root), [args.fold[0] if args.fold else folds[0]])
        prepare_structured_cache(config, run_root)
        payload = train_task(config, run_root, args.fold[0] if args.fold else folds[0], args.protocol[0] if args.protocol else protocols[0], args.condition, args.architecture, args.seed, resume=not args.no_resume)
    else:
        prepare_base_cache(config, _base_cache_run_root(config, run_root), folds)
        prepare_structured_cache(config, run_root)
        completed = failed = 0
        selected_conditions = tuple(args.condition or value["conditions"])
        selected_architectures = tuple(args.architecture or value["architectures"])
        selected_seeds = tuple(args.seed or value["seeds"])
        total_tasks = len(folds) * len(protocols) * len(selected_conditions) * len(selected_architectures) * len(selected_seeds)
        task_index = 0
        LOGGER.info(
            "MATRIX-PLAN dataset=%s folds=%s protocols=%s conditions=%d architectures=%d seeds=%s total_tasks=%d run_root=%s",
            config.dataset, list(folds), list(protocols), len(selected_conditions), len(selected_architectures),
            list(selected_seeds), total_tasks, run_root,
        )
        for fold in folds:
            for protocol in protocols:
                fit_atlas(config, run_root, fold, protocol)
                for condition in selected_conditions:
                    for architecture in selected_architectures:
                        for seed in selected_seeds:
                            task_index += 1
                            LOGGER.info(
                                "MATRIX-TASK %d/%d dataset=%s fold=%02d condition=%s architecture=%s seed=%d",
                                task_index, total_tasks, config.dataset, fold, condition, architecture, seed,
                            )
                            try:
                                train_task(config, run_root, fold, protocol, condition, architecture, seed, resume=not args.no_resume)
                                completed += 1
                            except BaseException:
                                failed += 1
                                LOGGER.exception("SLST task failed: %s/%s/fold-%02d/%s/%s/seed-%d", config.dataset, protocol, fold, condition, architecture, seed)
        payload = {"status": "complete" if failed == 0 else "partial", "planned_tasks": total_tasks, "processed_tasks": completed + failed, "completed_tasks": completed, "failed_tasks": failed, "evidence_status": "exploratory_target_monitored", "updated_at": utc_now()}
        write_json(run_root / f"matrix_slst_{config.dataset}.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0
