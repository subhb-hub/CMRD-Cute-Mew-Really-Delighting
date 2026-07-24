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
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from cmrd.config import ExperimentConfig, load_config
from cmrd.data.records import TrialSample
from cmrd.faced import (
    EEG_CHANNELS,
    EMOTION_NAMES,
    FOLDS,
    SUBJECTS,
    VIDEO_LABELS,
    VIDEOS,
    official_fold_subjects,
)
from cmrd.faced_runner import (
    _load_spectra,
    _spectra_root,
    validate_sources as validate_faced_sources,
)
from cmrd.features.rd import (
    fisher_rao_log_map,
    fit_balanced_multiclass_lda_from_moments,
    native_frequency_grid,
    transform_native_fisher_rao_supervised,
    transform_native_frequency_point_rjsd,
)
from cmrd.fixed_protocol import (
    _autocast,
    _loader,
    _shutdown_persistent_loader,
    _valid_indices_on_device,
    scaling_statistics,
)
from cmrd.io import read_json, write_json, write_npz
from cmrd.models import (
    FrequencyPointChannelBandTransformer,
    HierarchicalChannelBandTransformer,
    VectorBandHierarchicalChannelTransformer,
)
from cmrd.training.metrics import classification_metrics
from cmrd.training.runtime import seed_everything, select_device


LOGGER = logging.getLogger("cmrd.faced_relative")
SCHEMA_VERSION = 1
FAMILY = "FACED-Relative-Supervised-Monitor-v1"
VECTOR_FAMILY = "FACED-Vector-Preserving-Supervised-Monitor-v1"
SEED = 42
CONDITIONS = {
    "frequency_point_rjsd_base": "frequency_point_rjsd",
    "fisher_rao_supervised_lda2_base": "fisher_rao_supervised_lda",
}
VECTOR_CONDITIONS = {
    "frequency_vector_rjsd_base": "frequency_point_rjsd",
    "fisher_rao_supervised_full_vector_base": "fisher_rao_supervised_lda",
}
SUPPORTED_CONDITIONS = {**CONDITIONS, **VECTOR_CONDITIONS}


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


def relative_settings(config: ExperimentConfig) -> dict[str, Any]:
    if config.dataset != "faced":
        raise ValueError("FACED relative runner requires experiment.dataset=faced")
    raw = copy.deepcopy(config.raw.get("faced_relative", {}))
    required = {
        "base_cache_root",
        "seed",
        "supervised_components",
        "lda_regularization",
        "feature_storage_dtype",
        "frequency_encoder_hidden",
        "monitor_interval",
        "monitor_epoch_zero",
        "conditions",
        "architecture",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"Missing faced_relative settings: {missing}")
    if int(raw["seed"]) != SEED:
        raise ValueError("FACED monitored relative experiments are frozen to seed=42")
    declared_conditions = tuple(raw["conditions"])
    if declared_conditions not in (tuple(CONDITIONS), tuple(VECTOR_CONDITIONS)):
        raise ValueError(
            "Conditions must be one supported ordered preset: "
            f"{tuple(CONDITIONS)} or {tuple(VECTOR_CONDITIONS)}"
        )
    if "supervised_components_by_band" in raw:
        band_names = tuple(config.raw["signal"]["bands_hz"])
        declared_bands = tuple(raw["supervised_components_by_band"])
        if declared_bands != band_names:
            raise ValueError(f"supervised_components_by_band must follow {band_names}")
        if any(int(value) < 1 for value in raw["supervised_components_by_band"].values()):
            raise ValueError("Every supervised component count must be positive")
    elif int(raw["supervised_components"]) < 1:
        raise ValueError("supervised_components must be positive")
    if float(raw["lda_regularization"]) <= 0:
        raise ValueError("lda_regularization must be positive")
    if str(raw["feature_storage_dtype"]) not in {"float16", "float32"}:
        raise ValueError("feature_storage_dtype must be float16 or float32")
    if int(raw["monitor_interval"]) < 1:
        raise ValueError("monitor_interval must be positive")
    active = CONDITIONS if declared_conditions == tuple(CONDITIONS) else VECTOR_CONDITIONS
    for condition, representation in active.items():
        if raw["conditions"][condition].get("representation") != representation:
            raise ValueError(f"{condition} must use {representation}")
    architecture = raw["architecture"]
    expected_architecture = {
        "d_model": 128,
        "heads": 4,
        "layers": 3,
        "feedforward": 512,
    }
    for key, expected in expected_architecture.items():
        if float(architecture.get(key, float("nan"))) != float(expected):
            raise ValueError(f"Base architecture requires {key}={expected}")
    dropout = float(architecture.get("dropout", float("nan")))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Base architecture dropout must be in [0,1)")
    if active is CONDITIONS and dropout != 0.15:
        raise ValueError("Legacy FACED monitored training requires architecture.dropout=0.15")
    training = config.raw["training"]
    if active is CONDITIONS:
        for key, expected in {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "minimum_learning_rate": 1e-6,
            "weight_decay": 0.01,
            "label_smoothing": 0.05,
            "gradient_clip_norm": 1.0,
        }.items():
            if float(training.get(key, float("nan"))) != float(expected):
                raise ValueError(f"Legacy FACED monitored training requires {key}={expected}")
    else:
        for key in ("epochs", "batch_size", "learning_rate", "gradient_clip_norm"):
            if float(training.get(key, 0.0)) <= 0:
                raise ValueError(f"Adapted FACED monitored training requires positive {key}")
        if float(training.get("minimum_learning_rate", -1.0)) < 0:
            raise ValueError("minimum_learning_rate must be non-negative")
        if not 0.0 <= float(training.get("label_smoothing", 0.0)) < 1.0:
            raise ValueError("label_smoothing must be in [0,1)")
    if training.get("deterministic") is not True or list(map(int, training.get("seeds", []))) != [SEED]:
        raise ValueError("FACED monitored training requires deterministic=true and seeds=[42]")
    return raw


def _active_conditions(config: ExperimentConfig) -> dict[str, str]:
    settings = relative_settings(config)
    return {
        name: str(specification["representation"])
        for name, specification in settings["conditions"].items()
    }


def _family(config: ExperimentConfig) -> str:
    return VECTOR_FAMILY if tuple(_active_conditions(config)) == tuple(VECTOR_CONDITIONS) else FAMILY


def _supervised_components_by_band(config: ExperimentConfig) -> list[int]:
    settings = relative_settings(config)
    band_sizes = [
        len(grid)
        for grid in native_frequency_grid(250, 1.0, config.raw["signal"]["bands_hz"])
    ]
    if "supervised_components_by_band" in settings:
        values = [
            int(settings["supervised_components_by_band"][name])
            for name in config.raw["signal"]["bands_hz"]
        ]
    else:
        values = [int(settings["supervised_components"])] * len(band_sizes)
    class_limit = len(EMOTION_NAMES) - 1
    for name, value, frequencies in zip(
        config.raw["signal"]["bands_hz"], values, band_sizes, strict=True
    ):
        maximum = min(class_limit, frequencies)
        if value > maximum:
            raise ValueError(f"{name} supervised components must be <= {maximum}, got {value}")
    return values


def _base_root(config: ExperimentConfig) -> Path:
    return config.resolve_path(relative_settings(config)["base_cache_root"])


def _load_base_fold_state(
    config: ExperimentConfig,
    fold: int,
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any], str]:
    base_root = _base_root(config)
    lock_path = base_root / "experiment_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing completed FACED base lock: {lock_path}")
    base_hash = str(read_json(lock_path)["protocol_hash"])
    state_root = base_root / "cache" / f"fold-{fold:02d}" / base_hash
    state_path = state_root / "feature_state.npz"
    metadata_path = state_root / "feature_state.json"
    if not state_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing source-only base reference state: {state_root}")
    metadata = read_json(metadata_path)
    source, target = official_fold_subjects(fold)
    if metadata.get("source_subjects") != source or metadata.get("target_subjects") != target:
        raise ValueError(f"Base reference state does not match official fold {fold}")
    band_names = list(config.raw["signal"]["bands_hz"])
    with np.load(state_path, allow_pickle=False) as archive:
        state = {
            key: [
                np.asarray(archive[f"{key}_{name}"], dtype=np.float32)
                for name in band_names
            ]
            for key in ("arithmetic", "fisher")
        }
    return state, metadata, base_hash


def validate_sources(config: ExperimentConfig, *, deep: bool = False) -> dict[str, Any]:
    settings = relative_settings(config)
    # The reusable native runner freezes its historical optimization recipe as
    # part of its validator.  Vector experiments intentionally change only the
    # model/training recipe, so validate the identical data declarations using
    # a temporary legacy training view while the active recipe is validated by
    # ``relative_settings`` above and retained in this protocol hash.
    validation_config = config
    if tuple(_active_conditions(config)) == tuple(VECTOR_CONDITIONS):
        validation_raw = copy.deepcopy(config.raw)
        validation_raw["training"].update({
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "minimum_learning_rate": 1e-6,
            "weight_decay": 0.01,
            "label_smoothing": 0.05,
            "gradient_clip_norm": 1.0,
        })
        validation_config = ExperimentConfig(config.path, validation_raw)
    audit = validate_faced_sources(validation_config, deep=deep)
    spectra_manifest = _spectra_root(config, _base_root(config)) / "manifest.json"
    if not spectra_manifest.is_file():
        raise FileNotFoundError(f"Missing reusable native spectra manifest: {spectra_manifest}")
    spectra = read_json(spectra_manifest)
    if not spectra.get("all_subjects_complete"):
        raise RuntimeError("Reusable FACED native spectra cache is incomplete")
    base_hashes = []
    for fold in range(1, FOLDS + 1):
        _, _, base_hash = _load_base_fold_state(config, fold)
        base_hashes.append(base_hash)
    audit.update({
        "family": _family(config),
        "base_cache_root": str(_base_root(config)),
        "base_protocol_hashes": sorted(set(base_hashes)),
        "native_spectra_manifest": str(spectra_manifest.resolve()),
        "relative_conditions": settings["conditions"],
        "target_monitoring_interval_epochs": int(settings["monitor_interval"]),
        "target_used_for_gradients": False,
        "target_used_for_checkpoint_selection": False,
        "evidence_label": "exploratory target-monitored",
    })
    return audit


def protocol_payload(config: ExperimentConfig, audit: dict[str, Any]) -> dict[str, Any]:
    settings = relative_settings(config)
    grids = native_frequency_grid(250, 1.0, config.raw["signal"]["bands_hz"])
    vector_preserving = tuple(_active_conditions(config)) == tuple(VECTOR_CONDITIONS)
    return {
        "schema_version": SCHEMA_VERSION,
        "family": _family(config),
        "dataset": "faced",
        "base_cache_root": str(_base_root(config)),
        "base_protocol_hashes": audit["base_protocol_hashes"],
        "fold_subjects": {
            f"fold-{fold:02d}": official_fold_subjects(fold)[1]
            for fold in range(1, FOLDS + 1)
        },
        "frequency_grids_hz": [grid.astype(float).tolist() for grid in grids],
        "frequency_point_rjsd": (
            "sign(P_f-Q_f)*sqrt(0.5*P_f*log(P_f/M_f)+0.5*Q_f*log(Q_f/M_f)); "
            "single outer-source arithmetic Q; whole physical-band vector encoder"
            if vector_preserving
            else
            "sign(P_f-Q_f)*sqrt(0.5*P_f*log(P_f/M_f)+0.5*Q_f*log(Q_f/M_f)); "
            "single outer-source arithmetic Q; learned within-band frequency encoder"
        ),
        "fisher_rao_supervised": {
            "reference": "single outer-source normalized mean square-root embedding",
            "projection": "class-balanced regularized multiclass LDA on outer-source tangent vectors",
            "components_per_channel_band": (
                {
                    name: value
                    for name, value in zip(
                        config.raw["signal"]["bands_hz"],
                        _supervised_components_by_band(config),
                        strict=True,
                    )
                }
                if vector_preserving
                else int(settings["supervised_components"])
            ),
            "regularization": float(settings["lda_regularization"]),
            "target_labels_used": False,
        },
        "feature_storage_dtype": str(settings["feature_storage_dtype"]),
        "conditions": settings["conditions"],
        "architecture": settings["architecture"],
        "training": config.raw["training"],
        "monitor_interval": int(settings["monitor_interval"]),
        "monitor_epoch_zero": bool(settings["monitor_epoch_zero"]),
        "target_monitored_during_training": True,
        "target_used_for_gradients": False,
        "target_used_for_checkpoint_selection": False,
        "checkpoint_selection": "fixed final epoch only",
        "evidence_label": "exploratory target-monitored",
    }


def lock_experiment(config: ExperimentConfig, run_root: Path) -> dict[str, Any]:
    audit = validate_sources(config)
    protocol = protocol_payload(config, audit)
    protocol_hash = _json_hash(protocol)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": _family(config),
        "protocol_hash": protocol_hash,
        "target_monitored_during_training": True,
        "target_used_for_gradients": False,
        "target_used_for_checkpoint_selection": False,
        "checkpoint_selection": "fixed final epoch only",
        "locked_at": utc_now(),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "experiment_lock.json", payload)
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
        raise ValueError("FACED relative experiment lock does not match active config/base cache")
    return audit, expected


def _fold_cache_root(run_root: Path, fold: int, protocol_hash: str) -> Path:
    return run_root / "cache" / f"fold-{fold:02d}" / protocol_hash


def _supervised_state_path(run_root: Path, fold: int, protocol_hash: str) -> Path:
    return _fold_cache_root(run_root, fold, protocol_hash) / "supervised_fisher_state.npz"


def _fit_supervised_fisher_state(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol_hash: str,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    root = _fold_cache_root(run_root, fold, protocol_hash)
    state_path = _supervised_state_path(run_root, fold, protocol_hash)
    metadata_path = root / "supervised_fisher_state.json"
    settings = relative_settings(config)
    components_by_band = _supervised_components_by_band(config)
    band_names = list(config.raw["signal"]["bands_hz"])
    source, target = official_fold_subjects(fold)
    if state_path.is_file() and metadata_path.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("source_subjects") == source:
            with np.load(state_path, allow_pickle=False) as archive:
                centers = [np.asarray(archive[f"center_{name}"], dtype=np.float32) for name in band_names]
                axes = [np.asarray(archive[f"axes_{name}"], dtype=np.float32) for name in band_names]
            return centers, axes, metadata

    base_state, _, base_hash = _load_base_fold_state(config, fold)
    references = base_state["fisher"]
    class_count = len(EMOTION_NAMES)
    counts = [np.zeros(class_count, dtype=np.int64) for _ in references]
    sums = [
        np.zeros((class_count, EEG_CHANNELS, reference.shape[-1]), dtype=np.float64)
        for reference in references
    ]
    crosses = [
        np.zeros(
            (class_count, EEG_CHANNELS, reference.shape[-1], reference.shape[-1]),
            dtype=np.float64,
        )
        for reference in references
    ]
    window_labels = np.repeat(VIDEO_LABELS, 30)
    for index, subject in enumerate(source, 1):
        spectra = _load_spectra(config, _base_root(config), subject)
        for band, (values, reference) in enumerate(zip(spectra, references, strict=True)):
            tangent = fisher_rao_log_map(
                values.reshape(-1, EEG_CHANNELS, values.shape[-1]), reference
            )
            for label in range(class_count):
                selected = tangent[window_labels == label]
                counts[band][label] += selected.shape[0]
                sums[band][label] += selected.sum(axis=0, dtype=np.float64)
                crosses[band][label] += np.einsum(
                    "ncf,ncg->cfg", selected, selected, dtype=np.float64
                )
        if index % 12 == 0 or index == len(source):
            LOGGER.info("Fold %02d supervised Fisher moments %d/%d subjects", fold, index, len(source))

    centers: list[np.ndarray] = []
    axes: list[np.ndarray] = []
    captured: dict[str, list[list[float]]] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, band_counts, band_sums, band_crosses, components in zip(
        band_names, counts, sums, crosses, components_by_band, strict=True
    ):
        center, projection, ratio = fit_balanced_multiclass_lda_from_moments(
            band_counts,
            band_sums,
            band_crosses,
            components,
            float(settings["lda_regularization"]),
        )
        centers.append(center)
        axes.append(projection)
        captured[name] = ratio.astype(float).tolist()
        arrays[f"center_{name}"] = center
        arrays[f"axes_{name}"] = projection
    write_npz(state_path, **arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fold": fold,
        "protocol_hash": protocol_hash,
        "base_protocol_hash": base_hash,
        "source_subjects": source,
        "target_subjects": target,
        "classes": list(EMOTION_NAMES),
        "class_window_counts_by_band": {
            name: value.astype(int).tolist() for name, value in zip(band_names, counts, strict=True)
        },
        "components_by_band": dict(zip(band_names, components_by_band, strict=True)),
        "regularization": float(settings["lda_regularization"]),
        "captured_positive_generalized_eigenvalue_ratio_by_channel": captured,
        "class_balanced": True,
        "target_data_used": False,
        "created_at": utc_now(),
    }
    write_json(metadata_path, metadata)
    return centers, axes, metadata


def _feature_dimensions(config: ExperimentConfig) -> tuple[int, int, list[int]]:
    band_sizes = [
        len(grid)
        for grid in native_frequency_grid(250, 1.0, config.raw["signal"]["bands_hz"])
    ]
    components = _supervised_components_by_band(config)
    return EEG_CHANNELS * sum(band_sizes), EEG_CHANNELS * sum(components), band_sizes


def _valid_subject_feature(config: ExperimentConfig, path: Path) -> bool:
    if not path.is_file():
        return False
    point_dim, fisher_dim, _ = _feature_dimensions(config)
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                archive["frequency_point_rjsd"].shape == (VIDEOS, 30, point_dim)
                and archive["fisher_rao_supervised_lda"].shape == (VIDEOS, 30, fisher_dim)
                and np.isfinite(archive["frequency_point_rjsd"]).all()
                and np.isfinite(archive["fisher_rao_supervised_lda"]).all()
            )
    except (OSError, ValueError, KeyError):
        return False


def prepare_fold_features(
    config: ExperimentConfig,
    run_root: Path,
    fold: int,
    protocol_hash: str | None = None,
    *,
    subjects: Sequence[int] | None = None,
) -> dict[str, Any]:
    if protocol_hash is None:
        _, protocol_hash = _require_lock(config, run_root)
    if not 1 <= int(fold) <= FOLDS:
        raise ValueError("FACED fold must be between 1 and 10")
    source, target = official_fold_subjects(fold)
    selected_subjects = list(range(SUBJECTS)) if subjects is None else sorted(set(map(int, subjects)))
    if any(subject < 0 or subject >= SUBJECTS for subject in selected_subjects):
        raise ValueError("Invalid FACED subject in feature request")
    settings = relative_settings(config)
    storage_dtype = np.dtype(str(settings["feature_storage_dtype"]))
    base_state, _, base_hash = _load_base_fold_state(config, fold)
    centers, axes, supervised_metadata = _fit_supervised_fisher_state(
        config, run_root, fold, protocol_hash
    )
    root = _fold_cache_root(run_root, fold, protocol_hash)
    point_dim, fisher_dim, band_sizes = _feature_dimensions(config)
    for index, subject in enumerate(selected_subjects, 1):
        output = root / "subjects" / f"sub{subject:03d}.npz"
        if _valid_subject_feature(config, output):
            continue
        spectra = _load_spectra(config, _base_root(config), subject)
        flattened = [
            values.reshape(-1, EEG_CHANNELS, values.shape[-1]) for values in spectra
        ]
        pointwise = transform_native_frequency_point_rjsd(
            flattened, base_state["arithmetic"]
        ).reshape(VIDEOS, 30, point_dim)
        fisher = transform_native_fisher_rao_supervised(
            flattened, base_state["fisher"], centers, axes
        ).reshape(VIDEOS, 30, fisher_dim)
        write_npz(
            output,
            frequency_point_rjsd=pointwise.astype(storage_dtype),
            fisher_rao_supervised_lda=fisher.astype(storage_dtype),
        )
        if index % 12 == 0 or index == len(selected_subjects):
            LOGGER.info("Fold %02d relative feature pass %d/%d subjects", fold, index, len(selected_subjects))

    complete = [
        subject
        for subject in selected_subjects
        if _valid_subject_feature(config, root / "subjects" / f"sub{subject:03d}.npz")
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fold": fold,
        "protocol_hash": protocol_hash,
        "base_protocol_hash": base_hash,
        "source_state_subjects": source,
        "official_target_subjects": target,
        "subjects_requested": selected_subjects,
        "subjects_complete": complete,
        "representations": sorted(set(_active_conditions(config).values())),
        "frequency_band_sizes": band_sizes,
        "frequency_point_feature_shape": [VIDEOS, 30, point_dim],
        "fisher_supervised_feature_shape": [VIDEOS, 30, fisher_dim],
        "supervised_components_by_band": dict(zip(
            config.raw["signal"]["bands_hz"],
            _supervised_components_by_band(config),
            strict=True,
        )),
        "target_used_to_fit_reference_or_projection": False,
        "supervised_state": supervised_metadata,
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
        path = root / f"sub{int(subject):03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing relative feature {path}; run PrepareFeatures")
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive[representation])
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


def _build_model(
    config: ExperimentConfig,
    condition: str,
    representation: str,
    input_dim: int,
    max_length: int,
) -> nn.Module:
    settings = relative_settings(config)
    architecture = settings["architecture"]
    common = {
        "input_dim": input_dim,
        "channels": EEG_CHANNELS,
        "classes": len(EMOTION_NAMES),
        "max_length": max_length,
        "d_model": int(architecture["d_model"]),
        "channel_heads": int(architecture["heads"]),
        "temporal_heads": int(architecture["heads"]),
        "temporal_layers": int(architecture["layers"]),
        "feedforward": int(architecture["feedforward"]),
        "dropout": float(architecture["dropout"]),
    }
    if condition in VECTOR_CONDITIONS:
        if representation == "frequency_point_rjsd":
            band_sizes = _feature_dimensions(config)[2]
        else:
            band_sizes = _supervised_components_by_band(config)
        return VectorBandHierarchicalChannelTransformer(
            **common,
            band_sizes=band_sizes,
        )
    if representation == "frequency_point_rjsd":
        _, _, band_sizes = _feature_dimensions(config)
        return FrequencyPointChannelBandTransformer(
            **common,
            band_sizes=band_sizes,
            frequency_hidden=int(settings["frequency_encoder_hidden"]),
        )
    return HierarchicalChannelBandTransformer(**common)


def _forward(model: nn.Module, data: torch.Tensor, mask: torch.Tensor, valid_indices: torch.Tensor):
    return model(data, mask, valid_indices=valid_indices)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    precision: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for data, mask, labels in loader:
        valid_indices = _valid_indices_on_device(mask, device)
        with _autocast({"precision": precision}, device):
            logits = _forward(
                model,
                data.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                valid_indices,
            )
        targets.append(labels.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    y = np.concatenate(targets)
    pred = np.concatenate(predictions)
    metrics = classification_metrics(y, pred, len(EMOTION_NAMES))
    counts = np.bincount(pred, minlength=len(EMOTION_NAMES))
    metrics["prediction_counts"] = counts.astype(int).tolist()
    metrics["prediction_fractions"] = (counts / len(pred)).astype(float).tolist()
    metrics["dominant_prediction_fraction"] = float(counts.max() / len(pred))
    return metrics, y, pred


def _plot_confusion(ax, matrix: np.ndarray, title: str, cmap: str) -> None:
    matrix = np.asarray(matrix, dtype=int)
    ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "black",
            )
    labels = [name[:4] for name in EMOTION_NAMES]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="True", title=title)


def _save_monitor_plot(
    output: Path,
    condition: str,
    fold: int,
    epoch: int,
    monitor_rows: Sequence[dict[str, Any]],
    source_metrics: dict[str, Any],
    target_metrics: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    epochs = [int(row["epoch"]) for row in monitor_rows]
    axes[0, 0].plot(epochs, [100 * float(row["source_accuracy"]) for row in monitor_rows], marker="o", label="source")
    axes[0, 0].plot(epochs, [100 * float(row["target_accuracy"]) for row in monitor_rows], marker="o", label="target monitor")
    axes[0, 0].axhline(100 / 7, color="gray", linestyle="--", label="majority baseline")
    axes[0, 0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Monitored accuracy")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, [float(row["source_macro_f1"]) for row in monitor_rows], marker="o", label="source")
    axes[0, 1].plot(epochs, [float(row["target_macro_f1"]) for row in monitor_rows], marker="o", label="target monitor")
    axes[0, 1].axhline(1 / 9, color="gray", linestyle=":", label="balanced chance")
    axes[0, 1].set(xlabel="Epoch", ylabel="Macro-F1", title="Monitored macro-F1")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()
    _plot_confusion(
        axes[1, 0],
        np.asarray(source_metrics["confusion_matrix"]),
        f"Source CM - acc={source_metrics['accuracy']:.3f}",
        "Blues",
    )
    _plot_confusion(
        axes[1, 1],
        np.asarray(target_metrics["confusion_matrix"]),
        f"Target CM (monitor only) - acc={target_metrics['accuracy']:.3f}",
        "Oranges",
    )
    fig.suptitle(f"{condition} - fold {fold}, epoch {epoch}", fontsize=14)
    fig.savefig(output / f"monitor_epoch_{epoch:03d}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    active_conditions = _active_conditions(config)
    if condition not in active_conditions:
        raise ValueError(f"Unknown FACED relative condition: {condition}")
    started = time.perf_counter()
    source, target = official_fold_subjects(fold)
    if source_override is not None:
        source = list(source_override)
    if target_override is not None:
        target = list(target_override)
    representation = active_conditions[condition]
    output = _task_output(run_root, condition, fold)
    output.mkdir(parents=True, exist_ok=True)
    training = copy.deepcopy(config.raw["training"])
    total_epochs = int(epochs or training["epochs"])
    settings = relative_settings(config)
    monitor_interval = int(settings["monitor_interval"])
    device = select_device(str(training.get("device", "auto")))
    seed_everything(SEED, bool(training.get("deterministic", True)))
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(training.get("matmul_precision", "high")))

    # Explicitly target-monitored exploratory training. Target samples enter
    # evaluation only and never affect gradients, the scheduler, or checkpoint
    # selection (the endpoint is fixed before training starts).
    source_samples = _load_samples(config, run_root, fold, protocol_hash, source, representation)
    target_samples = _load_samples(config, run_root, fold, protocol_hash, target, representation)
    mean, std = scaling_statistics(source_samples, True)
    input_dim = int(source_samples[0].x.shape[1])
    max_length = max(sample.x.shape[0] for sample in source_samples + target_samples)
    model = _build_model(config, condition, representation, input_dim, max_length).to(device)
    cache_normalized = representation != "frequency_point_rjsd"
    workers = int(training.get("num_workers", 0))
    source_loader = _loader(
        source_samples,
        mean,
        std,
        int(training["batch_size"]),
        True,
        SEED,
        num_workers=workers,
        persistent_workers=bool(training.get("persistent_workers", False)),
        prefetch_factor=int(training.get("prefetch_factor", 1)),
        cache_normalized=cache_normalized,
    )
    source_eval_loader = _loader(
        source_samples, mean, std, int(training["batch_size"]), False, SEED,
        cache_normalized=False,
    )
    target_loader = _loader(
        target_samples, mean, std, int(training["batch_size"]), False, SEED,
        cache_normalized=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    class_weights: torch.Tensor | None = None
    if bool(training.get("class_balanced_loss", False)):
        label_counts = np.bincount(
            [sample.label for sample in source_samples], minlength=len(EMOTION_NAMES)
        ).astype(np.float64)
        if np.any(label_counts == 0):
            raise ValueError("Class-balanced loss requires every source class")
        balanced = label_counts.sum() / (len(EMOTION_NAMES) * label_counts)
        class_weights = torch.as_tensor(balanced, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(training.get("label_smoothing", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs, 1),
        eta_min=float(training.get("minimum_learning_rate", 1e-6)),
    )
    precision = str(training.get("precision", "float32"))
    use_scaler = device.type == "cuda" and precision == "float16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    training_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    final_source_metrics: dict[str, Any] | None = None
    final_target_metrics: dict[str, Any] | None = None
    final_target_y: np.ndarray | None = None
    final_target_prediction: np.ndarray | None = None

    def monitor(epoch: int) -> None:
        nonlocal final_source_metrics, final_target_metrics, final_target_y, final_target_prediction
        source_metrics, _, _ = _evaluate(model, source_eval_loader, device, precision)
        target_metrics, target_y, target_prediction = _evaluate(model, target_loader, device, precision)
        row = {
            "epoch": epoch,
            "source_accuracy": float(source_metrics["accuracy"]),
            "source_balanced_accuracy": float(source_metrics["balanced_accuracy"]),
            "source_macro_f1": float(source_metrics["macro_f1"]),
            "source_dominant_prediction_fraction": float(source_metrics["dominant_prediction_fraction"]),
            "target_accuracy": float(target_metrics["accuracy"]),
            "target_balanced_accuracy": float(target_metrics["balanced_accuracy"]),
            "target_macro_f1": float(target_metrics["macro_f1"]),
            "target_dominant_prediction_fraction": float(target_metrics["dominant_prediction_fraction"]),
        }
        snapshot = {"epoch": epoch, "source": source_metrics, "target": target_metrics}
        monitor_rows.append(row)
        snapshots.append(snapshot)
        _write_csv(output / "monitor_history.csv", monitor_rows)
        write_json(output / "monitor_snapshots.json", snapshots)
        write_json(output / f"monitor_epoch_{epoch:03d}.json", snapshot)
        _save_monitor_plot(output, condition, fold, epoch, monitor_rows, source_metrics, target_metrics)
        LOGGER.info(
            "Fold %02d %s epoch %03d source acc %.4f target acc %.4f target dominant %.3f",
            fold,
            condition,
            epoch,
            source_metrics["accuracy"],
            target_metrics["accuracy"],
            target_metrics["dominant_prediction_fraction"],
        )
        final_source_metrics = source_metrics
        final_target_metrics = target_metrics
        final_target_y = target_y
        final_target_prediction = target_prediction

    try:
        if bool(settings["monitor_epoch_zero"]):
            monitor(0)
        for epoch in range(1, total_epochs + 1):
            model.train()
            loss_sum = 0.0
            example_count = 0
            gradient_norms: list[float] = []
            for data, mask, labels in source_loader:
                valid_indices = _valid_indices_on_device(mask, device)
                data = data.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(training, device):
                    loss = criterion(_forward(model, data, mask, valid_indices), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("gradient_clip_norm", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach().item()) * labels.shape[0]
                example_count += int(labels.shape[0])
                gradient_norms.append(float(grad_norm.detach().cpu()))
            scheduler.step()
            training_rows.append({
                "epoch": epoch,
                "train_loss": loss_sum / max(example_count, 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "mean_gradient_norm": float(np.mean(gradient_norms)),
                "maximum_gradient_norm": float(np.max(gradient_norms)),
            })
            _write_csv(output / "training_history.csv", training_rows)
            if epoch % monitor_interval == 0 or epoch == total_epochs:
                monitor(epoch)
    finally:
        for loader in (source_loader, source_eval_loader, target_loader):
            _shutdown_persistent_loader(loader)

    assert final_source_metrics is not None and final_target_metrics is not None
    assert final_target_y is not None and final_target_prediction is not None
    checkpoint = output / "fixed_final_epoch_model.pt"
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalization_mean": mean,
        "normalization_std": std,
        "representation": representation,
        "condition": condition,
        "architecture": settings["architecture"],
        "frequency_band_sizes": _feature_dimensions(config)[2],
        "frequency_encoder_hidden": int(settings["frequency_encoder_hidden"]),
        "supervised_components_by_band": dict(zip(
            config.raw["signal"]["bands_hz"],
            _supervised_components_by_band(config),
            strict=True,
        )),
        "training": training,
        "epochs": total_epochs,
        "classes": len(EMOTION_NAMES),
        "input_dim": input_dim,
        "max_length": max_length,
        "seed": SEED,
        "target_monitored_during_training": True,
        "target_used_for_gradients": False,
        "checkpoint_selection": "fixed_final_epoch_only",
    }, checkpoint)
    predictions = [
        {
            "source_index": int(sample.source_index),
            "subject": int(sample.subject),
            "session": int(sample.session),
            "trial": int(sample.trial),
            "target": int(y),
            "prediction": int(prediction),
        }
        for sample, y, prediction in zip(
            target_samples, final_target_y, final_target_prediction, strict=True
        )
    ]
    _write_csv(output / "predictions.csv", predictions)
    result = {
        "schema_version": SCHEMA_VERSION,
        "family": _family(config),
        "status": "complete",
        "dataset": "faced",
        "condition": condition,
        "representation": representation,
        "architecture": (
            "VectorBandHierarchicalChannelTransformer Base"
            if condition in VECTOR_CONDITIONS
            else "HierarchicalChannelBandTransformer Base"
        ),
        "fold": fold,
        "seed": SEED,
        "protocol_hash": protocol_hash,
        "source_subjects": source,
        "target_subjects": target,
        "epochs": total_epochs,
        "monitor_interval": monitor_interval,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "source_only_feature_state": True,
        "source_labels_used_for_supervised_projection": representation == "fisher_rao_supervised_lda",
        "target_monitored_during_training": True,
        "target_used_for_gradients": False,
        "target_used_for_checkpoint_selection": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "class_balanced_loss": bool(training.get("class_balanced_loss", False)),
        "final_source": final_source_metrics,
        "final_target_test": final_target_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
        "diagnostic_smoke": source_override is not None or target_override is not None,
        "evidence_label": "exploratory target-monitored",
    }
    write_json(output / "result.json", result)
    write_json(output / "COMPLETE.json", {
        "protocol_hash": protocol_hash,
        "condition": condition,
        "fold": fold,
        "completed_at": result["completed_at"],
    })
    write_json(output / "protocol_audit.json", {
        "source_reference_and_projection_subjects": source,
        "target_subjects": target,
        "target_monitored_every_epochs": monitor_interval,
        "target_used_for_gradients": False,
        "target_used_for_checkpoint_selection": False,
        "early_stopping": False,
        "checkpoint_selection": "fixed_final_epoch_only",
        "evidence_label": "exploratory target-monitored",
    })
    del source_samples, target_samples, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _declared_tasks(folds: Sequence[int], conditions: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"faced_relative__{condition}__fold-{fold:02d}__seed-{SEED}",
            "fold": fold,
            "condition": condition,
            "status": "pending",
            "attempts": 0,
        }
        for fold in folds
        for condition in conditions
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
    active_conditions = _active_conditions(config)
    selected_folds = list(folds or range(1, FOLDS + 1))
    selected_conditions = list(conditions or active_conditions)
    if any(fold < 1 or fold > FOLDS for fold in selected_folds):
        raise ValueError("Fold filter must be between 1 and 10")
    if set(selected_conditions) - set(active_conditions):
        raise ValueError("Unknown condition filter")
    tasks = _declared_tasks(selected_folds, selected_conditions)
    manifest_path = run_root / "matrix_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {
        "schema_version": SCHEMA_VERSION,
        "family": _family(config),
        "protocol_hash": protocol_hash,
        "tasks": {},
        "created_at": utc_now(),
    }
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("Existing manifest uses another FACED relative protocol")
    for task in tasks:
        manifest["tasks"].setdefault(task["task_id"], task)
    write_json(manifest_path, manifest)

    runnable = []
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
            source, target = official_fold_subjects(int(task["fold"]))
            prepare_fold_features(
                config, run_root, int(task["fold"]), protocol_hash,
                subjects=[*source, *target],
            )
            result = run_task(
                config,
                run_root,
                int(task["fold"]),
                str(task["condition"]),
                protocol_hash,
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
            output = _task_output(run_root, task["condition"], task["fold"])
            write_json(output / "failure.json", {
                "error": task["error"],
                "traceback": traceback.format_exc(),
            })
            LOGGER.exception("FACED relative task failed: %s", task["task_id"])
        finally:
            write_json(manifest_path, manifest)
    return matrix_status(run_root)


def run_smoke(config: ExperimentConfig, run_root: Path, epochs: int = 1) -> dict[str, Any]:
    _, protocol_hash = _require_lock(config, run_root)
    smoke_root = Path(f"{run_root}_smoke")
    source = [12, 13]
    target = [0]
    prepare_fold_features(config, smoke_root, 1, protocol_hash, subjects=[*source, *target])
    results = {}
    for condition in _active_conditions(config):
        results[condition] = run_task(
            config,
            smoke_root,
            1,
            condition,
            protocol_hash,
            epochs=epochs,
            source_override=source,
            target_override=target,
        )
    return results


def matrix_status(run_root: Path) -> dict[str, Any]:
    path = run_root / "matrix_manifest.json"
    if not path.is_file():
        return {"status": "not_started", "run_root": str(run_root.resolve())}
    manifest = read_json(path)
    tasks = list(manifest.get("tasks", {}).values())
    counts = {
        status: sum(task.get("status") == status for task in tasks)
        for status in ("pending", "running", "complete", "failed")
    }
    failed = [
        {"task_id": task["task_id"], "error": task.get("error")}
        for task in tasks
        if task.get("status") == "failed"
    ]
    _write_csv(run_root / "failed_tasks.csv", failed)
    return {
        "status": "complete" if tasks and counts["complete"] == len(tasks) else "in_progress",
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
        target = result["final_target_test"]
        source = result["final_source"]
        rows.append({
            "condition": result["condition"],
            "fold": int(result["fold"]),
            "target_subjects": json.dumps(result["target_subjects"]),
            "source_accuracy": float(source["accuracy"]),
            "target_accuracy": float(target["accuracy"]),
            "target_balanced_accuracy": float(target["balanced_accuracy"]),
            "target_macro_f1": float(target["macro_f1"]),
        })
    condition_names = list(dict.fromkeys(row["condition"] for row in rows))
    expected = len(condition_names) * FOLDS
    if not allow_partial and len(rows) != expected:
        raise RuntimeError(f"Strict summary requires {expected} completed tasks, found {len(rows)}")
    summaries = []
    for condition in condition_names:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        summaries.append({
            "condition": condition,
            "folds": len(subset),
            **{
                f"{metric}_{stat}": float(getattr(np, stat)([row[metric] for row in subset]))
                for metric in (
                    "source_accuracy",
                    "target_accuracy",
                    "target_balanced_accuracy",
                    "target_macro_f1",
                )
                for stat in ("mean", "std")
            },
        })
    _write_csv(run_root / "fold_results.csv", rows)
    _write_csv(run_root / "condition_summary.csv", summaries)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family": (
            read_json(next(iter(sorted((run_root / "faced").glob("*/fold-*/seed-42/result.json")))))
            .get("family", FAMILY)
            if rows else FAMILY
        ),
        "status": "complete" if len(rows) == expected else "partial",
        "completed_tasks": len(rows),
        "expected_tasks": expected,
        "conditions": summaries,
        "evidence_label": "exploratory target-monitored",
        "updated_at": utc_now(),
    }
    write_json(run_root / "summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FACED frequency-point RJSD and supervised Fisher-Rao monitored experiments"
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    def configured(name: str) -> argparse.ArgumentParser:
        child = subparsers.add_parser(name)
        child.add_argument("--config", default="configs/faced/relative_supervised_monitor.yaml")
        child.add_argument("--run-root")
        return child

    validate = configured("validate-data")
    validate.add_argument("--deep", action="store_true")
    configured("lock")
    features = configured("prepare-features")
    features.add_argument("--fold", type=int, action="append")
    smoke = configured("smoke")
    smoke.add_argument("--smoke-epochs", type=int, default=1)
    matrix = configured("matrix")
    matrix.add_argument("--fold", type=int, action="append")
    matrix.add_argument("--condition", choices=tuple(SUPPORTED_CONDITIONS), action="append")
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
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.command == "status":
        payload = matrix_status(Path(args.run_root).expanduser().resolve())
    elif args.command == "summarize":
        payload = summarize(
            Path(args.run_root).expanduser().resolve(), allow_partial=args.allow_partial
        )
    else:
        config = load_config(args.config)
        run_root = Path(args.run_root).expanduser().resolve() if args.run_root else config.run_root
        if args.command == "validate-data":
            payload = validate_sources(config, deep=args.deep)
        elif args.command == "lock":
            payload = lock_experiment(config, run_root)
        elif args.command == "prepare-features":
            _, protocol_hash = _require_lock(config, run_root)
            folds = args.fold or list(range(1, FOLDS + 1))
            payload = {
                f"fold-{fold:02d}": prepare_fold_features(
                    config, run_root, fold, protocol_hash
                )
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
