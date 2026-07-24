from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .faced import EMOTION_NAMES, SUBJECTS, VIDEO_LABELS, VIDEOS, official_fold_subjects
from .faced_adversarial_runner import (
    SubjectEmotionBatchSampler,
    bootstrap_subject_metrics,
    independent_probe,
    protocol_videos,
    stimulus_split,
)
from .features.str_jsd import CONDITIONS, build_condition_features, reference_quality, response_starts
from .models.str_jsd import STRJSDHCBT
from .training.metrics import classification_metrics
from .training.runtime import seed_everything, select_device


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")


@dataclass(frozen=True)
class STRJSDConfig:
    path: Path
    base_run_root: Path
    run_root: Path
    fold: int
    seed: int
    development_subjects: tuple[int, ...]
    protocols: tuple[str, ...]
    temporal_settings: dict[str, dict[str, Any]]
    model: dict[str, Any]
    training: dict[str, Any]
    evidence_label: str


@dataclass(frozen=True)
class Example:
    x: np.ndarray
    label: int
    raw_subject: int
    local_subject: int
    video: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> STRJSDConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parents[2]
    protocol = raw["protocol"]
    temporal = {str(key): dict(value) for key, value in protocol["temporal_settings"].items()}
    config = STRJSDConfig(
        path=config_path,
        base_run_root=_resolve(repo_root, raw["paths"]["base_run_root"]),
        run_root=_resolve(repo_root, raw["paths"]["run_root"]),
        fold=int(protocol["fold"]),
        seed=int(protocol["seed"]),
        development_subjects=tuple(map(int, protocol["development_subjects"])),
        protocols=tuple(map(str, protocol["evaluation_protocols"])),
        temporal_settings=temporal,
        model=dict(raw["model"]),
        training=dict(raw["training"]),
        evidence_label=str(raw["experiment"]["evidence_label"]),
    )
    source, target = official_fold_subjects(config.fold)
    if not set(config.development_subjects) <= set(source) or set(config.development_subjects) & set(target):
        raise ValueError("Development subjects must be source-only and target-disjoint")
    allowed_protocols = {"conventional_subject_holdout", "subject_and_stimulus_holdout"}
    if not set(config.protocols) <= allowed_protocols:
        raise ValueError("Unknown STR-JSD evaluation protocol")
    for name, setting in config.temporal_settings.items():
        response_starts(int(setting["response_average_windows"]))
        conditions = tuple(map(str, setting["conditions"]))
        if len(set(conditions)) != len(conditions) or not set(conditions) <= set(CONDITIONS):
            raise ValueError(f"Invalid conditions in temporal setting {name}")
        setting["conditions"] = conditions
    return config


def _spectra_root(config: STRJSDConfig) -> Path:
    candidates = []
    for manifest in sorted((config.base_run_root / "cache" / "native_spectra").glob("*/manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", [])) == SUBJECTS:
            if tuple(payload.get("band_names", [])) == BAND_NAMES:
                candidates.append(manifest.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one complete native FACED spectra cache, found {candidates}")
    return candidates[0]


def spectra_manifest(config: STRJSDConfig) -> dict[str, Any]:
    return json.loads((_spectra_root(config) / "manifest.json").read_text(encoding="utf-8"))


def _load_subject(config: STRJSDConfig, subject: int) -> tuple[np.ndarray, list[np.ndarray]]:
    path = _spectra_root(config) / "subjects" / f"sub{int(subject):03d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        de = np.asarray(archive["de"], dtype=np.float32)
        spectra = [np.asarray(archive[name], dtype=np.float32) for name in BAND_NAMES]
    return de, spectra


def fit_gate_scales(
    config: STRJSDConfig,
    subjects: Sequence[int],
    videos: Sequence[int],
) -> np.ndarray:
    """Fit one robust instability scale per band using source-fit trials only."""
    selected = np.asarray(videos, dtype=np.int64)
    values: list[list[np.ndarray]] = [[] for _ in BAND_NAMES]
    for subject in map(int, subjects):
        de, spectra = _load_subject(config, subject)
        quality = reference_quality(de, spectra)[selected]
        for band in range(len(BAND_NAMES)):
            values[band].append(quality[..., band].reshape(-1))
    scales = np.asarray([
        np.median(np.concatenate(band_values)) for band_values in values
    ], dtype=np.float32)
    scales[scales < 1e-6] = 1.0
    return scales


def load_examples(
    config: STRJSDConfig,
    subjects: Sequence[int],
    videos: Sequence[int],
    subject_map: dict[int, int] | None,
    condition: str,
    response_average_windows: int,
    gate_scales: Sequence[float] | None,
) -> list[Example]:
    selected_videos = list(map(int, videos))
    output: list[Example] = []
    for subject in map(int, subjects):
        de, spectra = _load_subject(config, subject)
        features = build_condition_features(
            de, spectra, condition, response_average_windows, gate_scales=gate_scales
        )
        for video in selected_videos:
            output.append(Example(
                x=np.ascontiguousarray(features[video], dtype=np.float32),
                label=int(VIDEO_LABELS[video]),
                raw_subject=subject,
                local_subject=int(subject_map[subject]) if subject_map is not None else -1,
                video=video,
            ))
    return output


def fit_normalizer(examples: Sequence[Example]) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("Cannot fit normalizer without source-fit examples")
    dimension = examples[0].x.shape[-1]
    total = np.zeros(dimension, dtype=np.float64)
    square = np.zeros(dimension, dtype=np.float64)
    count = 0
    for example in examples:
        value = example.x.astype(np.float64)
        total += value.sum(axis=0)
        square += np.square(value).sum(axis=0)
        count += value.shape[0]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class STRJSDDataset(Dataset):
    def __init__(self, examples: Sequence[Example], mean: np.ndarray, std: np.ndarray) -> None:
        self.data = torch.from_numpy(np.stack([
            np.ascontiguousarray((example.x - mean) / std, dtype=np.float32)
            for example in examples
        ]))
        self.labels = torch.as_tensor([example.label for example in examples], dtype=torch.long)
        self.local_subjects = torch.as_tensor([example.local_subject for example in examples], dtype=torch.long)
        self.raw_subjects = np.asarray([example.raw_subject for example in examples], dtype=np.int64)
        self.videos = np.asarray([example.video for example in examples], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index], self.local_subjects[index]


def _autocast(config: STRJSDConfig, device: torch.device):
    precision = str(config.training.get("precision", "float32"))
    if device.type != "cuda" or precision not in {"float16", "bfloat16"}:
        return nullcontext()
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _valid(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    return mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).to(device, non_blocking=True)


def _loader(dataset: STRJSDDataset, batch_size: int = 128) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(
    model: STRJSDHCBT,
    dataset: STRJSDDataset,
    device: torch.device,
    config: STRJSDConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for data, labels, _ in _loader(dataset, int(config.training.get("evaluation_batch_size", 128))):
        mask = torch.ones(data.shape[:2], dtype=torch.bool)
        valid = _valid(mask, device)
        with _autocast(config, device):
            logits = model(data.to(device, non_blocking=True), mask.to(device, non_blocking=True), valid)
        targets.append(labels.numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    return classification_metrics(y, prediction, len(EMOTION_NAMES)), y, prediction


@torch.no_grad()
def extract_embeddings(
    model: STRJSDHCBT,
    dataset: STRJSDDataset,
    device: torch.device,
    config: STRJSDConfig,
) -> np.ndarray:
    model.eval()
    values = []
    for data, _, _ in _loader(dataset, int(config.training.get("evaluation_batch_size", 128))):
        mask = torch.ones(data.shape[:2], dtype=torch.bool)
        valid = _valid(mask, device)
        with _autocast(config, device):
            embedding = model.encode(data.to(device, non_blocking=True), mask.to(device, non_blocking=True), valid)
        values.append(embedding.float().cpu().numpy())
    return np.concatenate(values)


def source_only_probes(
    model: STRJSDHCBT,
    config: STRJSDConfig,
    mean: np.ndarray,
    std: np.ndarray,
    fit_subjects: Sequence[int],
    dev_subjects: Sequence[int],
    condition: str,
    response_average_windows: int,
    gate_scales: Sequence[float] | None,
    device: torch.device,
) -> dict[str, Any]:
    source_map = {subject: index for index, subject in enumerate(fit_subjects)}
    fit_examples = load_examples(
        config, fit_subjects, np.arange(VIDEOS), source_map, condition,
        response_average_windows, gate_scales,
    )
    dev_examples = load_examples(
        config, dev_subjects, np.arange(VIDEOS), None, condition,
        response_average_windows, gate_scales,
    )
    fit_dataset = STRJSDDataset(fit_examples, mean, std)
    dev_dataset = STRJSDDataset(dev_examples, mean, std)
    del fit_examples, dev_examples
    fit_z = extract_embeddings(model, fit_dataset, device, config)
    dev_z = extract_embeddings(model, dev_dataset, device, config)
    probe_test_videos = stimulus_split()["test"]
    subject_train = ~np.isin(dev_dataset.videos, probe_test_videos)
    subject_test = np.isin(dev_dataset.videos, probe_test_videos)
    c = float(config.training.get("probe_c", 1.0))
    result = {
        "subject_id_probe": independent_probe(
            dev_z[subject_train], dev_dataset.raw_subjects[subject_train],
            dev_z[subject_test], dev_dataset.raw_subjects[subject_test], c,
        ),
        "video_id_probe": independent_probe(fit_z, fit_dataset.videos, dev_z, dev_dataset.videos, c),
        "subject_probe_train_samples": int(subject_train.sum()),
        "subject_probe_test_samples": int(subject_test.sum()),
        "video_probe_train_samples": int(len(fit_dataset)),
        "video_probe_test_samples": int(len(dev_dataset)),
        "subject_random_chance": 1.0 / len(dev_subjects),
        "video_random_chance": 1.0 / VIDEOS,
    }
    del fit_dataset, dev_dataset, fit_z, dev_z
    gc.collect()
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protocol_hash(config: STRJSDConfig) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fold": config.fold,
        "seed": config.seed,
        "development_subjects": config.development_subjects,
        "protocols": config.protocols,
        "temporal_settings": config.temporal_settings,
        "model": config.model,
        "training": config.training,
        "stimulus_split": {key: value.tolist() for key, value in stimulus_split().items()},
        "target_policy": "load only after source-development checkpoint lock",
        "reference_name": "Early-State Temporal Reference",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _task_root(config: STRJSDConfig, protocol: str, temporal: str, condition: str, root: Path | None = None) -> Path:
    return (root or config.run_root) / protocol / temporal / condition / f"fold-{config.fold:02d}" / f"seed-{config.seed}"


def _build_model(config: STRJSDConfig, band_sizes: Sequence[int], sequence_length: int) -> STRJSDHCBT:
    return STRJSDHCBT(
        channels=int(config.model["channels"]),
        band_sizes=[3 + int(size) for size in band_sizes],
        classes=int(config.model["classes"]),
        max_length=sequence_length,
        d_model=int(config.model["d_model"]),
        heads=int(config.model["heads"]),
        layers=int(config.model["layers"]),
        feedforward=int(config.model["feedforward"]),
        dropout=float(config.model["dropout"]),
    )


def train_task(
    config: STRJSDConfig,
    protocol: str,
    temporal: str,
    condition: str,
    *,
    run_root: Path | None = None,
    fit_subjects_override: Sequence[int] | None = None,
    dev_subjects_override: Sequence[int] | None = None,
    target_subjects_override: Sequence[int] | None = None,
    epochs_override: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = _task_root(config, protocol, temporal, condition, run_root)
    result_path = output / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    setting = config.temporal_settings[temporal]
    response_average_windows = int(setting["response_average_windows"])
    sequence_length = len(response_starts(response_average_windows))
    source, target = official_fold_subjects(config.fold)
    dev_subjects = list(dev_subjects_override or config.development_subjects)
    fit_subjects = list(fit_subjects_override or [s for s in source if s not in set(dev_subjects)])
    target_subjects = list(target_subjects_override or target)
    train_videos, dev_videos, target_videos = protocol_videos(protocol)
    subject_map = {subject: index for index, subject in enumerate(fit_subjects)}
    epochs = int(epochs_override or config.training["epochs"])
    seed_everything(config.seed, bool(config.training.get("deterministic", True)))
    device = select_device(str(config.training.get("device", "auto")))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    LOGGER.info("%s %s %s: source-only feature fitting/loading", protocol, temporal, condition)
    gate_scales = (
        fit_gate_scales(config, fit_subjects, train_videos)
        if condition == "C6_c4_reference_quality_gate" else None
    )
    train_examples = load_examples(
        config, fit_subjects, train_videos, subject_map, condition,
        response_average_windows, gate_scales,
    )
    dev_examples = load_examples(
        config, dev_subjects, dev_videos, None, condition,
        response_average_windows, gate_scales,
    )
    mean, std = fit_normalizer(train_examples)
    train_dataset = STRJSDDataset(train_examples, mean, std)
    dev_dataset = STRJSDDataset(dev_examples, mean, std)
    del train_examples, dev_examples
    gc.collect()

    sampler = SubjectEmotionBatchSampler(
        train_dataset,
        int(config.training["batch_subjects"]),
        int(config.training["batch_emotions"]),
        config.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    manifest = spectra_manifest(config)
    band_sizes = list(map(int, manifest["band_sizes"]))
    model = _build_model(config, band_sizes, sequence_length).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.training["learning_rate"]),
        weight_decay=float(config.training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(config.training["minimum_learning_rate"]),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.training.get("label_smoothing", 0.0)))
    precision = str(config.training.get("precision", "float32"))
    use_scaler = device.type == "cuda" and precision == "float16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    checkpoint = output / "source_selected_model.pt"
    history: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0
        gradients = []
        for data, labels, _ in train_loader:
            mask = torch.ones(data.shape[:2], dtype=torch.bool)
            valid = _valid(mask, device)
            data = data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                logits = model(data, mask, valid)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = nn.utils.clip_grad_norm_(model.parameters(), float(config.training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            batch = int(labels.shape[0])
            total_loss += float(loss.detach()) * batch
            correct += int((logits.argmax(1) == labels).sum())
            count += batch
            gradients.append(float(gradient.detach().cpu()))
        scheduler.step()
        dev_metrics, _, _ = evaluate(model, dev_dataset, device, config)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": total_loss / count,
            "train_accuracy": correct / count,
            "source_dev_accuracy": dev_metrics["accuracy"],
            "source_dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
            "source_dev_macro_f1": dev_metrics["macro_f1"],
            "mean_gradient_norm": float(np.mean(gradients)),
        }
        history.append(row)
        _write_csv(output / "training_history.csv", history)
        key = (float(dev_metrics["macro_f1"]), float(dev_metrics["balanced_accuracy"]), -float(epoch))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save({
                "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "normalization_mean": mean,
                "normalization_std": std,
                "gate_scales": gate_scales,
                "condition": condition,
                "temporal": temporal,
                "protocol": protocol,
                "best_epoch": best_epoch,
                "fit_subjects": fit_subjects,
                "development_subjects": dev_subjects,
                "target_loaded_during_training": False,
            }, checkpoint)
        LOGGER.info(
            "%s %s %s epoch %02d dev BACC %.4f F1 %.4f",
            protocol, temporal, condition, epoch,
            dev_metrics["balanced_accuracy"], dev_metrics["macro_f1"],
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    selected_dev_metrics, _, _ = evaluate(model, dev_dataset, device, config)
    # Probes are source-only and run after checkpoint lock; their labels never
    # influence training, feature fitting, or checkpoint selection.
    probes = source_only_probes(
        model, config, mean, std, fit_subjects, dev_subjects, condition,
        response_average_windows, gate_scales, device,
    )
    # Target subjects are deliberately loaded only below this boundary.
    target_examples = load_examples(
        config, target_subjects, target_videos, None, condition,
        response_average_windows, gate_scales,
    )
    target_dataset = STRJSDDataset(target_examples, mean, std)
    target_metrics, target_y, target_prediction = evaluate(model, target_dataset, device, config)
    bootstrap = bootstrap_subject_metrics(
        target_y, target_prediction, target_dataset.raw_subjects,
        int(config.training["bootstrap_repeats"]), config.seed,
    )
    prediction_rows = [
        {
            "subject": int(example.raw_subject), "video": int(example.video),
            "emotion": EMOTION_NAMES[int(example.label)], "target": int(y),
            "prediction": int(prediction),
        }
        for example, y, prediction in zip(target_examples, target_y, target_prediction, strict=True)
    ]
    _write_csv(output / "target_predictions.csv", prediction_rows)
    sample_de, sample_spectra = _load_subject(config, fit_subjects[0])
    _, feature_audit = build_condition_features(
        sample_de, sample_spectra, condition, response_average_windows,
        gate_scales=gate_scales, return_audit=True,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "family": "FACED-STR-JSD-Early-State-Temporal-Reference-Light-v1",
        "reference_name": "Early-State Temporal Reference",
        "reference_windows_zero_based": [0, 1, 2, 3, 4],
        "reference_is_fixation_or_rest": False,
        "condition": condition,
        "temporal_setting": temporal,
        "response_average_windows": response_average_windows,
        "sequence_length": sequence_length,
        "evaluation_protocol": protocol,
        "fold": config.fold,
        "seed": config.seed,
        "protocol_hash": protocol_hash(config),
        "fit_subjects": fit_subjects,
        "development_subjects": dev_subjects,
        "target_subjects": target_subjects,
        "train_videos_zero_based": train_videos.tolist(),
        "development_videos_zero_based": dev_videos.tolist(),
        "target_videos_zero_based": target_videos.tolist(),
        "native_band_names": list(BAND_NAMES),
        "native_band_sizes": band_sizes,
        "capacity_matched_band_sizes": [3 + size for size in band_sizes],
        "energy_reconstruction": "native normalized PSD shape multiplied by exp(2*cached_DE); common factor cancels",
        "c2_unsigned_definition": "sqrt(pointwise JSD contribution)",
        "c6_gate": "exp(-u/source_fit_band_median_u), gamma=1",
        "source_fit_gate_scales": gate_scales.tolist() if gate_scales is not None else None,
        "feature_audit": feature_audit,
        "epochs_maximum": epochs,
        "source_selected_epoch": best_epoch,
        "checkpoint_selection": "maximum source-development Macro-F1; balanced accuracy then earlier epoch break ties",
        "target_loaded_during_training": False,
        "target_used_for_checkpoint_or_hyperparameter_selection": False,
        "target_used_for_gradients": False,
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "source_fit_samples": len(train_dataset),
        "source_development_samples": len(dev_dataset),
        "target_test_samples": len(target_dataset),
        "source_development": selected_dev_metrics,
        "selected_epoch_training": history[best_epoch - 1],
        "target_test": target_metrics,
        "target_subject_bootstrap": bootstrap,
        "independent_source_only_probes": probes,
        "evidence_label": config.evidence_label,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
        "diagnostic_smoke": fit_subjects_override is not None,
    }
    _write_json(result_path, result)
    _write_json(output / "COMPLETE.json", {
        "protocol_hash": result["protocol_hash"], "condition": condition,
        "temporal": temporal, "protocol": protocol, "completed_at": result["completed_at"],
    })
    del model, train_dataset, dev_dataset, target_dataset, target_examples
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def task_specs(config: STRJSDConfig):
    for protocol in config.protocols:
        for temporal, setting in config.temporal_settings.items():
            for condition in setting["conditions"]:
                yield protocol, temporal, condition


def paired_target_bootstrap(config: STRJSDConfig) -> dict[str, Any]:
    repeats = int(config.training["bootstrap_repeats"])
    output: dict[str, Any] = {}
    for protocol in config.protocols:
        output[protocol] = {}
        for temporal, setting in config.temporal_settings.items():
            conditions = list(setting["conditions"])
            if "C0_absolute_de" not in conditions:
                continue
            records: dict[str, list[dict[str, int]]] = {}
            for condition in conditions:
                path = _task_root(config, protocol, temporal, condition) / "target_predictions.csv"
                with path.open("r", encoding="utf-8", newline="") as stream:
                    rows = [{key: int(row[key]) for key in ("subject", "video", "target", "prediction")} for row in csv.DictReader(stream)]
                rows.sort(key=lambda row: (row["subject"], row["video"]))
                records[condition] = rows
            base = records["C0_absolute_de"]
            keys = [(row["subject"], row["video"], row["target"]) for row in base]
            for condition, rows in records.items():
                if [(row["subject"], row["video"], row["target"]) for row in rows] != keys:
                    raise ValueError(f"Unpaired target records for {protocol}/{temporal}/{condition}")
            subjects = np.asarray([row["subject"] for row in base], dtype=np.int64)
            y = np.asarray([row["target"] for row in base], dtype=np.int64)
            base_prediction = np.asarray([row["prediction"] for row in base], dtype=np.int64)
            unique = np.unique(subjects)
            comparisons = {}
            for offset, condition in enumerate(conditions[1:]):
                candidate = np.asarray([row["prediction"] for row in records[condition]], dtype=np.int64)
                rng = np.random.default_rng(config.seed + offset)
                acc_differences = []
                f1_differences = []
                for _ in range(repeats):
                    sampled = rng.choice(unique, size=len(unique), replace=True)
                    indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
                    acc_differences.append(float(accuracy_score(y[indices], candidate[indices]) - accuracy_score(y[indices], base_prediction[indices])))
                    f1_differences.append(float(f1_score(y[indices], candidate[indices], average="macro", zero_division=0) - f1_score(y[indices], base_prediction[indices], average="macro", zero_division=0)))
                comparisons[condition] = {
                    "accuracy_difference": float(accuracy_score(y, candidate) - accuracy_score(y, base_prediction)),
                    "accuracy_difference_ci95": np.quantile(acc_differences, [0.025, 0.975]).tolist(),
                    "accuracy_probability_greater_than_c0": float(np.mean(np.asarray(acc_differences) > 0)),
                    "macro_f1_difference": float(f1_score(y, candidate, average="macro", zero_division=0) - f1_score(y, base_prediction, average="macro", zero_division=0)),
                    "macro_f1_difference_ci95": np.quantile(f1_differences, [0.025, 0.975]).tolist(),
                    "macro_f1_probability_greater_than_c0": float(np.mean(np.asarray(f1_differences) > 0)),
                }
            output[protocol][temporal] = comparisons
    return output


def run_matrix(config: STRJSDConfig, *, force: bool = False) -> dict[str, Any]:
    config.run_root.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(config.run_root / "experiment.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    rows = []
    try:
        for protocol, temporal, condition in task_specs(config):
            result = train_task(config, protocol, temporal, condition, force=force)
            rows.append({
                "protocol": protocol, "temporal": temporal, "condition": condition,
                "best_epoch": result["source_selected_epoch"], "parameters": result["parameter_count"],
                "dev_balanced_accuracy": result["source_development"]["balanced_accuracy"],
                "dev_macro_f1": result["source_development"]["macro_f1"],
                "target_accuracy": result["target_test"]["accuracy"],
                "target_balanced_accuracy": result["target_test"]["balanced_accuracy"],
                "target_macro_f1": result["target_test"]["macro_f1"],
                "subject_probe": result["independent_source_only_probes"]["subject_id_probe"],
                "video_probe": result["independent_source_only_probes"]["video_id_probe"],
                "elapsed_seconds": result["elapsed_seconds"],
            })
            _write_csv(config.run_root / "matrix_results.csv", rows)
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
    summary = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "family": "FACED-STR-JSD-Early-State-Temporal-Reference-Light-v1",
        "protocol_hash": protocol_hash(config), "tasks": len(rows),
        "target_used_for_selection": False, "rows": rows,
        "paired_target_bootstrap_vs_c0": paired_target_bootstrap(config),
        "completed_at": utc_now(),
    }
    _write_json(config.run_root / "summary.json", summary)
    return summary


def run_smoke(config: STRJSDConfig, *, force: bool = False) -> dict[str, Any]:
    source, target = official_fold_subjects(config.fold)
    fit = [subject for subject in source if subject not in set(config.development_subjects)][:8]
    dev = list(config.development_subjects[:4])
    root = config.run_root.parent / f"{config.run_root.name}_smoke"
    rows = []
    for condition in ("C0_absolute_de", "C4_absolute_de_delta_de_signed_pointwise_jsd", "C6_c4_reference_quality_gate"):
        rows.append(train_task(
            config, "conventional_subject_holdout", "five_second_average", condition,
            run_root=root, fit_subjects_override=fit, dev_subjects_override=dev,
            target_subjects_override=target[:1], epochs_override=1, force=force,
        ))
    summary = {"status": "complete", "tasks": len(rows), "run_root": str(root), "results": rows}
    _write_json(root / "summary.json", summary)
    return summary


def status(config: STRJSDConfig) -> dict[str, Any]:
    tasks = []
    for protocol, temporal, condition in task_specs(config):
        complete = (_task_root(config, protocol, temporal, condition) / "result.json").is_file()
        tasks.append({"protocol": protocol, "temporal": temporal, "condition": condition, "complete": complete})
    return {
        "run_root": str(config.run_root), "complete": all(task["complete"] for task in tasks),
        "completed_tasks": sum(task["complete"] for task in tasks), "total_tasks": len(tasks), "tasks": tasks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FACED STR-JSD Early-State Temporal Reference light ablation")
    parser.add_argument("command", choices=("run", "smoke", "status"))
    parser.add_argument("--config", default="configs/faced/str_jsd_fold1_light.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        result = run_matrix(config, force=args.force)
    elif args.command == "smoke":
        result = run_smoke(config, force=args.force)
    else:
        result = status(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
