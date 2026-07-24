from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .faced import EMOTION_NAMES, SUBJECTS, VIDEO_LABELS, VIDEOS, official_fold_subjects
from .models.faced_psd_jsd import (
    NativeBandChannelTemporalTransformer,
    NativeBandFlattenTemporalTransformer,
    PaddedCNNTemporalTransformer,
    parameter_count,
)


BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
BAND_SIZES = (3, 4, 6, 16, 17)
CHANNELS = 30
TIME_STEPS = 30
FREQUENCY_MAX = max(BAND_SIZES)
CLASSES = len(EMOTION_NAMES)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_state(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def frequency_mask() -> np.ndarray:
    mask = np.zeros((len(BAND_SIZES), FREQUENCY_MAX), dtype=np.float32)
    for band, size in enumerate(BAND_SIZES):
        mask[band, :size] = 1.0
    return mask


@dataclass(frozen=True)
class SplitDefinition:
    name: str
    train_subjects: tuple[int, ...]
    development_subjects: tuple[int, ...]
    outer_target_subjects: tuple[int, ...]

    def validate(self) -> None:
        train = set(self.train_subjects)
        development = set(self.development_subjects)
        target = set(self.outer_target_subjects)
        if not train or not development or not target:
            raise ValueError("Every split role must be non-empty")
        if train & development or train & target or development & target:
            raise ValueError(f"Split {self.name} is not subject-disjoint")


def smoke_split(outer_fold: int, seed: int, train_subjects: int, dev_subjects: int) -> SplitDefinition:
    source, target = official_fold_subjects(outer_fold)
    shuffled = np.random.default_rng(seed).permutation(source).tolist()
    definition = SplitDefinition(
        name="smoke",
        train_subjects=tuple(map(int, shuffled[dev_subjects : dev_subjects + train_subjects])),
        development_subjects=tuple(map(int, shuffled[:dev_subjects])),
        outer_target_subjects=tuple(map(int, target)),
    )
    definition.validate()
    return definition


def inner_cv_splits(outer_fold: int, seed: int, folds: int) -> list[SplitDefinition]:
    if folds < 2:
        raise ValueError("inner folds must be at least two")
    source, target = official_fold_subjects(outer_fold)
    shuffled = np.random.default_rng(seed).permutation(source)
    development_folds = np.array_split(shuffled, folds)
    output: list[SplitDefinition] = []
    source_set = set(source)
    for index, development in enumerate(development_folds, 1):
        development_set = set(map(int, development.tolist()))
        definition = SplitDefinition(
            name=f"inner-{index:02d}",
            train_subjects=tuple(sorted(source_set - development_set)),
            development_subjects=tuple(sorted(development_set)),
            outer_target_subjects=tuple(map(int, target)),
        )
        definition.validate()
        output.append(definition)
    validation_union = set().union(*(set(split.development_subjects) for split in output))
    if validation_union != source_set:
        raise AssertionError("Inner validation folds must cover every outer-source subject once")
    return output


class SpectraStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("all_subjects_complete"):
            raise ValueError("Native spectra cache is incomplete")
        if tuple(self.manifest.get("band_sizes", [])) != BAND_SIZES:
            raise ValueError("Native spectra band sizes do not match the experiment")
        if len(self.manifest.get("subjects_complete", [])) != SUBJECTS:
            raise ValueError("Native spectra cache does not contain all FACED subjects")
        self.loaded_subjects: set[int] = set()

    def load(self, subject: int) -> list[np.ndarray]:
        subject = int(subject)
        path = self.root / "subjects" / f"sub{subject:03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            values = [np.asarray(archive[name], dtype=np.float32) for name in BAND_NAMES]
        for name, value, size in zip(BAND_NAMES, values, BAND_SIZES, strict=True):
            expected = (VIDEOS, TIME_STEPS, CHANNELS, size)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"{path.name}/{name}: expected {expected}, got {value.shape}")
            if not np.allclose(value.sum(axis=-1), 1.0, atol=2e-5):
                raise ValueError(f"{path.name}/{name}: probability normalization failed")
        self.loaded_subjects.add(subject)
        return values


def fit_reference(store: SpectraStore, subjects: Sequence[int]) -> tuple[list[np.ndarray], int]:
    sums = [np.zeros((CHANNELS, size), dtype=np.float64) for size in BAND_SIZES]
    windows = 0
    for subject in subjects:
        for band, value in enumerate(store.load(subject)):
            sums[band] += value.sum(axis=(0, 1), dtype=np.float64)
        windows += VIDEOS * TIME_STEPS
    references = []
    for value in sums:
        reference = value / windows
        reference /= reference.sum(axis=-1, keepdims=True)
        references.append(reference.astype(np.float32))
    return references, windows


def _pointwise_sqrt_jsd(p: np.ndarray, q: np.ndarray, epsilon: float) -> tuple[np.ndarray, float]:
    q_expanded = q[None, None, :, :]
    midpoint = 0.5 * (p + q_expanded)
    contribution = 0.5 * (
        p * (np.log(p + epsilon) - np.log(midpoint + epsilon))
        + q_expanded * (np.log(q_expanded + epsilon) - np.log(midpoint + epsilon))
    )
    contribution = np.maximum(contribution, 0.0).astype(np.float32)
    field = np.sqrt(contribution, dtype=np.float32)
    error = float(
        np.max(np.abs(np.square(field).sum(axis=-1) - contribution.sum(axis=-1)))
    )
    return field, error


def transform_subject(
    store: SpectraStore,
    subject: int,
    reference: Sequence[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, float]:
    output = np.zeros(
        (VIDEOS, TIME_STEPS, CHANNELS, len(BAND_SIZES), FREQUENCY_MAX),
        dtype=np.float32,
    )
    maximum_error = 0.0
    for band, (value, q, size) in enumerate(
        zip(store.load(subject), reference, BAND_SIZES, strict=True)
    ):
        field, error = _pointwise_sqrt_jsd(value, q, epsilon)
        output[..., band, :size] = field
        maximum_error = max(maximum_error, error)
    return output, maximum_error


def materialize_split(
    store: SpectraStore,
    subjects: Sequence[int],
    reference: Sequence[np.ndarray],
    epsilon: float,
    storage_dtype: str,
) -> dict[str, np.ndarray | float]:
    dtype = np.float16 if storage_dtype == "float16" else np.float32
    samples = len(subjects) * VIDEOS
    features = np.empty(
        (samples, TIME_STEPS, CHANNELS, len(BAND_SIZES), FREQUENCY_MAX), dtype=dtype
    )
    labels = np.tile(VIDEO_LABELS, len(subjects)).astype(np.int64)
    subject_ids = np.repeat(np.asarray(subjects, dtype=np.int64), VIDEOS)
    maximum_error = 0.0
    for index, subject in enumerate(subjects):
        value, error = transform_subject(store, int(subject), reference, epsilon)
        start = index * VIDEOS
        features[start : start + VIDEOS] = value.astype(dtype)
        maximum_error = max(maximum_error, error)
    return {
        "x": features,
        "y": labels,
        "subjects": subject_ids,
        "maximum_invariant_error": maximum_error,
    }


def fit_standardizer(values: np.ndarray, chunk_trials: int = 32) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(values.shape[2:], dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for start in range(0, len(values), chunk_trials):
        chunk = values[start : start + chunk_trials].astype(np.float32)
        total += chunk.sum(axis=(0, 1), dtype=np.float64)
        square += np.square(chunk).sum(axis=(0, 1), dtype=np.float64)
        count += chunk.shape[0] * chunk.shape[1]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-7] = 1.0
    mask = frequency_mask()[None, :, :]
    mean *= mask
    std = np.where(mask > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


class TrialDataset(Dataset):
    def __init__(self, split: dict[str, Any], mean: np.ndarray, std: np.ndarray) -> None:
        self.x = split["x"]
        self.y = torch.as_tensor(split["y"], dtype=torch.long)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.x[index].astype(np.float32)
        value = (value - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(value)), self.y[index]


def build_model(candidate: dict[str, Any], dropout_override: float | None = None) -> nn.Module:
    dropout = float(candidate["dropout"] if dropout_override is None else dropout_override)
    common = {
        "classes": CLASSES,
        "d_model": int(candidate["d_model"]),
        "heads": int(candidate["heads"]),
        "feedforward": int(candidate["feedforward"]),
        "dropout": dropout,
    }
    model_type = str(candidate["model"])
    if model_type == "cnn_tiny":
        return PaddedCNNTemporalTransformer(
            **common,
            layers=int(candidate["temporal_layers"]),
            cnn_channels=tuple(map(int, candidate.get("cnn_channels", [8, 16, 32]))),
            frequency_mask=torch.from_numpy(frequency_mask()),
        )
    if model_type == "native_band":
        return NativeBandChannelTemporalTransformer(
            **common,
            temporal_layers=int(candidate["temporal_layers"]),
            channel_layers=int(candidate.get("channel_layers", 1)),
            frequency_hidden=int(candidate.get("frequency_hidden", 32)),
        )
    if model_type == "native_band_flatten":
        return NativeBandFlattenTemporalTransformer(
            **common,
            temporal_layers=int(candidate["temporal_layers"]),
            frequency_hidden=int(candidate.get("frequency_hidden", 32)),
            band_dim=int(candidate.get("band_dim", 8)),
        )
    raise ValueError(f"Unknown model type {model_type}")


def _stratified_sanity_indices(labels: np.ndarray, per_class: int) -> np.ndarray:
    selected: list[int] = []
    for label in range(CLASSES):
        indices = np.flatnonzero(labels == label)
        if len(indices) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} sanity samples")
        selected.extend(indices[:per_class].tolist())
    return np.asarray(selected, dtype=np.int64)


def run_sanity_gate(
    candidate: dict[str, Any],
    dataset: TrialDataset,
    labels: np.ndarray,
    device: torch.device,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    indices = _stratified_sanity_indices(labels, int(config["samples_per_class"]))
    values = torch.stack([dataset[int(index)][0] for index in indices]).to(device)
    targets = torch.stack([dataset[int(index)][1] for index in indices]).to(device)
    model = build_model(candidate, dropout_override=0.0).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=0.0
    )
    maximum_steps = int(config["maximum_steps"])
    target_accuracy = float(config["target_accuracy"])
    target_loss = float(config["target_loss"])
    losses: list[float] = []
    final_accuracy = 0.0
    passed = False
    completed_steps = 0
    for step in range(1, maximum_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(values)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        completed_steps = step
        current_loss = float(loss.detach())
        losses.append(current_loss)
        if step == 1 or step % 10 == 0:
            model.eval()
            with torch.no_grad():
                evaluation_logits = model(values)
                final_accuracy = float(
                    (evaluation_logits.argmax(1) == targets).float().mean()
                )
                evaluation_loss = float(
                    nn.functional.cross_entropy(evaluation_logits, targets)
                )
            if final_accuracy >= target_accuracy and evaluation_loss <= target_loss:
                losses[-1] = evaluation_loss
                passed = True
                break
    model.eval()
    with torch.no_grad():
        final_logits = model(values)
        final_accuracy = float((final_logits.argmax(1) == targets).float().mean())
        final_loss = float(nn.functional.cross_entropy(final_logits, targets))
    passed = passed or (final_accuracy >= target_accuracy and final_loss <= target_loss)
    result = {
        "status": "passed" if passed else "failed",
        "samples": len(indices),
        "samples_per_class": int(config["samples_per_class"]),
        "regularization_disabled": True,
        "gradient_clipping_disabled": True,
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "steps": completed_steps,
        "target_accuracy": target_accuracy,
        "target_loss": target_loss,
    }
    del model, optimizer, values, targets, final_logits
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "float32":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    for value, label in loader:
        value = value.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with _autocast(device, precision):
            logits = model(value)
            loss = criterion(logits, label)
        batch = len(label)
        loss_sum += float(loss) * batch
        count += batch
        targets.append(label.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
    y = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    logits_np = np.concatenate(logits_all)
    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    histogram = np.bincount(prediction, minlength=CLASSES)
    return {
        "loss": loss_sum / count,
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "predicted_classes": int(np.count_nonzero(histogram)),
        "prediction_histogram": histogram.tolist(),
        "mean_prediction_entropy": float(entropy.mean()),
        "mean_logit_std": float(logits_np.std(axis=1).mean()),
        "confusion_matrix": confusion_matrix(y, prediction, labels=np.arange(CLASSES)).tolist(),
    }


def _lr_factor(step: int, total_steps: int, warmup_steps: int, minimum_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(step + 1, 1) / warmup_steps
    remaining = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def train_candidate(
    *,
    candidate_name: str,
    candidate: dict[str, Any],
    train_split: dict[str, Any],
    development_split: dict[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
    sanity: dict[str, Any],
    split: SplitDefinition,
    store: SpectraStore,
    output_dir: Path,
    config: dict[str, Any],
    protocol_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    candidate_hash = _json_hash({"candidate": candidate, "training": config})
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("candidate_hash") != candidate_hash:
            raise RuntimeError(f"Refusing to mix candidate protocols in {output_dir}")
        if existing.get("status") == "complete":
            return existing
    output_dir.mkdir(parents=True, exist_ok=True)
    if sanity["status"] != "passed":
        result = {
            "status": "sanity_failed",
            "candidate": candidate_name,
            "candidate_hash": candidate_hash,
            "protocol_hash": protocol_hash,
            "split": split.name,
            "sanity": sanity,
            "target_loaded": False,
        }
        _write_json(result_path, result)
        return result

    seed = int(config["seed"])
    seed_everything(seed)
    train_dataset = TrialDataset(train_split, mean, std)
    development_dataset = TrialDataset(development_split, mean, std)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=int(config["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=int(config["evaluation_batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = build_model(candidate).to(device)
    parameters = parameter_count(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    total_steps = epochs * len(train_loader)
    warmup_steps = round(total_steps * float(config["warmup_fraction"]))
    minimum_ratio = float(config["minimum_learning_rate"]) / float(config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_factor(step, total_steps, warmup_steps, minimum_ratio),
    )
    precision = str(config["precision"])
    use_scaler = device.type == "cuda" and precision == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    clip_norm = float(config["gradient_clip_norm"])
    chance = 1.0 / CLASSES
    gate_config = config["selection_gate"]
    history: list[dict[str, Any]] = []
    best_key = (-1, -math.inf, -math.inf, -math.inf)
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    checkpoint_path = output_dir / "source_selected_model.pt"
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_mode_loss = 0.0
        train_mode_correct = 0
        train_mode_count = 0
        gradient_norms: list[float] = []
        clipped_steps = 0
        for value, label in train_loader:
            value = value.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                logits = model(value)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_norm > 0:
                gradient = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                gradient_value = float(gradient.detach().cpu())
            else:
                squared_norm = torch.zeros((), device=device)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        squared_norm += parameter.grad.detach().float().square().sum()
                gradient_value = float(torch.sqrt(squared_norm).cpu())
            gradient_norms.append(gradient_value)
            clipped_steps += int(clip_norm > 0 and gradient_value > clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_mode_loss += float(loss.detach()) * len(label)
            train_mode_correct += int((logits.argmax(1) == label).sum())
            train_mode_count += len(label)

        train_metrics = evaluate(model, train_eval_loader, device, precision, criterion)
        development_metrics = evaluate(
            model, development_loader, device, precision, criterion
        )
        gate_passed = (
            train_metrics["accuracy"] >= chance + float(gate_config["minimum_train_margin"])
            and development_metrics["balanced_accuracy"]
            >= chance + float(gate_config["minimum_dev_margin"])
            and development_metrics["predicted_classes"]
            >= int(gate_config["minimum_predicted_classes"])
        )
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_mode_loss": train_mode_loss / train_mode_count,
            "train_mode_accuracy": train_mode_correct / train_mode_count,
            "eval_train_loss": train_metrics["loss"],
            "eval_train_accuracy": train_metrics["accuracy"],
            "eval_train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "eval_train_macro_f1": train_metrics["macro_f1"],
            "development_loss": development_metrics["loss"],
            "development_accuracy": development_metrics["accuracy"],
            "development_balanced_accuracy": development_metrics["balanced_accuracy"],
            "development_macro_f1": development_metrics["macro_f1"],
            "development_predicted_classes": development_metrics["predicted_classes"],
            "development_prediction_histogram": json.dumps(
                development_metrics["prediction_histogram"], separators=(",", ":")
            ),
            "development_mean_prediction_entropy": development_metrics[
                "mean_prediction_entropy"
            ],
            "development_mean_logit_std": development_metrics["mean_logit_std"],
            "mean_preclip_gradient_norm": float(np.mean(gradient_norms)),
            "clipped_step_fraction": clipped_steps / len(gradient_norms),
            "selection_gate_passed": bool(gate_passed),
        }
        history.append(row)
        key = (
            int(gate_passed),
            development_metrics["macro_f1"],
            development_metrics["balanced_accuracy"],
            -development_metrics["loss"],
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = {
                "train": train_metrics,
                "development": development_metrics,
                "selection_gate_passed": bool(gate_passed),
            }
            torch.save(
                {
                    "model_state_dict": {
                        name: value.detach().cpu() for name, value in model.state_dict().items()
                    },
                    "feature_mean": mean,
                    "feature_std": std,
                    "candidate": candidate,
                    "candidate_hash": candidate_hash,
                    "protocol_hash": protocol_hash,
                    "best_epoch": best_epoch,
                    "target_loaded_during_selection": False,
                },
                checkpoint_path,
            )
        print(
            f"{split.name}/{candidate_name} epoch {epoch:03d}/{epochs} "
            f"train={train_metrics['accuracy']:.3f} "
            f"dev_bacc={development_metrics['balanced_accuracy']:.3f} "
            f"dev_f1={development_metrics['macro_f1']:.3f} "
            f"classes={development_metrics['predicted_classes']} gate={gate_passed}",
            flush=True,
        )

    if best_metrics is None:
        raise AssertionError("Training completed without a checkpoint")
    with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    target_set = set(split.outer_target_subjects)
    if store.loaded_subjects & target_set:
        raise RuntimeError("Outer-target spectra were loaded during source selection")
    result = {
        "status": "complete",
        "candidate": candidate_name,
        "candidate_hash": candidate_hash,
        "protocol_hash": protocol_hash,
        "split": split.name,
        "parameter_count": parameters,
        "train_trials": len(train_dataset),
        "development_trials": len(development_dataset),
        "best_epoch": best_epoch,
        "best": best_metrics,
        "sanity": sanity,
        "elapsed_seconds": time.perf_counter() - started,
        "loaded_subjects": sorted(store.loaded_subjects),
        "outer_target_subjects": list(split.outer_target_subjects),
        "target_loaded": False,
        "target_used_for_selection": False,
        "completed_at": _now(),
    }
    _write_json(result_path, result)
    del model, optimizer, scheduler, scaler, train_loader, train_eval_loader, development_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def prepare_split(
    split: SplitDefinition,
    spectra_root: Path,
    feature_config: dict[str, Any],
) -> tuple[SpectraStore, dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    store = SpectraStore(spectra_root)
    started = time.perf_counter()
    reference, windows = fit_reference(store, split.train_subjects)
    train_split = materialize_split(
        store,
        split.train_subjects,
        reference,
        float(feature_config["epsilon"]),
        str(feature_config["storage_dtype"]),
    )
    development_split = materialize_split(
        store,
        split.development_subjects,
        reference,
        float(feature_config["epsilon"]),
        str(feature_config["storage_dtype"]),
    )
    mean, std = fit_standardizer(train_split["x"])
    target_set = set(split.outer_target_subjects)
    if store.loaded_subjects & target_set:
        raise RuntimeError("Outer-target spectra were loaded while preparing source features")
    audit = {
        "split": split.name,
        "reference_scope": "inner_train_subjects_only",
        "reference_windows": windows,
        "train_subjects": list(split.train_subjects),
        "development_subjects": list(split.development_subjects),
        "outer_target_subjects": list(split.outer_target_subjects),
        "loaded_subjects": sorted(store.loaded_subjects),
        "target_loaded": False,
        "train_class_counts": np.bincount(train_split["y"], minlength=CLASSES).tolist(),
        "development_class_counts": np.bincount(
            development_split["y"], minlength=CLASSES
        ).tolist(),
        "maximum_jsd_invariant_error": max(
            float(train_split["maximum_invariant_error"]),
            float(development_split["maximum_invariant_error"]),
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return store, train_split, development_split, mean, std, audit


def aggregate_results(results: Sequence[dict[str, Any]], candidates: Iterable[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in candidates:
        rows = [
            result
            for result in results
            if result.get("candidate") == candidate and result.get("status") == "complete"
        ]
        if not rows:
            output[candidate] = {"completed_folds": 0}
            continue
        metrics = [row["best"]["development"] for row in rows]
        output[candidate] = {
            "completed_folds": len(rows),
            "parameter_count": rows[0]["parameter_count"],
            "sanity_pass_rate": float(
                np.mean([row["sanity"]["status"] == "passed" for row in rows])
            ),
            "selection_gate_pass_rate": float(
                np.mean([row["best"]["selection_gate_passed"] for row in rows])
            ),
            "development_accuracy_mean": float(np.mean([m["accuracy"] for m in metrics])),
            "development_accuracy_std": float(np.std([m["accuracy"] for m in metrics])),
            "development_balanced_accuracy_mean": float(
                np.mean([m["balanced_accuracy"] for m in metrics])
            ),
            "development_balanced_accuracy_std": float(
                np.std([m["balanced_accuracy"] for m in metrics])
            ),
            "development_macro_f1_mean": float(np.mean([m["macro_f1"] for m in metrics])),
            "development_macro_f1_std": float(np.std([m["macro_f1"] for m in metrics])),
            "predicted_classes_min": min(m["predicted_classes"] for m in metrics),
            "best_epochs": [row["best_epoch"] for row in rows],
        }
    return output


def _selection_lock(summary: dict[str, Any], expected_folds: int) -> dict[str, Any]:
    eligible = [
        (name, values)
        for name, values in summary.items()
        if values.get("completed_folds") == expected_folds
        and values.get("sanity_pass_rate") == 1.0
        and values.get("selection_gate_pass_rate", 0.0) >= 2.0 / 3.0
    ]
    if not eligible:
        return {
            "status": "no_configuration_passed_noncollapse_gate",
            "selected_candidate": None,
        }
    eligible.sort(
        key=lambda item: (
            item[1]["development_macro_f1_mean"],
            item[1]["development_balanced_accuracy_mean"],
            -item[1]["development_macro_f1_std"],
        ),
        reverse=True,
    )
    return {
        "status": "source_configuration_locked",
        "selected_candidate": eligible[0][0],
        "selection_basis": "3-fold source-only mean Macro-F1, then balanced accuracy and stability",
        "outer_target_evaluated": False,
    }


def load_experiment_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a mapping")
    return payload


def run_stage(config_path: Path, stage: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve()
    config = load_experiment_config(config_path)
    paths = config["paths"]
    spectra_root = (root / paths["spectra_root"]).resolve()
    run_root = (root / paths["run_root"]).resolve()
    protocol = config["protocol"]
    code_hashes = {
        "runner": _file_hash(Path(__file__).resolve()),
        "models": _file_hash(Path(__file__).resolve().parent / "models" / "faced_psd_jsd.py"),
    }
    protocol_payload = {
        "family": "FACED-PSD-JSD-Recommended-Source-Only-v1",
        "config": config,
        "config_sha256": _file_hash(config_path),
        "code_hashes": code_hashes,
        "target_loading_policy": "forbidden_during_sanity_smoke_and_inner_cv",
    }
    protocol_hash = _json_hash(protocol_payload)
    manifest_path = run_root / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_hash") != protocol_hash:
            raise RuntimeError(f"Refusing to mix protocols in {run_root}")
    else:
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "active",
            "protocol_hash": protocol_hash,
            "protocol": protocol_payload,
            "git": _git_state(root),
            "created_at": _now(),
            "target_loaded": False,
        }
        _write_json(manifest_path, manifest)
    if stage == "status":
        status = {"manifest": manifest}
        for name in ("smoke_summary.json", "inner_cv_summary.json", "source_selection_lock.json"):
            path = run_root / name
            status[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return status

    device = resolve_device(str(config["training"]["device"]))
    seed_everything(int(protocol["seed"]))
    print(f"protocol={protocol_hash} device={device} target policy=locked", flush=True)
    candidates = config["candidates"]
    all_results: list[dict[str, Any]] = []

    if stage in {"smoke", "all"}:
        split = smoke_split(
            int(protocol["outer_fold"]),
            int(protocol["seed"]),
            int(protocol["smoke_train_subjects"]),
            int(protocol["smoke_development_subjects"]),
        )
        store, train_split, development_split, mean, std, audit = prepare_split(
            split, spectra_root, config["feature"]
        )
        smoke_root = run_root / "smoke"
        _write_json(smoke_root / "source_isolation_audit.json", audit)
        for index, (name, candidate) in enumerate(candidates.items()):
            sanity = run_sanity_gate(
                candidate,
                TrialDataset(train_split, mean, std),
                train_split["y"],
                device,
                config["sanity"],
                int(protocol["seed"]) + index,
            )
            _write_json(smoke_root / name / "sanity.json", sanity)
            result = train_candidate(
                candidate_name=name,
                candidate=candidate,
                train_split=train_split,
                development_split=development_split,
                mean=mean,
                std=std,
                sanity=sanity,
                split=split,
                store=store,
                output_dir=smoke_root / name,
                config={**config["training"], "epochs": int(config["training"]["smoke_epochs"])},
                protocol_hash=protocol_hash,
                device=device,
            )
            all_results.append(result)
        smoke_summary = aggregate_results(all_results, candidates)
        _write_json(run_root / "smoke_summary.json", smoke_summary)
        del train_split, development_split, mean, std, store
        gc.collect()

    if stage in {"inner-cv", "all"}:
        all_results = []
        splits = inner_cv_splits(
            int(protocol["outer_fold"]), int(protocol["seed"]), int(protocol["inner_folds"])
        )
        for fold_index, split in enumerate(splits, 1):
            print(f"preparing {split.name}", flush=True)
            store, train_split, development_split, mean, std, audit = prepare_split(
                split, spectra_root, config["feature"]
            )
            fold_root = run_root / "inner_cv" / split.name
            _write_json(fold_root / "source_isolation_audit.json", audit)
            for candidate_index, (name, candidate) in enumerate(candidates.items()):
                sanity = run_sanity_gate(
                    candidate,
                    TrialDataset(train_split, mean, std),
                    train_split["y"],
                    device,
                    config["sanity"],
                    int(protocol["seed"]) + 100 * fold_index + candidate_index,
                )
                _write_json(fold_root / name / "sanity.json", sanity)
                result = train_candidate(
                    candidate_name=name,
                    candidate=candidate,
                    train_split=train_split,
                    development_split=development_split,
                    mean=mean,
                    std=std,
                    sanity=sanity,
                    split=split,
                    store=store,
                    output_dir=fold_root / name,
                    config=config["training"],
                    protocol_hash=protocol_hash,
                    device=device,
                )
                all_results.append(result)
            del train_split, development_split, mean, std, store
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        summary = aggregate_results(all_results, candidates)
        _write_json(run_root / "inner_cv_summary.json", summary)
        selection = _selection_lock(summary, len(splits))
        selection.update(
            {
                "protocol_hash": protocol_hash,
                "outer_fold": int(protocol["outer_fold"]),
                "outer_target_subjects": list(
                    official_fold_subjects(int(protocol["outer_fold"]))[1]
                ),
                "outer_target_evaluated": False,
                "locked_at": _now(),
            }
        )
        _write_json(run_root / "source_selection_lock.json", selection)
        manifest["status"] = "source_selection_complete"
        manifest["target_loaded"] = False
        manifest["updated_at"] = _now()
        _write_json(manifest_path, manifest)
        return {"summary": summary, "selection": selection}

    return {"smoke": json.loads((run_root / "smoke_summary.json").read_text(encoding="utf-8"))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/faced/psd_jsd_recommended.yaml"),
    )
    parser.add_argument("--stage", choices=("smoke", "inner-cv", "all", "status"), default="status")
    arguments = parser.parse_args(argv)
    run_stage(arguments.config, arguments.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
