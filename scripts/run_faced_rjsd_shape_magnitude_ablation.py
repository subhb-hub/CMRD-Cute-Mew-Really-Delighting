from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmrd.faced import (
    EEG_CHANNEL_NAMES,
    EMOTION_NAMES,
    SUBJECTS,
    VIDEO_LABELS,
    VIDEOS,
    load_processed_subject,
    official_fold_subjects,
)
from scripts.run_faced_axial_no_cls_whitebox import PlainTransformerBlock


BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
BAND_LIMITS = ((1, 4), (4, 8), (8, 14), (14, 30), (30, 47))
BAND_SIZES = tuple(high - low for low, high in BAND_LIMITS)
TIME_STEPS = 30
CHANNELS = len(EEG_CHANNEL_NAMES)
CLASSES = len(EMOTION_NAMES)
EPSILON = 1e-12


@dataclass(frozen=True)
class Variant:
    name: str
    spectrum: str
    normalized_shape: bool
    signed_shape: bool
    magnitude_bypass: bool


VARIANTS = {
    "A": Variant("A_1s_unsigned_raw", "1s", False, False, False),
    "B": Variant("B_4s_welch_unsigned_raw", "4s", False, False, False),
    "C": Variant("C_1s_normalized_shape_magnitude", "1s", True, False, True),
    "D": Variant("D_4s_welch_normalized_shape_magnitude", "4s", True, False, True),
    "E": Variant("E_4s_welch_signed_shape_magnitude", "4s", True, True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACED RJSD stable-spectrum shape/magnitude ablation")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    parser.add_argument("--folds", nargs="+", type=int, default=(1,))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers-per-axis", type=int, default=1)
    parser.add_argument("--ffn-ratio", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--linear-magnitude-diagnostic", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--processed-dir", type=Path, default=ROOT.parent / "Dataset" / "Ori" / "FACED" / "Processed_data")
    parser.add_argument("--four-second-cache", type=Path, default=Path("runs/faced_4s_welch_probability_cache"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/faced_rjsd_shape_magnitude_ablation"))
    args = parser.parse_args()
    if any(fold < 1 or fold > 10 for fold in args.folds):
        parser.error("--folds must be in 1..10")
    if args.epochs <= 0 or args.eval_every <= 0 or args.epochs % args.eval_every:
        parser.error("--epochs must be positive and divisible by --eval-every")
    if args.d_model % args.heads:
        parser.error("--d-model must be divisible by --heads")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def find_one_second_cache() -> Path:
    base = ROOT / "runs" / "faced_native_compact_base_seed42" / "cache" / "native_spectra"
    candidates = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("all_subjects_complete") and tuple(payload.get("band_sizes", ())) == BAND_SIZES:
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete 1-second cache, found {candidates}")
    return candidates[0]


def extract_four_second_probabilities(signal: np.ndarray) -> list[np.ndarray]:
    value = np.asarray(signal[:CHANNELS], dtype=np.float32)
    if value.shape != (CHANNELS, 7500):
        raise ValueError(f"Expected [30,7500], got {value.shape}")
    pad = 375
    padded = np.pad(value, ((0, 0), (pad, pad)), mode="reflect")
    framed = np.lib.stride_tricks.sliding_window_view(padded, 1000, axis=-1)
    framed = np.moveaxis(framed[:, np.arange(TIME_STEPS) * 250, :], 1, 0)
    frequencies, psd = welch(
        framed,
        fs=250.0,
        window="hann",
        nperseg=500,
        noverlap=250,
        nfft=500,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    if psd.shape[:2] != (TIME_STEPS, CHANNELS):
        raise RuntimeError(f"Unexpected Welch shape: {psd.shape}")
    df = float(frequencies[1] - frequencies[0])
    if not np.isclose(df, 0.5):
        raise RuntimeError(f"Expected 0.5 Hz resolution, got {df}")
    one_hz = []
    for left in range(1, 47):
        selected = (frequencies >= left) & (frequencies < left + 1)
        if int(selected.sum()) != 2:
            raise RuntimeError(f"1 Hz bin [{left},{left + 1}) has {selected.sum()} points")
        one_hz.append(psd[..., selected].sum(axis=-1, dtype=np.float32) * df)
    integrated = np.stack(one_hz, axis=-1).astype(np.float32)
    output = []
    offset = 0
    for name, size in zip(BAND_NAMES, BAND_SIZES, strict=True):
        power = integrated[..., offset : offset + size]
        denominator = power.sum(axis=-1, keepdims=True, dtype=np.float32)
        if np.any(denominator <= EPSILON):
            raise FloatingPointError(f"Invalid power in {name}")
        output.append(np.ascontiguousarray(power / denominator, dtype=np.float32))
        offset += size
    return output


def build_four_second_cache(cache_root: Path, processed_dir: Path) -> None:
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", ())) == SUBJECTS:
            return
    if not processed_dir.is_dir():
        raise FileNotFoundError(processed_dir)
    subjects_root = cache_root / "subjects"
    subjects_root.mkdir(parents=True, exist_ok=True)
    complete = []
    started = time.perf_counter()
    for subject in range(SUBJECTS):
        path = subjects_root / f"sub{subject:03d}.npz"
        valid = False
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as archive:
                    valid = all(
                        archive[name].shape == (VIDEOS, TIME_STEPS, CHANNELS, size)
                        for name, size in zip(BAND_NAMES, BAND_SIZES, strict=True)
                    )
            except (OSError, ValueError, KeyError):
                valid = False
        if not valid:
            raw = load_processed_subject(processed_dir, subject)
            by_band = [[] for _ in BAND_NAMES]
            for video in range(VIDEOS):
                values = extract_four_second_probabilities(raw[video])
                for target, item in zip(by_band, values, strict=True):
                    target.append(item.astype(np.float16))
            arrays = {name: np.stack(items) for name, items in zip(BAND_NAMES, by_band, strict=True)}
            np.savez(path, **arrays)
            del raw, by_band, arrays
            gc.collect()
        complete.append(subject)
        if (subject + 1) % 10 == 0 or subject + 1 == SUBJECTS:
            print(f"4s cache {subject + 1:03d}/{SUBJECTS} elapsed={time.perf_counter() - started:.1f}s")
    manifest = {
        "dataset": "FACED",
        "feature": "centered_4s_welch_probability_spectrum",
        "analysis_seconds": 4,
        "output_stride_seconds": 1,
        "reflection_padding_samples_each_side": 375,
        "welch": {"nperseg": 500, "noverlap": 250, "nfft": 500, "segments": 3, "frequency_resolution_hz": 0.5},
        "frequency_integration": "sum PSD density over [k,k+1) then multiply by df=0.5",
        "band_names": list(BAND_NAMES),
        "band_sizes": list(BAND_SIZES),
        "storage_dtype": "float16",
        "subjects_complete": complete,
        "all_subjects_complete": len(complete) == SUBJECTS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_probabilities(cache_root: Path, subject: int) -> list[np.ndarray]:
    path = cache_root / "subjects" / f"sub{subject:03d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        values = [np.asarray(archive[name], dtype=np.float32) for name in BAND_NAMES]
    for value, size in zip(values, BAND_SIZES, strict=True):
        if value.shape != (VIDEOS, TIME_STEPS, CHANNELS, size):
            raise ValueError(f"Bad probability cache shape: {path}/{value.shape}")
        if not np.allclose(value.sum(axis=-1), 1.0, atol=2e-3):
            raise ValueError(f"Probability normalization failed: {path}")
    return values


def fit_reference(cache_root: Path, subjects: Sequence[int]) -> list[np.ndarray]:
    sums = [np.zeros((CHANNELS, size), dtype=np.float64) for size in BAND_SIZES]
    windows = 0
    for subject in subjects:
        for band, value in enumerate(load_probabilities(cache_root, int(subject))):
            sums[band] += value.sum(axis=(0, 1), dtype=np.float64)
        windows += VIDEOS * TIME_STEPS
    output = []
    for value in sums:
        reference = value / windows
        reference /= reference.sum(axis=-1, keepdims=True)
        output.append(reference.astype(np.float32))
    return output


def materialize_variant(
    cache_root: Path,
    subjects: Sequence[int],
    reference: Sequence[np.ndarray],
    variant: Variant,
) -> dict[str, np.ndarray]:
    samples = len(subjects) * VIDEOS
    shape = np.zeros((samples, TIME_STEPS, CHANNELS, len(BAND_NAMES), max(BAND_SIZES)), dtype=np.float16)
    magnitude = np.zeros((samples, TIME_STEPS, CHANNELS, len(BAND_NAMES)), dtype=np.float32)
    for subject_index, subject in enumerate(subjects):
        start = subject_index * VIDEOS
        for band, (p, q, size) in enumerate(zip(load_probabilities(cache_root, int(subject)), reference, BAND_SIZES, strict=True)):
            midpoint = 0.5 * (p + q[None, None, :, :])
            contribution = 0.5 * (
                p * (np.log(p + EPSILON) - np.log(midpoint + EPSILON))
                + q[None, None, :, :] * (np.log(q[None, None, :, :] + EPSILON) - np.log(midpoint + EPSILON))
            )
            contribution = np.maximum(contribution, 0.0).astype(np.float32)
            r = np.sqrt(contribution, dtype=np.float32)
            m = np.sqrt(np.square(r).sum(axis=-1), dtype=np.float32)
            magnitude[start : start + VIDEOS, ..., band] = m
            if variant.normalized_shape:
                current = r / np.maximum(m[..., None], EPSILON)
                if variant.signed_shape:
                    current *= np.sign(p - q[None, None, :, :]).astype(np.float32)
            else:
                current = r
            shape[start : start + VIDEOS, ..., band, :size] = current.astype(np.float16)
    return {
        "shape": shape,
        "magnitude": magnitude,
        "labels": np.tile(VIDEO_LABELS, len(subjects)).astype(np.int64),
    }


def materialize_magnitude(
    cache_root: Path,
    subjects: Sequence[int],
    reference: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    samples = len(subjects) * VIDEOS
    magnitude = np.zeros((samples, TIME_STEPS, CHANNELS, len(BAND_NAMES)), dtype=np.float32)
    for subject_index, subject in enumerate(subjects):
        start = subject_index * VIDEOS
        for band, (p, q) in enumerate(zip(load_probabilities(cache_root, int(subject)), reference, strict=True)):
            midpoint = 0.5 * (p + q[None, None, :, :])
            contribution = 0.5 * (
                p * (np.log(p + EPSILON) - np.log(midpoint + EPSILON))
                + q[None, None, :, :] * (np.log(q[None, None, :, :] + EPSILON) - np.log(midpoint + EPSILON))
            )
            magnitude[start : start + VIDEOS, ..., band] = np.sqrt(
                np.maximum(contribution, 0.0).sum(axis=-1),
                dtype=np.float32,
            )
    labels = np.tile(VIDEO_LABELS, len(subjects)).astype(np.int64)
    return magnitude, labels


def fit_shape_standardizer(values: np.ndarray, valid_sizes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(values.shape[2:], dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for start in range(0, len(values), 32):
        chunk = values[start : start + 32].astype(np.float32)
        total += chunk.sum(axis=(0, 1), dtype=np.float64)
        square += np.square(chunk).sum(axis=(0, 1), dtype=np.float64)
        count += chunk.shape[0] * chunk.shape[1]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-7] = 1.0
    mask = np.zeros((1, len(valid_sizes), max(valid_sizes)), dtype=np.float32)
    for band, size in enumerate(valid_sizes):
        mask[:, band, :size] = 1.0
    mean *= mask
    std = np.where(mask > 0, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


class ShapeMagnitudeDataset(Dataset):
    def __init__(
        self,
        payload: dict[str, np.ndarray],
        shape_mean: np.ndarray,
        shape_std: np.ndarray,
        magnitude_mean: np.ndarray,
        magnitude_std: np.ndarray,
    ):
        self.shape = payload["shape"]
        self.magnitude = payload["magnitude"]
        self.labels = payload["labels"]
        self.shape_mean = shape_mean
        self.shape_std = shape_std
        self.magnitude_mean = magnitude_mean
        self.magnitude_std = magnitude_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        shape = (self.shape[index].astype(np.float32) - self.shape_mean) / self.shape_std
        magnitude = (self.magnitude[index] - self.magnitude_mean) / self.magnitude_std
        return (
            torch.from_numpy(np.ascontiguousarray(shape)),
            torch.from_numpy(np.ascontiguousarray(magnitude, dtype=np.float32)),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class ShapeMagnitudeAxialTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        layers_per_axis: int,
        ffn_ratio: int,
        magnitude_bypass: bool,
    ):
        super().__init__()
        self.magnitude_bypass = magnitude_bypass
        self.input_encoders = nn.ModuleList(nn.Linear(size, d_model, bias=False) for size in BAND_SIZES)
        self.band_embedding = nn.Parameter(torch.zeros(1, 1, 1, len(BAND_NAMES), d_model))
        self.channel_embedding = nn.Parameter(torch.zeros(1, 1, CHANNELS, 1, d_model))
        self.time_embedding = nn.Parameter(torch.zeros(1, TIME_STEPS, 1, 1, d_model))
        self.band_blocks = nn.ModuleList(PlainTransformerBlock(d_model, heads, ffn_ratio) for _ in range(layers_per_axis))
        self.channel_blocks = nn.ModuleList(PlainTransformerBlock(d_model, heads, ffn_ratio) for _ in range(layers_per_axis))
        self.time_blocks = nn.ModuleList(PlainTransformerBlock(d_model, heads, ffn_ratio) for _ in range(layers_per_axis))
        self.output_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        magnitude_dim = 2 * CHANNELS * len(BAND_NAMES) if magnitude_bypass else 0
        self.classifier = nn.Linear(2 * d_model + magnitude_dim, CLASSES, bias=False)
        nn.init.trunc_normal_(self.band_embedding, std=0.02)
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def encode_representation(self, shape: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        encoded = [
            encoder(shape[..., band, :size])
            for band, (size, encoder) in enumerate(zip(BAND_SIZES, self.input_encoders, strict=True))
        ]
        tokens = torch.stack(encoded, dim=3)
        tokens = tokens + self.band_embedding + self.channel_embedding + self.time_embedding
        batch, time_steps, channels, bands, d_model = tokens.shape
        sequence = tokens.reshape(batch * time_steps * channels, bands, d_model)
        for block in self.band_blocks:
            sequence = block(sequence)
        tokens = sequence.reshape(batch, time_steps, channels, bands, d_model)
        sequence = tokens.permute(0, 1, 3, 2, 4).reshape(batch * time_steps * bands, channels, d_model)
        for block in self.channel_blocks:
            sequence = block(sequence)
        tokens = sequence.reshape(batch, time_steps, bands, channels, d_model).permute(0, 1, 3, 2, 4)
        sequence = tokens.permute(0, 2, 3, 1, 4).reshape(batch * channels * bands, time_steps, d_model)
        for block in self.time_blocks:
            sequence = block(sequence)
        tokens = sequence.reshape(batch, channels, bands, time_steps, d_model).permute(0, 3, 1, 2, 4)
        tokens = self.output_norm(tokens)
        shape_mean = tokens.mean(dim=(1, 2, 3))
        shape_std = torch.sqrt(tokens.var(dim=(1, 2, 3), unbiased=False).clamp_min(1e-6))
        representation = [shape_mean, shape_std]
        if self.magnitude_bypass:
            magnitude_mean = magnitude.mean(dim=1)
            magnitude_std = torch.sqrt(magnitude.var(dim=1, unbiased=False).clamp_min(1e-6))
            representation.extend((magnitude_mean.flatten(1), magnitude_std.flatten(1)))
        return torch.cat(representation, dim=-1)

    def forward(self, shape: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_representation(shape, magnitude))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, device: torch.device, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    total = 0
    targets = []
    predictions = []
    for shape, magnitude, label in loader:
        shape = shape.to(device, non_blocking=True)
        magnitude = magnitude.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(shape, magnitude)
        loss_sum += float(criterion(logits, label).cpu()) * len(label)
        total += len(label)
        targets.append(label.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(CLASSES))
    per_class_recall = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.zeros(CLASSES, dtype=np.float64),
        where=matrix.sum(axis=1) > 0,
    )
    return {
        "loss": loss_sum / total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_classes": int(np.unique(y_pred).size),
        "per_class_recall": per_class_recall.tolist(),
        "prediction_histogram": np.bincount(y_pred, minlength=CLASSES).tolist(),
        "confusion_matrix": matrix.tolist(),
    }, y_true, y_pred


def save_confusion(matrix: Sequence[Sequence[int]], title: str, path: Path) -> None:
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(values, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title=title,
        xlabel="Predicted class",
        ylabel="True class",
        xticks=np.arange(CLASSES),
        yticks=np.arange(CLASSES),
        xticklabels=EMOTION_NAMES,
        yticklabels=EMOTION_NAMES,
    )
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_linear_magnitude_diagnostic(
    args: argparse.Namespace,
    fold: int,
    spectrum: str,
    cache_root: Path,
) -> dict[str, Any]:
    source_subjects, target_subjects = official_fold_subjects(fold)
    reference = fit_reference(cache_root, source_subjects)
    source_magnitude, source_labels = materialize_magnitude(cache_root, source_subjects, reference)
    target_magnitude, target_labels = materialize_magnitude(cache_root, target_subjects, reference)
    mean = source_magnitude.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = source_magnitude.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-7] = 1.0

    def summarize(value: np.ndarray) -> np.ndarray:
        standardized = (value - mean) / std
        time_mean = standardized.mean(axis=1)
        time_std = standardized.std(axis=1)
        return np.concatenate((time_mean.reshape(len(value), -1), time_std.reshape(len(value), -1)), axis=1)

    source_features = summarize(source_magnitude)
    target_features = summarize(target_magnitude)
    classifier = LinearSVC(C=1.0, dual="auto", max_iter=20_000, random_state=args.seed + fold)
    classifier.fit(source_features, source_labels)

    output = args.run_root / "linear_magnitude_diagnostic" / spectrum / f"fold-{fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"spectrum": spectrum, "fold": fold, "feature_dimensions": source_features.shape[1]}
    for split, features, labels in (
        ("source", source_features, source_labels),
        ("target", target_features, target_labels),
    ):
        prediction = classifier.predict(features)
        matrix = confusion_matrix(labels, prediction, labels=np.arange(CLASSES))
        metrics = {
            "accuracy": float(accuracy_score(labels, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
            "predicted_classes": int(np.unique(prediction).size),
            "confusion_matrix": matrix.tolist(),
        }
        result[split] = metrics
        save_confusion(matrix, f"{spectrum} scalar magnitude LinearSVC {split}", output / f"{split}_confusion.png")
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[linear magnitude {spectrum} fold {fold:02d}] "
        f"source_f1={result['source']['macro_f1']:.3f} "
        f"target_f1={result['target']['macro_f1']:.3f} "
        f"target_bacc={result['target']['balanced_accuracy']:.3f}"
    )
    return result


def run_variant(
    args: argparse.Namespace,
    variant: Variant,
    fold: int,
    cache_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    source_subjects, target_subjects = official_fold_subjects(fold)
    output = args.run_root / variant.name / f"fold-{fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    reference = fit_reference(cache_root, source_subjects)
    source = materialize_variant(cache_root, source_subjects, reference, variant)
    target = materialize_variant(cache_root, target_subjects, reference, variant)
    if variant.normalized_shape:
        shape_mean = np.zeros(source["shape"].shape[2:], dtype=np.float32)
        shape_std = np.ones_like(shape_mean)
    else:
        shape_mean, shape_std = fit_shape_standardizer(source["shape"], BAND_SIZES)
    magnitude_mean = source["magnitude"].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    magnitude_std = source["magnitude"].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    magnitude_std[magnitude_std < 1e-7] = 1.0
    source_dataset = ShapeMagnitudeDataset(source, shape_mean, shape_std, magnitude_mean, magnitude_std)
    target_dataset = ShapeMagnitudeDataset(target, shape_mean, shape_std, magnitude_mean, magnitude_std)
    train_loader = make_loader(source_dataset, args.batch_size, True, args.seed + fold, device, args.num_workers)
    source_loader = make_loader(source_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers)
    target_loader = make_loader(target_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers)
    seed_everything(args.seed + fold)
    model = ShapeMagnitudeAxialTransformer(
        args.d_model,
        args.heads,
        args.layers_per_axis,
        args.ffn_ratio,
        variant.magnitude_bypass,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    rows = []
    started = time.perf_counter()
    print(f"\n[{variant.name} fold {fold:02d}] params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        for shape, magnitude, label in train_loader:
            shape = shape.to(device, non_blocking=True)
            magnitude = magnitude.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(shape, magnitude)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_correct += int((logits.argmax(1) == label).sum().detach().cpu())
            train_total += len(label)
        if epoch % args.eval_every == 0:
            source_metrics, _, _ = evaluate(model, source_loader, device)
            target_metrics, target_true, target_pred = evaluate(model, target_loader, device)
            for split, metrics in (("source", source_metrics), ("target", target_metrics)):
                rows.append({
                    "variant": variant.name,
                    "fold": fold,
                    "epoch": epoch,
                    "split": split,
                    **{key: metrics[key] for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1", "predicted_classes")},
                    "per_class_recall": metrics["per_class_recall"],
                    "prediction_histogram": metrics["prediction_histogram"],
                })
                save_confusion(metrics["confusion_matrix"], f"{variant.name} epoch {epoch} {split}", output / f"epoch-{epoch:03d}_{split}_confusion.png")
            print(
                f"epoch {epoch:03d}/{args.epochs} train={train_correct / train_total:.3f} "
                f"source_f1={source_metrics['macro_f1']:.3f} source_bacc={source_metrics['balanced_accuracy']:.3f} "
                f"target_f1={target_metrics['macro_f1']:.3f} target_bacc={target_metrics['balanced_accuracy']:.3f} "
                f"classes={target_metrics['predicted_classes']} "
                f"class9_recall={target_metrics['per_class_recall'][8]:.3f}"
            )
    checkpoint = {
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "variant": variant.__dict__,
        "fold": fold,
        "shape_mean": shape_mean,
        "shape_std": shape_std,
        "magnitude_mean": magnitude_mean,
        "magnitude_std": magnitude_std,
    }
    torch.save(checkpoint, output / "model_final.pt")
    summary = {
        "variant": variant.__dict__,
        "fold": fold,
        "source_trials": len(source_dataset),
        "target_trials": len(target_dataset),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "final_source": source_metrics,
        "final_target": target_metrics,
        "history": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["target_true"] = target_true
    summary["target_pred"] = target_pred
    del model, optimizer, source, target, source_dataset, target_dataset, train_loader, source_loader, target_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    args.processed_dir = resolve_path(args.processed_dir)
    args.four_second_cache = resolve_path(args.four_second_cache)
    args.run_root = resolve_path(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    one_second_cache = find_one_second_cache()
    if args.linear_magnitude_diagnostic or any(VARIANTS[name].spectrum == "4s" for name in args.variants):
        build_four_second_cache(args.four_second_cache, args.processed_dir)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = vars(args).copy()
    for key in ("processed_dir", "four_second_cache", "run_root"):
        config[key] = str(config[key])
    config["device_resolved"] = str(device)
    config["training"] = {
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "label_smoothing": 0.0,
        "scheduler": None,
        "warmup": None,
        "early_stopping": None,
        "gradient_clipping": None,
        "amp": False,
    }
    config_name = "diagnostic_config.json" if args.diagnostic_only else "config.json"
    (args.run_root / config_name).write_text(json.dumps(config, indent=2), encoding="utf-8")
    if args.linear_magnitude_diagnostic:
        diagnostic_results = []
        for fold in args.folds:
            diagnostic_results.append(run_linear_magnitude_diagnostic(args, fold, "1s", one_second_cache))
            diagnostic_results.append(run_linear_magnitude_diagnostic(args, fold, "4s", args.four_second_cache))
        (args.run_root / "linear_magnitude_diagnostic.json").write_text(
            json.dumps(diagnostic_results, indent=2),
            encoding="utf-8",
        )
    if args.diagnostic_only:
        return
    results = []
    for name in args.variants:
        variant = VARIANTS[name]
        cache_root = one_second_cache if variant.spectrum == "1s" else args.four_second_cache
        for fold in args.folds:
            results.append(run_variant(args, variant, fold, cache_root, device))
    aggregate = {"epochs": args.epochs, "folds": args.folds, "results": {}}
    csv_rows = []
    for name in args.variants:
        variant = VARIANTS[name]
        selected = [item for item in results if item["variant"]["name"] == variant.name]
        target_true = np.concatenate([item.pop("target_true") for item in selected])
        target_pred = np.concatenate([item.pop("target_pred") for item in selected])
        target_metrics = [item["final_target"] for item in selected]
        row = {
            "variant": variant.name,
            "accuracy": float(accuracy_score(target_true, target_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(target_true, target_pred)),
            "macro_f1": float(f1_score(target_true, target_pred, average="macro", zero_division=0)),
            "predicted_classes": int(np.unique(target_pred).size),
        }
        csv_rows.append(row)
        aggregate["results"][variant.name] = {
            **row,
            "fold_mean_macro_f1": float(np.mean([metrics["macro_f1"] for metrics in target_metrics])),
            "fold_mean_balanced_accuracy": float(np.mean([metrics["balanced_accuracy"] for metrics in target_metrics])),
            "confusion_matrix": confusion_matrix(target_true, target_pred, labels=np.arange(CLASSES)).tolist(),
        }
    with (args.run_root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.run_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("\n" + json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
