from __future__ import annotations

import pickle
from collections import defaultdict
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cmrd.data.records import TrialSample
from cmrd.features.rd import normalize_histograms, transform_rd
from cmrd.io import read_json
from cmrd.models import HierarchicalChannelBandTransformer, PlainTransformer
from cmrd.training.engine import (
    LegacyDataLoaderRandomSampler,
    SequenceDataset,
    collate_sequences,
    fit_normalizer,
)
from cmrd.training.metrics import classification_metrics
from cmrd.training.runtime import seed_everything


FIXED_SEED = 42
REPRESENTATIONS = ("histogram", "de_raw", "de_zscore", "rjsd_zscore")
REFERENCE_METHODS = (
    "pooled_mean",
    "robust_median",
    "subject_balanced",
    "session_balanced",
)
MODEL_NAMES = ("logistic_regression", "linear_svm", "small_mlp", "plain_transformer", "hierarchical_attention")


def assert_fixed_protocol(signature_payload: dict[str, Any]) -> None:
    """Reject caches that do not implement the frozen first-stage protocol."""
    expected = {
        "window_seconds": 1.0,
        "hop_seconds": 1.0,
        "sampling_rate": 200,
        "hist_bins_per_band": 32,
    }
    errors = []
    for key, value in expected.items():
        if signature_payload.get(key) != value:
            errors.append(f"{key}={signature_payload.get(key)!r}, expected {value!r}")
    welch = signature_payload.get("welch", {})
    if welch.get("noverlap") != 0:
        errors.append(f"welch.noverlap={welch.get('noverlap')!r}, expected 0")
    if signature_payload.get("rjsd_definition") != "Jensen-Shannon divergence(P_window, Q_source_train)":
        errors.append("cache is not the source-only RJSD representation")
    if errors:
        raise ValueError("Cache violates the fixed protocol: " + "; ".join(errors))


def resolve_complete_cache(parent: Path, dataset: str) -> Path:
    candidates: list[Path]
    if (parent / "pipeline_manifest.json").is_file():
        candidates = [parent]
    elif parent.is_dir():
        candidates = sorted(
            (path for path in parent.iterdir() if (path / "pipeline_manifest.json").is_file()),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    else:
        candidates = []
    for path in candidates:
        pipeline = read_json(path / "pipeline_manifest.json")
        if not pipeline.get("all_15_folds_complete"):
            continue
        environment = read_json(path / "environment.json")
        assert_fixed_protocol(environment["signature_payload"])
        expected = "SEED-IV" if dataset == "seediv" else "SEED"
        if pipeline.get("dataset") != expected:
            continue
        return path
    raise FileNotFoundError(f"No complete fixed-protocol {dataset} cache under {parent}")


@lru_cache(maxsize=2048)
def _load_histogram_file(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = normalize_histograms(np.asarray(archive["p_hist"], dtype=np.float32))
    if value.ndim != 4 or not np.isfinite(value).all():
        raise ValueError(f"Invalid histogram in {path}: {value.shape}")
    value.setflags(write=False)
    return value


def _load_histogram(root: Path, entry: dict[str, Any]) -> np.ndarray:
    return _load_histogram_file(str((root / entry["de_phist_path"]).resolve()))


@lru_cache(maxsize=2048)
def _load_de_file(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["de"], dtype=np.float32)
    if value.ndim != 3 or value.shape[1:] != (62, 5) or not np.isfinite(value).all():
        raise ValueError(f"Invalid DE feature in {path}: {value.shape}")
    value.setflags(write=False)
    return value


def _load_de(root: Path, entry: dict[str, Any]) -> np.ndarray:
    return _load_de_file(str((root / entry["de_phist_path"]).resolve()))


def feature_cache_info() -> dict[str, Any]:
    """Expose decompression-cache statistics for runtime auditing."""
    histogram = _load_histogram_file.cache_info()
    de = _load_de_file.cache_info()
    return {
        "histogram": histogram._asdict(),
        "de": de._asdict(),
    }


def clear_feature_cache() -> None:
    _load_histogram_file.cache_clear()
    _load_de_file.cache_clear()


def fit_reference(
    root: Path,
    entries: Sequence[dict[str, Any]],
    method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit an RJSD reference without using validation or target entries."""
    if method not in REFERENCE_METHODS:
        raise ValueError(f"Unknown reference method: {method}")
    if not entries:
        raise ValueError("Cannot fit a reference from zero source-training trials")

    grouped_sums: dict[Any, np.ndarray] = {}
    grouped_counts: dict[Any, int] = defaultdict(int)
    pooled_sum: np.ndarray | None = None
    pooled_count = 0
    for entry in entries:
        histogram = _load_histogram(root, entry)
        current = histogram.sum(axis=0, dtype=np.float64)
        pooled_sum = current if pooled_sum is None else pooled_sum + current
        pooled_count += histogram.shape[0]
        if method in {"subject_balanced", "robust_median"}:
            key: Any = int(entry["subject"])
        elif method == "session_balanced":
            key = (int(entry["subject"]), int(entry["session"]))
        else:
            continue
        if key not in grouped_sums:
            grouped_sums[key] = current
        else:
            grouped_sums[key] += current
        grouped_counts[key] += histogram.shape[0]

    assert pooled_sum is not None and pooled_count > 0
    if method == "pooled_mean":
        raw = pooled_sum / pooled_count
        units = pooled_count
    else:
        unit_references = np.stack(
            [grouped_sums[key] / grouped_counts[key] for key in sorted(grouped_sums, key=str)]
        )
        raw = np.median(unit_references, axis=0) if method == "robust_median" else unit_references.mean(axis=0)
        units = unit_references.shape[0]
    reference = normalize_histograms(raw).astype(np.float32)
    return reference, {
        "method": method,
        "source_subjects": sorted({int(entry["subject"]) for entry in entries}),
        "source_sessions": sorted({(int(entry["subject"]), int(entry["session"])) for entry in entries}),
        "source_trials": len(entries),
        "source_windows": pooled_count,
        "balanced_units": units,
    }


def reference_leave_one_subject_sensitivity(
    root: Path,
    entries: Sequence[dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    full, _ = fit_reference(root, entries, method)
    shifts = []
    for subject in sorted({int(entry["subject"]) for entry in entries}):
        reduced = [entry for entry in entries if int(entry["subject"]) != subject]
        reference, _ = fit_reference(root, reduced, method)
        shifts.append({
            "left_out_subject": subject,
            "mean_absolute_shift": float(np.mean(np.abs(reference - full))),
            "max_absolute_shift": float(np.max(np.abs(reference - full))),
        })
    return {
        "method": method,
        "by_subject": shifts,
        "mean_absolute_shift": float(np.mean([row["mean_absolute_shift"] for row in shifts])),
        "max_absolute_shift": float(max(row["max_absolute_shift"] for row in shifts)),
    }


def load_rjsd_samples(
    root: Path,
    entries: Iterable[dict[str, Any]],
    reference: np.ndarray,
    channels: int = 62,
) -> list[TrialSample]:
    samples = []
    for entry in entries:
        histogram = _load_histogram(root, entry)
        flat = transform_rd(histogram, reference)
        if flat.shape[1] % channels:
            raise ValueError(f"RJSD feature dimension {flat.shape[1]} is not divisible by {channels}")
        samples.append(TrialSample(
            np.ascontiguousarray(flat, dtype=np.float32),
            int(entry["label"]),
            int(entry["subject"]),
            int(entry["session"]),
            int(entry["trial"]),
            int(entry["source_index"]),
        ))
    return samples


def representation_uses_source_zscore(representation: str) -> bool:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    return representation in {"de_zscore", "rjsd_zscore"}


def load_representation_samples(
    root: Path,
    entries: Iterable[dict[str, Any]],
    representation: str,
    reference: np.ndarray | None = None,
    channels: int = 62,
) -> list[TrialSample]:
    """Load one declared representation without fitting any data-dependent state."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    if representation == "rjsd_zscore":
        if reference is None:
            raise ValueError("RJSD requires an explicitly fitted source reference")
        return load_rjsd_samples(root, entries, reference, channels)

    samples: list[TrialSample] = []
    for entry in entries:
        if representation == "histogram":
            value = _load_histogram(root, entry)
        else:
            value = _load_de(root, entry)
        flat = value.reshape(value.shape[0], -1)
        if flat.shape[1] % channels:
            raise ValueError(f"Feature dimension {flat.shape[1]} is not divisible by {channels}")
        samples.append(TrialSample(
            np.ascontiguousarray(flat, dtype=np.float16 if representation == "histogram" else np.float32),
            int(entry["label"]),
            int(entry["subject"]),
            int(entry["session"]),
            int(entry["trial"]),
            int(entry["source_index"]),
        ))
    return samples


def pooled_vectors(samples: Sequence[TrialSample]) -> np.ndarray:
    """Trial-level mean and standard deviation used by non-sequential baselines."""
    return np.stack([
        np.concatenate((
            sample.x.mean(axis=0, dtype=np.float32),
            sample.x.std(axis=0, dtype=np.float32),
        )) for sample in samples
    ]).astype(np.float32)


class SmallPooledMLP(nn.Module):
    """Small classifier for source-standardized trial-level pooled features."""

    def __init__(self, input_dim: int, classes: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            weights = mask.unsqueeze(-1).to(features.dtype)
            features = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.network(features)


class HistogramHierarchicalTransformer(nn.Module):
    """Compress each 32-bin histogram before channel/band/temporal attention.

    Expanding all 9,920 scalar histogram bins into ``d_model`` tokens is
    prohibitively expensive.  This adapter still consumes the complete
    probability histogram, but learns one projection per channel-band before
    applying the same hierarchical backbone used by the 310-dimensional
    DE/RJSD representations.
    """

    def __init__(
        self,
        input_dim: int,
        channels: int,
        classes: int,
        max_length: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        dropout: float,
        bands: int = 5,
        bins: int = 32,
    ) -> None:
        super().__init__()
        expected = channels * bands * bins
        if input_dim != expected:
            raise ValueError(f"Histogram input_dim={input_dim}, expected {expected}")
        self.channels = channels
        self.bands = bands
        self.bins = bins
        self.bin_projection = nn.Linear(bins, 1)
        self.backbone = HierarchicalChannelBandTransformer(
            input_dim=channels * bands,
            channels=channels,
            classes=classes,
            max_length=max_length,
            d_model=d_model,
            channel_heads=heads,
            temporal_heads=heads,
            temporal_layers=layers,
            feedforward=feedforward,
            dropout=dropout,
        )

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
        valid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        structured = data.reshape(*data.shape[:2], self.channels, self.bands, self.bins)
        compressed = self.bin_projection(structured).squeeze(-1).reshape(*data.shape[:2], -1)
        return self.backbone(
            compressed,
            mask,
            return_attention=return_attention,
            valid_indices=valid_indices,
        )


def build_model(
    config: dict[str, Any],
    input_dim: int,
    classes: int,
    max_length: int,
    channels: int = 62,
) -> nn.Module:
    name = str(config["name"])
    heads = int(config.get("heads", config.get("nhead", 4)))
    if name == "small_mlp":
        return SmallPooledMLP(input_dim, classes, int(config.get("hidden", 128)), float(config["dropout"]))
    if name == "plain_transformer":
        return PlainTransformer(
            input_dim, classes, max_length, int(config["d_model"]), heads,
            int(config["layers"]), int(config["feedforward"]), float(config["dropout"]),
        )
    if name == "hierarchical_attention":
        if input_dim == channels * 5 * 32:
            return HistogramHierarchicalTransformer(
                input_dim=input_dim,
                channels=channels,
                classes=classes,
                max_length=max_length,
                d_model=int(config["d_model"]),
                heads=heads,
                layers=int(config["layers"]),
                feedforward=int(config["feedforward"]),
                dropout=float(config["dropout"]),
            )
        return HierarchicalChannelBandTransformer(
            input_dim=input_dim,
            channels=channels,
            classes=classes,
            max_length=max_length,
            d_model=int(config["d_model"]),
            channel_heads=heads,
            temporal_heads=heads,
            temporal_layers=int(config["layers"]),
            feedforward=int(config["feedforward"]),
            dropout=float(config["dropout"]),
        )
    raise ValueError(f"Model {name!r} is not a neural baseline")


def _loader(
    samples: Sequence[TrialSample], mean: np.ndarray, std: np.ndarray,
    batch_size: int, shuffle: bool, seed: int,
    *,
    num_workers: int = 0,
    persistent_workers: bool = False,
    prefetch_factor: int = 1,
    cache_normalized: bool = False,
) -> DataLoader:
    workers = int(num_workers)
    dataset = SequenceDataset(
        list(samples),
        mean,
        std,
        cache_normalized=cache_normalized,
        share_memory=cache_normalized and workers > 0,
    )
    loader_options: dict[str, Any] = {}
    if workers > 0:
        loader_options["prefetch_factor"] = int(prefetch_factor)
    sampler = LegacyDataLoaderRandomSampler(dataset, seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=bool(persistent_workers and workers > 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_sequences,
        # Keep worker seeding separate from the sampler RNG so persistent
        # workers cannot alter the established epoch permutations.
        generator=torch.Generator().manual_seed(seed + 1_000_003),
        **loader_options,
    )


def _shutdown_persistent_loader(loader: DataLoader) -> None:
    """Release Windows worker processes and their shared mappings promptly."""
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    if iterator is not None:
        loader._iterator = None


def _valid_indices_on_device(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Build dynamic valid-window indices on CPU, then transfer asynchronously."""
    indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    return indices.to(device, non_blocking=True)


def _forward_sequence_model(
    model: nn.Module,
    data: torch.Tensor,
    mask: torch.Tensor,
    valid_indices: torch.Tensor | None,
    *,
    return_attention: bool = False,
):
    if isinstance(model, (HierarchicalChannelBandTransformer, HistogramHierarchicalTransformer)):
        if valid_indices is None:
            raise ValueError("Hierarchical models require CPU-precomputed valid_indices")
        return model(
            data,
            mask,
            return_attention=return_attention,
            valid_indices=valid_indices,
        )
    if return_attention:
        raise ValueError("Attention output is only available for hierarchical models")
    return model(data, mask)


def _autocast(training: dict[str, Any], device: torch.device):
    precision = str(training.get("precision", "float32")).lower()
    if device.type != "cuda" or precision == "float32":
        return nullcontext()
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported training precision: {precision}")


@torch.no_grad()
def predict_neural(
    model: nn.Module, loader: DataLoader, device: torch.device, precision: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets, predictions = [], []
    for data, mask, labels in loader:
        valid_indices = (
            _valid_indices_on_device(mask, device)
            if isinstance(model, (HierarchicalChannelBandTransformer, HistogramHierarchicalTransformer))
            else None
        )
        with _autocast({"precision": precision}, device):
            logits = _forward_sequence_model(
                model,
                data.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                valid_indices,
            )
        targets.append(labels.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def scaling_statistics(
    samples: Sequence[TrialSample],
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Cannot determine scaling without source samples")
    if enabled:
        return fit_normalizer(list(samples))
    dimension = int(samples[0].x.shape[1])
    return np.zeros(dimension, dtype=np.float32), np.ones(dimension, dtype=np.float32)


def fit_locked_source_model(
    source_samples: Sequence[TrialSample],
    model_config: dict[str, Any],
    training: dict[str, Any],
    classes: int,
    device: torch.device,
    checkpoint_path: Path,
    seed: int = FIXED_SEED,
    scale_inputs: bool = True,
    context: dict[str, Any] | None = None,
) -> None:
    """Fit locked epochs on all source subjects; target data cannot enter this function."""
    seed_everything(seed, True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(training.get("matmul_precision", "high")))
    if str(model_config["name"]) == "small_mlp":
        _fit_locked_pooled_mlp(
            source_samples, model_config, training, classes, device,
            checkpoint_path, seed, context,
        )
        return
    mean, std = scaling_statistics(source_samples, scale_inputs)
    input_dim = source_samples[0].x.shape[1]
    max_length = max(sample.x.shape[0] for sample in source_samples)
    model = build_model(model_config, input_dim, classes, max_length).to(device)
    loader = _loader(
        source_samples,
        mean,
        std,
        int(training["batch_size"]),
        True,
        seed,
        num_workers=int(training.get("num_workers", 0)),
        persistent_workers=bool(training.get("persistent_workers", False)),
        prefetch_factor=int(training.get("prefetch_factor", 1)),
        cache_normalized=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training.get("label_smoothing", 0.0)))
    epochs = int(training["locked_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(training.get("minimum_learning_rate", 1e-6))
    )
    try:
        for _ in range(epochs):
            model.train()
            for data, mask, labels in loader:
                valid_indices = (
                    _valid_indices_on_device(mask, device)
                    if isinstance(model, (HierarchicalChannelBandTransformer, HistogramHierarchicalTransformer))
                    else None
                )
                data = data.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(training, device):
                    loss = criterion(
                        _forward_sequence_model(model, data, mask, valid_indices),
                        labels,
                    )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
                optimizer.step()
            scheduler.step()
    finally:
        _shutdown_persistent_loader(loader)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalization_mean": mean,
        "normalization_std": std,
        "model": model_config,
        "training": training,
        "classes": classes,
        "input_dim": input_dim,
        "max_length": max_length,
        "seed": seed,
        "target_loaded": False,
        "source_zscore": scale_inputs,
        "context": context or {},
    }, checkpoint_path)


def _pooled_feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _fit_locked_pooled_mlp(
    source_samples: Sequence[TrialSample],
    model_config: dict[str, Any],
    training: dict[str, Any],
    classes: int,
    device: torch.device,
    checkpoint_path: Path,
    seed: int,
    context: dict[str, Any] | None,
) -> None:
    """Train the MLP on the same trial mean+std features as classical baselines."""
    features = pooled_vectors(source_samples)
    mean, std = _pooled_feature_statistics(features)
    normalized = np.ascontiguousarray((features - mean) / std, dtype=np.float32)
    labels = np.asarray([sample.label for sample in source_samples], dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(normalized), torch.from_numpy(labels))
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
    model = SmallPooledMLP(
        normalized.shape[1], classes, int(model_config.get("hidden", 128)), float(model_config["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training.get("label_smoothing", 0.0)))
    epochs = int(training["locked_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(training.get("minimum_learning_rate", 1e-6))
    )
    for _ in range(epochs):
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(training, device):
                loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 1.0)))
            optimizer.step()
        scheduler.step()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "normalization_mean": mean,
        "normalization_std": std,
        "model": model_config,
        "training": training,
        "classes": classes,
        "input_dim": int(normalized.shape[1]),
        "max_length": 1,
        "seed": seed,
        "target_loaded": False,
        "source_zscore": True,
        "input_adapter": "trial_mean_std_source_zscore_v1",
        "context": context or {},
    }, checkpoint_path)


def evaluate_locked_checkpoint(
    checkpoint_path: Path,
    target_samples: Sequence[TrialSample],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """The sole target-access boundary for neural final evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("input_adapter") == "trial_mean_std_source_zscore_v1":
        return _evaluate_locked_pooled_mlp(checkpoint, target_samples, device)
    model = build_model(
        checkpoint["model"], int(checkpoint["input_dim"]), int(checkpoint["classes"]),
        max(int(checkpoint["max_length"]), max(sample.x.shape[0] for sample in target_samples)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = _loader(
        target_samples,
        np.asarray(checkpoint["normalization_mean"]),
        np.asarray(checkpoint["normalization_std"]),
        int(checkpoint["training"]["batch_size"]),
        False,
        int(checkpoint["seed"]),
    )
    targets, predictions = predict_neural(
        model, loader, device, str(checkpoint["training"].get("precision", "float32"))
    )
    rows = [{
        "source_index": int(sample.source_index),
        "subject": int(sample.subject),
        "session": int(sample.session),
        "trial": int(sample.trial),
        "target": int(target),
        "prediction": int(prediction),
    } for sample, target, prediction in zip(target_samples, targets, predictions)]
    return classification_metrics(targets, predictions, int(checkpoint["classes"])), rows


@torch.no_grad()
def _evaluate_locked_pooled_mlp(
    checkpoint: dict[str, Any],
    target_samples: Sequence[TrialSample],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    model_config = checkpoint["model"]
    model = SmallPooledMLP(
        int(checkpoint["input_dim"]), int(checkpoint["classes"]),
        int(model_config.get("hidden", 128)), float(model_config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    features = pooled_vectors(target_samples)
    mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = np.ascontiguousarray((features - mean) / std, dtype=np.float32)
    loader = DataLoader(
        torch.from_numpy(normalized),
        batch_size=int(checkpoint["training"]["batch_size"]),
        shuffle=False,
    )
    model.eval()
    predictions = []
    for batch_features in loader:
        with _autocast(checkpoint["training"], device):
            logits = model(batch_features.to(device, non_blocking=True))
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    targets = np.asarray([sample.label for sample in target_samples], dtype=np.int64)
    predicted = np.concatenate(predictions)
    rows = [{
        "source_index": int(sample.source_index),
        "subject": int(sample.subject),
        "session": int(sample.session),
        "trial": int(sample.trial),
        "target": int(target),
        "prediction": int(prediction),
    } for sample, target, prediction in zip(target_samples, targets, predicted)]
    return classification_metrics(targets, predicted, int(checkpoint["classes"])), rows


def fit_locked_classical_model(
    source_samples: Sequence[TrialSample],
    name: str,
    classes: int,
    model_path: Path,
    seed: int = FIXED_SEED,
    scale_inputs: bool = True,
    context: dict[str, Any] | None = None,
    estimator_options: dict[str, Any] | None = None,
) -> None:
    if name not in {"logistic_regression", "linear_svm"}:
        raise ValueError(name)
    mean, std = scaling_statistics(source_samples, scale_inputs)
    scaled = (
        [TrialSample((sample.x - mean) / std, sample.label, sample.subject, sample.session, sample.trial, sample.source_index) for sample in source_samples]
        if scale_inputs else list(source_samples)
    )
    options = dict(estimator_options or {})
    estimator = (
        LogisticRegression(max_iter=2000, random_state=seed)
        if name == "logistic_regression"
        else LinearSVC(
            random_state=seed,
            dual="auto",
            tol=float(options.get("tol", 1e-3)),
            max_iter=int(options.get("max_iter", 5000)),
        )
    )
    estimator.fit(pooled_vectors(scaled), np.asarray([sample.label for sample in scaled]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump({
            "model": estimator,
            "mean": mean,
            "std": std,
            "classes": classes,
            "seed": seed,
            "name": name,
            "target_loaded": False,
            "source_zscore": scale_inputs,
            "estimator_options": options,
            "context": context or {},
        }, stream)


def evaluate_locked_classical_model(
    model_path: Path,
    target_samples: Sequence[TrialSample],
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    with model_path.open("rb") as stream:
        payload = pickle.load(stream)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    scaled = (
        [TrialSample((sample.x - mean) / std, sample.label, sample.subject, sample.session, sample.trial, sample.source_index) for sample in target_samples]
        if bool(payload.get("source_zscore", True)) else list(target_samples)
    )
    targets = np.asarray([sample.label for sample in scaled])
    predictions = np.asarray(payload["model"].predict(pooled_vectors(scaled)))
    rows = [{
        "source_index": int(sample.source_index),
        "subject": int(sample.subject),
        "session": int(sample.session),
        "trial": int(sample.trial),
        "target": int(target),
        "prediction": int(prediction),
    } for sample, target, prediction in zip(scaled, targets, predictions)]
    return classification_metrics(targets, predictions, int(payload["classes"])), rows
