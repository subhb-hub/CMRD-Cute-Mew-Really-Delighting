from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cmrd.data.records import TrialSample
from cmrd.io import write_json
from cmrd.models import PlainTransformer

from .metrics import classification_metrics
from .runtime import seed_everything


def fit_normalizer(samples: list[TrialSample]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Cannot fit normalization without source-training trials")
    feature_dim = samples[0].x.shape[1]
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for sample in samples:
        value = np.asarray(sample.x, dtype=np.float64)
        total += value.sum(axis=0)
        total_sq += np.square(value).sum(axis=0)
        count += value.shape[0]
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class SequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[TrialSample],
        mean: np.ndarray,
        std: np.ndarray,
        *,
        cache_normalized: bool = False,
        share_memory: bool = False,
    ) -> None:
        self.samples: list[TrialSample] | None = samples
        self.mean = mean
        self.std = std
        self._labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
        self._offsets: np.ndarray | None = None
        self._normalized: torch.Tensor | None = None
        if cache_normalized:
            lengths = np.asarray([sample.x.shape[0] for sample in samples], dtype=np.int64)
            offsets = np.empty(len(samples) + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(lengths, out=offsets[1:])
            feature_dim = int(samples[0].x.shape[1]) if samples else int(mean.shape[0])
            normalized = torch.empty((int(offsets[-1]), feature_dim), dtype=torch.float32)
            if share_memory:
                # Allocate shared storage before filling it to avoid a second
                # full-size copy at worker startup.
                normalized.share_memory_()
            for index, sample in enumerate(samples):
                value = np.ascontiguousarray((sample.x - mean) / std, dtype=np.float32)
                normalized[int(offsets[index]):int(offsets[index + 1])].copy_(torch.from_numpy(value))
            self._offsets = offsets
            self._normalized = normalized
            # Workers only receive the compact contiguous cache, not the large
            # original list of NumPy arrays. This matters on Windows spawn.
            self.samples = None

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int):
        if self._normalized is not None and self._offsets is not None:
            start, stop = int(self._offsets[index]), int(self._offsets[index + 1])
            return self._normalized[start:stop], int(self._labels[index])
        assert self.samples is not None
        sample = self.samples[index]
        normalized = (sample.x - self.mean) / self.std
        return torch.from_numpy(normalized.astype(np.float32)), sample.label


class LegacyDataLoaderRandomSampler(Sampler[int]):
    """Preserve the old num_workers=0 shuffle stream with persistent workers.

    PyTorch's regular DataLoader consumes one random value for its iterator
    seed before RandomSampler draws each epoch permutation. Persistent workers
    omit that iterator recreation on later epochs. Replaying that one draw in
    the sampler keeps every epoch's sample order identical to the old loader,
    while a separate generator is used solely for worker initialization.
    """

    def __init__(self, data_source: Dataset, seed: int) -> None:
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        yield from torch.randperm(len(self.data_source), generator=self.generator).tolist()
        # RandomSampler also draws a second permutation and takes an empty
        # remainder slice when num_samples == len(dataset). Preserve that RNG
        # advance so later epochs remain byte-for-byte compatible.
        torch.randperm(len(self.data_source), generator=self.generator)

    def __len__(self) -> int:
        return len(self.data_source)


def collate_sequences(batch):
    if not batch:
        raise ValueError("Cannot collate an empty sequence batch")
    lengths = [item[0].shape[0] for item in batch]
    if any(length < 1 for length in lengths):
        raise ValueError("Every sequence must contain at least one valid time step")
    maximum = max(lengths)
    features = batch[0][0].shape[1]
    data = torch.zeros(len(batch), maximum, features, dtype=torch.float32)
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.long)
    for index, (value, label) in enumerate(batch):
        length = value.shape[0]
        data[index, :length] = value
        mask[index, :length] = True
        labels[index] = int(label)
    return data, mask, labels


def _loader(samples: list[TrialSample], mean: np.ndarray, std: np.ndarray, training: dict, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        SequenceDataset(samples, mean, std),
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        num_workers=int(training.get("num_workers", 0)),
        pin_memory=bool(training.get("pin_memory", True) and torch.cuda.is_available()),
        collate_fn=collate_sequences,
        generator=generator,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, classes: int) -> dict[str, object]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for data, mask, labels in loader:
        logits = model(data.to(device), mask.to(device))
        targets.append(labels.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    if not targets:
        raise ValueError("Evaluation loader is empty")
    return classification_metrics(np.concatenate(targets), np.concatenate(predictions), classes)


def _write_history(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_once(
    train_samples: list[TrialSample],
    validation_samples: list[TrialSample],
    test_samples: list[TrialSample] | None,
    model_config: dict,
    training: dict,
    classes: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    context: dict[str, object],
) -> dict[str, object]:
    seed_everything(seed, bool(training.get("deterministic", True)))
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = fit_normalizer(train_samples)
    max_length = max(sample.x.shape[0] for sample in train_samples + validation_samples + (test_samples or []))
    input_dim = train_samples[0].x.shape[1]
    model = PlainTransformer(
        input_dim, classes, max_length,
        int(model_config["d_model"]), int(model_config["nhead"]), int(model_config["layers"]),
        int(model_config["feedforward"]), float(model_config["dropout"]),
    ).to(device)
    train_loader = _loader(train_samples, mean, std, training, True, seed)
    validation_loader = _loader(validation_samples, mean, std, training, False, seed)
    test_loader = _loader(test_samples, mean, std, training, False, seed) if test_samples else None
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=float(training.get("minimum_learning_rate", 1e-6)))
    best_score = (-float("inf"), -float("inf"))
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({**context, "seed": seed, "device": str(device)}) + "\n")
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            count = 0
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
                count += labels.shape[0]
            scheduler.step()
            metrics = evaluate(model, validation_loader, device, classes)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_accuracy": metrics["accuracy"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            }
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            score = (float(metrics["macro_f1"]), float(metrics["accuracy"]))
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(training["early_stopping_patience"]):
                    break
    if best_state is None:
        raise RuntimeError("Training produced no best model")
    _write_history(output_dir / "epochs.csv", history)
    model.load_state_dict(best_state)
    result: dict[str, object] = {
        **context,
        "seed": seed,
        "model": model_config,
        "optimization": {key: training[key] for key in ("learning_rate", "weight_decay", "label_smoothing", "batch_size", "epochs", "early_stopping_patience")},
        "best_epoch": best_epoch,
        "validation": evaluate(model, validation_loader, device, classes),
        "train_trials": len(train_samples),
        "validation_trials": len(validation_samples),
        "test_trials": len(test_samples or []),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if test_loader is not None:
        result["test"] = evaluate(model, test_loader, device, classes)
        torch.save({
            "model_state_dict": best_state,
            "normalization_mean": mean,
            "normalization_std": std,
            "model": model_config,
            "training": training,
            "context": context,
            "seed": seed,
            "best_epoch": best_epoch,
        }, output_dir / "best.pt")
    write_json(output_dir / "result.json", result)
    return result
