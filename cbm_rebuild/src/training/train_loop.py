from __future__ import annotations

import csv
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import project_path
from src.data.loso_split import source_train_validation_split
from src.models.plain_transformer import PlainTransformer
from src.preprocessing.normalization import fit_source_normalizer, normalize_trial

from .logger import close_logger, create_logger
from .metrics import classification_metrics
from .utils import seed_everything, select_device


class TrialDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.data = data
        self.mask = mask
        self.labels = labels
        self.indices = indices
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        real_mask = self.mask[index].astype(np.bool_, copy=False)
        normalized = np.zeros_like(self.data[index], dtype=np.float32)
        normalized[real_mask] = normalize_trial(self.data[index, real_mask], self.mean, self.std)
        return (
            torch.from_numpy(normalized),
            torch.from_numpy(real_mask.copy()),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> dict[str, object]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for data, mask, labels in loader:
        logits = model(data.to(device), mask.to(device))
        targets.append(labels.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return classification_metrics(np.concatenate(targets), np.concatenate(predictions), num_classes)


def _loader(dataset: Dataset, config: dict, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=bool(config.get("pin_memory", True) and torch.cuda.is_available()),
        generator=generator,
    )


def _train_fold(
    config: dict,
    dataset_name: str,
    arrays: dict[str, np.ndarray],
    target_subject: int,
    output_dirs: dict[str, Path],
    device: torch.device,
) -> dict[str, object]:
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed + target_subject, bool(training.get("deterministic", True)))
    train_idx, val_idx, test_idx = source_train_validation_split(
        arrays["subject"],
        arrays["y"],
        target_subject,
        float(training["validation_fraction"]),
        seed,
    )
    mean, std = fit_source_normalizer(arrays["X"], arrays["mask"], train_idx)
    train_set = TrialDataset(arrays["X"], arrays["mask"], arrays["y"], train_idx, mean, std)
    val_set = TrialDataset(arrays["X"], arrays["mask"], arrays["y"], val_idx, mean, std)
    test_set = TrialDataset(arrays["X"], arrays["mask"], arrays["y"], test_idx, mean, std)
    train_loader = _loader(train_set, training, True, seed + target_subject)
    val_loader = _loader(val_set, training, False, seed + target_subject)
    test_loader = _loader(test_set, training, False, seed + target_subject)

    model_config = config["model"]
    model = PlainTransformer(
        input_dim=arrays["X"].shape[-1],
        num_classes=int(config["data"]["num_classes"]),
        max_length=arrays["X"].shape[1],
        d_model=int(model_config["d_model"]),
        nhead=int(model_config["nhead"]),
        num_layers=int(model_config["num_layers"]),
        dim_feedforward=int(model_config["dim_feedforward"]),
        dropout=float(model_config["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    fold_log = output_dirs["logs"] / f"{dataset_name.lower().replace('-', '')}_fold_{target_subject:02d}.log"
    logger = create_logger(f"{dataset_name}.fold.{target_subject}", fold_log)
    logger.info(
        "target_subject=%02d device=%s train=%d validation=%d test=%d",
        target_subject,
        device,
        len(train_set),
        len(val_set),
        len(test_set),
    )

    best_score = (-float("inf"), -float("inf"))
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for data, mask, labels in train_loader:
            data = data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data, mask)
            loss = criterion(logits, labels)
            loss.backward()
            clip_norm = float(training.get("gradient_clip_norm", 0.0))
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            running_loss += float(loss.item()) * labels.shape[0]
            seen += labels.shape[0]
        val_metrics = evaluate(model, val_loader, device, int(config["data"]["num_classes"]))
        score = (float(val_metrics["macro_f1"]), float(val_metrics["accuracy"]))
        logger.info(
            "epoch=%03d train_loss=%.6f val_accuracy=%.6f val_macro_f1=%.6f",
            epoch,
            running_loss / max(seen, 1),
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                logger.info("early_stop epoch=%d", epoch)
                break
    if best_state is None:
        raise RuntimeError("Training completed without a model state")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, int(config["data"]["num_classes"]))
    checkpoint = output_dirs["checkpoints"] / f"{dataset_name.lower().replace('-', '')}_fold_{target_subject:02d}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "normalization_mean": mean,
            "normalization_std": std,
            "target_subject": target_subject,
            "best_epoch": best_epoch,
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
        checkpoint,
    )
    elapsed = time.perf_counter() - started
    logger.info(
        "final best_epoch=%d target_accuracy=%.6f target_macro_f1=%.6f checkpoint=%s",
        best_epoch,
        test_metrics["accuracy"],
        test_metrics["macro_f1"],
        checkpoint,
    )
    result = {
        "target_subject": target_subject,
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score[0],
        "best_validation_accuracy": best_score[1],
        "num_train_trials": len(train_set),
        "num_validation_trials": len(val_set),
        "num_test_trials": len(test_set),
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint),
        "log": str(fold_log),
    }
    close_logger(logger)
    return result


def _load_processed(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}. Run preprocessing first.")
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "mask", "y", "subject", "session", "trial"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing arrays: {sorted(missing)}")
        arrays = {key: archive[key] for key in required}
    if arrays["X"].ndim != 3 or arrays["mask"].shape != arrays["X"].shape[:2]:
        raise ValueError(f"Invalid X/mask shapes: {arrays['X'].shape}, {arrays['mask'].shape}")
    if not np.all(arrays["mask"].any(axis=1)):
        raise ValueError("Every trial must contain at least one real window")
    return arrays


def _write_csv(path: Path, folds: list[dict[str, object]], summary: dict[str, float]) -> None:
    fields = [
        "target_subject", "accuracy", "macro_f1", "best_epoch", "best_validation_macro_f1",
        "best_validation_accuracy", "num_train_trials", "num_validation_trials", "num_test_trials",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(folds)
        writer.writerow({"target_subject": "mean", "accuracy": summary["mean_accuracy"], "macro_f1": summary["mean_macro_f1"]})
        writer.writerow({"target_subject": "std", "accuracy": summary["std_accuracy"], "macro_f1": summary["std_macro_f1"]})


def run_loso(config: dict, dataset_name: str) -> tuple[Path, Path]:
    training = config["training"]
    seed_everything(int(training["seed"]), bool(training.get("deterministic", True)))
    device = select_device(str(training.get("device", "auto")))
    arrays = _load_processed(project_path(config["data"]["processed_path"]))
    expected_dim = int(config["data"]["channels"]) * len(config["preprocessing"]["bands_hz"])
    if arrays["X"].shape[-1] != expected_dim:
        raise ValueError(f"Feature dimension is {arrays['X'].shape[-1]}, expected {expected_dim}")
    output = config["output"]
    output_dirs = {
        "logs": project_path(output["log_dir"]),
        "checkpoints": project_path(output["checkpoint_dir"]),
        "results": project_path(output["result_dir"]),
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    stem = str(output["result_stem"])
    overall_log_path = output_dirs["logs"] / f"{stem}.log"
    logger = create_logger(f"{dataset_name}.overall", overall_log_path)
    logger.info("dataset=%s device=%s seed=%d shape=%s", dataset_name, device, training["seed"], arrays["X"].shape)

    subjects = sorted(int(value) for value in np.unique(arrays["subject"]))
    folds: list[dict[str, object]] = []
    for subject in subjects:
        logger.info("Starting fold target_subject=%02d", subject)
        fold = _train_fold(config, dataset_name, arrays, subject, output_dirs, device)
        folds.append(fold)
        logger.info("Completed fold target_subject=%02d accuracy=%.6f macro_f1=%.6f", subject, fold["accuracy"], fold["macro_f1"])
    accuracies = np.asarray([fold["accuracy"] for fold in folds], dtype=np.float64)
    macro_f1s = np.asarray([fold["macro_f1"] for fold in folds], dtype=np.float64)
    summary = {
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std(ddof=0)),
        "mean_macro_f1": float(macro_f1s.mean()),
        "std_macro_f1": float(macro_f1s.std(ddof=0)),
    }
    payload = {
        "dataset": dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(training["seed"]),
        "device": str(device),
        "window_seconds": float(config["preprocessing"]["window_seconds"]),
        "hop_seconds": float(config["preprocessing"]["hop_seconds"]),
        "maximum_sequence_length": int(arrays["X"].shape[1]),
        "trials_per_subject": dict(sorted(Counter(int(x) for x in arrays["subject"]).items())),
        "processed_shape": list(arrays["X"].shape),
        "summary": summary,
        "folds": folds,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "overall_log": str(overall_log_path),
    }
    csv_path = output_dirs["results"] / f"{stem}.csv"
    json_path = output_dirs["results"] / f"{stem}.json"
    config_path = output_dirs["results"] / f"{stem}_config.json"
    _write_csv(csv_path, folds, summary)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(payload["config"], indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("summary=%s", summary)
    logger.info("Saved %s and %s", csv_path, json_path)
    close_logger(logger)
    return csv_path, json_path
