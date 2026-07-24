from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmrd.faced import EMOTION_NAMES, official_fold_subjects
from scripts.run_faced_axial_no_cls_whitebox import (
    CLASSES,
    NoCLSFullGridAxialTransformer,
    find_complete_cache,
    prepare_feature,
)
from scripts.run_faced_rjsd_shape_magnitude_ablation import (
    BAND_SIZES,
    ShapeMagnitudeAxialTransformer,
    VARIANTS,
    build_four_second_cache,
    fit_reference as fit_e_reference,
    materialize_variant,
    save_confusion,
)


MODES = ("de_de_control", "de_e_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACED DE + signed RJSD E fusion and matched DE+DE control")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--folds", nargs="+", type=int, default=(1,))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers-per-axis", type=int, default=1)
    parser.add_argument("--ffn-ratio", type=int, default=2)
    parser.add_argument("--fusion-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--four-second-cache",
        type=Path,
        default=Path("runs/faced_4s_welch_probability_cache"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ROOT.parent / "Dataset" / "Ori" / "FACED" / "Processed_data",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/faced_de_signed_rjsd_fusion"),
    )
    args = parser.parse_args()
    if any(fold < 1 or fold > 10 for fold in args.folds):
        parser.error("--folds must be in 1..10")
    if args.epochs <= 0 or args.eval_every <= 0 or args.epochs % args.eval_every:
        parser.error("--epochs must be positive and divisible by --eval-every")
    if args.d_model % args.heads:
        parser.error("--d-model must be divisible by --heads")
    return args


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DEOnlyDataset(Dataset):
    def __init__(self, payload: dict[str, Any], mean: np.ndarray, std: np.ndarray):
        self.value = payload["x"]
        self.labels = payload["y"]
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        value = (self.value[index].astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(value)), torch.tensor(self.labels[index], dtype=torch.long)


class DEEDataset(Dataset):
    def __init__(
        self,
        de_payload: dict[str, Any],
        e_payload: dict[str, np.ndarray],
        de_mean: np.ndarray,
        de_std: np.ndarray,
        magnitude_mean: np.ndarray,
        magnitude_std: np.ndarray,
    ):
        if not np.array_equal(de_payload["y"], e_payload["labels"]):
            raise ValueError("DE and E labels are not aligned")
        self.de = de_payload["x"]
        self.shape = e_payload["shape"]
        self.magnitude = e_payload["magnitude"]
        self.labels = de_payload["y"]
        self.de_mean = de_mean
        self.de_std = de_std
        self.magnitude_mean = magnitude_mean
        self.magnitude_std = magnitude_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        de = (self.de[index].astype(np.float32) - self.de_mean) / self.de_std
        shape = self.shape[index].astype(np.float32)
        magnitude = (self.magnitude[index] - self.magnitude_mean) / self.magnitude_std
        return (
            torch.from_numpy(np.ascontiguousarray(de)),
            torch.from_numpy(np.ascontiguousarray(shape)),
            torch.from_numpy(np.ascontiguousarray(magnitude, dtype=np.float32)),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


class DualBranchModel(nn.Module):
    def __init__(self, args: argparse.Namespace, mode: str, hidden: int):
        super().__init__()
        self.mode = mode
        self.de_encoder = NoCLSFullGridAxialTransformer(
            (1, 1, 1, 1, 1), args.d_model, args.heads, args.layers_per_axis, args.ffn_ratio
        )
        self.de_encoder.classifier = nn.Identity()
        if mode == "de_e_fusion":
            self.second_encoder = ShapeMagnitudeAxialTransformer(
                args.d_model, args.heads, args.layers_per_axis, args.ffn_ratio, True
            )
            self.second_encoder.classifier = nn.Identity()
            representation_dim = 4 * args.d_model + 2 * 30 * 5
        elif mode == "de_de_control":
            self.second_encoder = NoCLSFullGridAxialTransformer(
                (1, 1, 1, 1, 1), args.d_model, args.heads, args.layers_per_axis, args.ffn_ratio
            )
            self.second_encoder.classifier = nn.Identity()
            representation_dim = 4 * args.d_model
        else:
            raise ValueError(mode)
        self.representation_dim = representation_dim
        self.hidden = hidden
        self.classifier = nn.Sequential(
            nn.Linear(representation_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, CLASSES, bias=False),
        )

    def forward(self, de: torch.Tensor, shape: torch.Tensor | None = None, magnitude: torch.Tensor | None = None):
        first = self.de_encoder.encode_representation(de)
        if self.mode == "de_e_fusion":
            if shape is None or magnitude is None:
                raise ValueError("DE+E requires shape and magnitude")
            second = self.second_encoder.encode_representation(shape, magnitude)
        else:
            second = self.second_encoder.encode_representation(de)
        return self.classifier(torch.cat((first, second), dim=-1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def matched_control_hidden(args: argparse.Namespace, target_parameters: int) -> int:
    best_hidden = 1
    best_difference = None
    for hidden in range(1, 2049):
        model = DualBranchModel(args, "de_de_control", hidden)
        difference = abs(parameter_count(model) - target_parameters)
        if best_difference is None or difference < best_difference:
            best_hidden = hidden
            best_difference = difference
        del model
    return best_hidden


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, device: torch.device, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def move_batch(batch: Sequence[torch.Tensor], device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor]:
    *inputs, labels = batch
    return [value.to(device, non_blocking=True) for value in inputs], labels.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    total = 0
    targets = []
    predictions = []
    for batch in loader:
        inputs, labels = move_batch(batch, device)
        logits = model(*inputs)
        loss_sum += float(criterion(logits, labels).cpu()) * len(labels)
        total += len(labels)
        targets.append(labels.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(CLASSES))
    recall = np.divide(
        np.diag(matrix), matrix.sum(1), out=np.zeros(CLASSES, dtype=np.float64), where=matrix.sum(1) > 0
    )
    return {
        "loss": loss_sum / total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_classes": int(np.unique(y_pred).size),
        "per_class_recall": recall.tolist(),
        "prediction_histogram": np.bincount(y_pred, minlength=CLASSES).tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def run_mode(
    args: argparse.Namespace,
    mode: str,
    fold: int,
    hidden: int,
    source_dataset: Dataset,
    target_dataset: Dataset,
    device: torch.device,
) -> dict[str, Any]:
    output = args.run_root / mode / f"fold-{fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(source_dataset, args.batch_size, True, args.seed + fold, device, args.num_workers)
    source_loader = make_loader(source_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers)
    target_loader = make_loader(target_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers)
    seed_everything(args.seed + fold)
    model = DualBranchModel(args, mode, hidden).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    print(f"\n[{mode} fold {fold:02d}] params={parameter_count(model):,} hidden={hidden}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            inputs, labels = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_correct += int((logits.argmax(1) == labels).sum().detach().cpu())
            train_total += len(labels)
        if epoch % args.eval_every == 0:
            source_metrics = evaluate(model, source_loader, device)
            target_metrics = evaluate(model, target_loader, device)
            for split, metrics in (("source", source_metrics), ("target", target_metrics)):
                history.append({"epoch": epoch, "split": split, **metrics})
                save_confusion(
                    metrics["confusion_matrix"],
                    f"{mode} fold {fold:02d} epoch {epoch:03d} {split}",
                    output / f"epoch-{epoch:03d}_{split}_confusion.png",
                )
            print(
                f"epoch {epoch:03d}/{args.epochs} train={train_correct / train_total:.3f} "
                f"source_f1={source_metrics['macro_f1']:.3f} target_f1={target_metrics['macro_f1']:.3f} "
                f"target_bacc={target_metrics['balanced_accuracy']:.3f} "
                f"classes={target_metrics['predicted_classes']} class9_recall={target_metrics['per_class_recall'][8]:.3f}"
            )
    summary = {
        "mode": mode,
        "fold": fold,
        "epochs": args.epochs,
        "hidden": hidden,
        "parameter_count": parameter_count(model),
        "source_trials": len(source_dataset),
        "target_trials": len(target_dataset),
        "final_source": source_metrics,
        "final_target": target_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {"model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}, **{
            key: summary[key] for key in ("mode", "fold", "epochs", "hidden", "parameter_count")
        }},
        output / "model_final.pt",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model, optimizer, train_loader, source_loader, target_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    args.four_second_cache = resolve_path(args.four_second_cache)
    args.processed_dir = resolve_path(args.processed_dir)
    args.run_root = resolve_path(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    if "de_e_fusion" in args.modes:
        build_four_second_cache(args.four_second_cache, args.processed_dir)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    one_second_cache, _ = find_complete_cache()

    probe = DualBranchModel(args, "de_e_fusion", args.fusion_hidden)
    fusion_parameters = parameter_count(probe)
    del probe
    control_hidden = matched_control_hidden(args, fusion_parameters)
    control_probe = DualBranchModel(args, "de_de_control", control_hidden)
    control_parameters = parameter_count(control_probe)
    del control_probe
    config = vars(args).copy()
    for key in ("four_second_cache", "processed_dir", "run_root"):
        config[key] = str(config[key])
    config.update({
        "device_resolved": str(device),
        "fusion_parameter_count": fusion_parameters,
        "control_hidden_matched": control_hidden,
        "control_parameter_count": control_parameters,
        "parameter_difference": control_parameters - fusion_parameters,
        "training": {
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "scheduler": None,
            "label_smoothing": 0.0,
            "early_stopping": None,
            "gradient_clipping": None,
            "amp": False,
        },
    })
    (args.run_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    for fold in args.folds:
        source_subjects, target_subjects = official_fold_subjects(fold)
        source_de, target_de, de_mean, de_std, _, _ = prepare_feature(
            "de", one_second_cache, source_subjects, target_subjects
        )
        if "de_de_control" in args.modes:
            results.append(run_mode(
                args,
                "de_de_control",
                fold,
                control_hidden,
                DEOnlyDataset(source_de, de_mean, de_std),
                DEOnlyDataset(target_de, de_mean, de_std),
                device,
            ))
        if "de_e_fusion" in args.modes:
            reference = fit_e_reference(args.four_second_cache, source_subjects)
            source_e = materialize_variant(args.four_second_cache, source_subjects, reference, VARIANTS["E"])
            target_e = materialize_variant(args.four_second_cache, target_subjects, reference, VARIANTS["E"])
            magnitude_mean = source_e["magnitude"].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            magnitude_std = source_e["magnitude"].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
            magnitude_std[magnitude_std < 1e-7] = 1.0
            results.append(run_mode(
                args,
                "de_e_fusion",
                fold,
                args.fusion_hidden,
                DEEDataset(source_de, source_e, de_mean, de_std, magnitude_mean, magnitude_std),
                DEEDataset(target_de, target_e, de_mean, de_std, magnitude_mean, magnitude_std),
                device,
            ))
            del source_e, target_e
        del source_de, target_de
        gc.collect()

    rows = []
    aggregate = {"epochs": args.epochs, "folds": args.folds, "results": {}}
    for mode in args.modes:
        selected = [item for item in results if item["mode"] == mode]
        row = {
            "mode": mode,
            "parameter_count": selected[0]["parameter_count"],
            "accuracy": float(np.mean([item["final_target"]["accuracy"] for item in selected])),
            "balanced_accuracy": float(np.mean([item["final_target"]["balanced_accuracy"] for item in selected])),
            "macro_f1": float(np.mean([item["final_target"]["macro_f1"] for item in selected])),
        }
        rows.append(row)
        aggregate["results"][mode] = row
    with (args.run_root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.run_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print("\n" + json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
