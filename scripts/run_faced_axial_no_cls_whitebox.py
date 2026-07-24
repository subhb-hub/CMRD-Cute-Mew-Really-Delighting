from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmrd.faced import EEG_CHANNEL_NAMES, EMOTION_NAMES, SUBJECTS, VIDEO_LABELS, VIDEOS, official_fold_subjects
from cmrd.faced_psd_jsd_experiment import SpectraStore, fit_reference, fit_standardizer, materialize_split


BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
RJSD_BAND_SIZES = (3, 4, 6, 16, 17)
TIME_STEPS = 30
CHANNELS = len(EEG_CHANNEL_NAMES)
CLASSES = len(EMOTION_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FACED white-box axial Transformer: no CLS bottlenecks, no channel vote, "
            "balanced band/channel/time stages, plain fixed-LR training."
        )
    )
    parser.add_argument("--features", nargs="+", choices=("de", "rjsd"), default=("de", "rjsd"))
    parser.add_argument("--folds", nargs="+", type=int, default=tuple(range(1, 11)))
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs/faced_axial_no_cls_whitebox"),
    )
    args = parser.parse_args()
    if not args.features:
        parser.error("--features must not be empty")
    if not args.folds or any(fold < 1 or fold > 10 for fold in args.folds):
        parser.error("--folds must contain values from 1 to 10")
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds must not contain duplicates")
    if args.epochs <= 0 or args.eval_every <= 0 or args.epochs % args.eval_every:
        parser.error("--epochs must be positive and divisible by --eval-every")
    if args.d_model <= 0 or args.heads <= 0 or args.d_model % args.heads:
        parser.error("--d-model must be positive and divisible by --heads")
    if args.layers_per_axis <= 0 or args.ffn_ratio <= 0:
        parser.error("Transformer depths and FFN ratio must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def find_complete_cache() -> tuple[Path, dict[str, Any]]:
    base = ROOT / "runs" / "faced_native_compact_base_seed42" / "cache" / "native_spectra"
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for manifest_path in sorted(base.glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", [])) == SUBJECTS:
            candidates.append((manifest_path.parent, payload))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one complete FACED native cache, found {len(candidates)}")
    root, manifest = candidates[0]
    if tuple(manifest.get("band_names", ())) != BAND_NAMES:
        raise ValueError(f"Unexpected band names: {manifest.get('band_names')}")
    if tuple(manifest.get("band_sizes", ())) != RJSD_BAND_SIZES:
        raise ValueError(f"Unexpected native band sizes: {manifest.get('band_sizes')}")
    return root, manifest


class StructuredFeatureDataset(Dataset):
    def __init__(self, values: np.ndarray, labels: np.ndarray, mean: np.ndarray, std: np.ndarray):
        if values.ndim != 5:
            raise ValueError(f"Expected [N,T,C,B,F], got {values.shape}")
        if values.shape[1:4] != (TIME_STEPS, CHANNELS, len(BAND_NAMES)):
            raise ValueError(f"Unexpected feature structure: {values.shape}")
        if len(values) != len(labels):
            raise ValueError("Feature/label length mismatch")
        self.values = values
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.values[index].astype(np.float32)
        value = (value - self.mean) / self.std
        return torch.from_numpy(np.ascontiguousarray(value)), torch.tensor(self.labels[index], dtype=torch.long)


def fit_plain_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(values.shape[2:], dtype=np.float64)
    square = np.zeros_like(total)
    count = 0
    for start in range(0, len(values), 64):
        chunk = values[start : start + 64].astype(np.float32)
        total += chunk.sum(axis=(0, 1), dtype=np.float64)
        square += np.square(chunk).sum(axis=(0, 1), dtype=np.float64)
        count += chunk.shape[0] * chunk.shape[1]
    mean = total / count
    variance = np.maximum(square / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-7] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def load_de(cache_root: Path, subjects: Sequence[int]) -> dict[str, Any]:
    feature_parts = []
    for subject in subjects:
        path = cache_root / "subjects" / f"sub{int(subject):03d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            de = np.asarray(archive["de"], dtype=np.float32)
        expected = (VIDEOS, TIME_STEPS, CHANNELS * len(BAND_NAMES))
        if de.shape != expected or not np.isfinite(de).all():
            raise ValueError(f"Invalid DE for subject {subject}: {de.shape}")
        feature_parts.append(
            de.reshape(VIDEOS, TIME_STEPS, CHANNELS, len(BAND_NAMES), 1)
        )
    return {
        "x": np.concatenate(feature_parts),
        "y": np.tile(VIDEO_LABELS, len(subjects)).astype(np.int64),
        "subjects": np.repeat(np.asarray(subjects, dtype=np.int64), VIDEOS),
    }


def prepare_feature(
    feature: str,
    cache_root: Path,
    source_subjects: Sequence[int],
    target_subjects: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, tuple[int, ...], dict[str, Any]]:
    if feature == "de":
        source = load_de(cache_root, source_subjects)
        target = load_de(cache_root, target_subjects)
        mean, std = fit_plain_standardizer(source["x"])
        band_sizes = (1,) * len(BAND_NAMES)
        audit = {
            "feature": "five_band_differential_entropy",
            "reference": "none",
            "standardizer_scope": "all source subjects only",
        }
        return source, target, mean, std, band_sizes, audit

    if feature == "rjsd":
        store = SpectraStore(cache_root)
        reference, reference_windows = fit_reference(store, source_subjects)
        source = materialize_split(store, source_subjects, reference, 1e-12, "float16")
        target = materialize_split(store, target_subjects, reference, 1e-12, "float16")
        mean, std = fit_standardizer(source["x"])
        audit = {
            "feature": "frequency_resolved_sqrt_rjsd",
            "reference": "arithmetic PSD probability mean over all source subjects only",
            "reference_windows": reference_windows,
            "maximum_sqrt_reconstruction_error": max(
                float(source["maximum_invariant_error"]),
                float(target["maximum_invariant_error"]),
            ),
            "standardizer_scope": "all source subjects only",
        }
        return source, target, mean, std, RJSD_BAND_SIZES, audit

    raise ValueError(feature)


class PlainTransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_ratio: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio, bias=False),
            nn.GELU(),
            nn.Linear(d_model * ffn_ratio, d_model, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        value = value + attended
        return value + self.ffn(self.norm2(value))


class NoCLSFullGridAxialTransformer(nn.Module):
    """Keep every time/channel/band token through all three Transformer axes."""

    def __init__(
        self,
        band_sizes: Sequence[int],
        d_model: int,
        heads: int,
        layers_per_axis: int,
        ffn_ratio: int,
    ):
        super().__init__()
        self.band_sizes = tuple(map(int, band_sizes))
        self.input_encoders = nn.ModuleList(
            nn.Linear(size, d_model, bias=False) for size in self.band_sizes
        )
        self.band_embedding = nn.Parameter(torch.zeros(1, 1, 1, len(self.band_sizes), d_model))
        self.channel_embedding = nn.Parameter(torch.zeros(1, 1, CHANNELS, 1, d_model))
        self.time_embedding = nn.Parameter(torch.zeros(1, TIME_STEPS, 1, 1, d_model))
        self.band_blocks = nn.ModuleList(
            PlainTransformerBlock(d_model, heads, ffn_ratio) for _ in range(layers_per_axis)
        )
        self.channel_blocks = nn.ModuleList(
            PlainTransformerBlock(d_model, heads, ffn_ratio) for _ in range(layers_per_axis)
        )
        self.time_blocks = nn.ModuleList(
            PlainTransformerBlock(d_model, heads, ffn_ratio) for _ in range(layers_per_axis)
        )
        self.output_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.classifier = nn.Linear(2 * d_model, CLASSES, bias=False)
        nn.init.trunc_normal_(self.band_embedding, std=0.02)
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def encode_representation(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"Expected [B,T,C,B,F], got {tuple(value.shape)}")
        if value.shape[1:4] != (TIME_STEPS, CHANNELS, len(self.band_sizes)):
            raise ValueError(f"Unexpected input grid: {tuple(value.shape)}")
        encoded = [
            encoder(value[..., band, :size])
            for band, (size, encoder) in enumerate(zip(self.band_sizes, self.input_encoders, strict=True))
        ]
        tokens = torch.stack(encoded, dim=3)
        tokens = tokens + self.band_embedding + self.channel_embedding + self.time_embedding
        batch, time_steps, channels, bands, d_model = tokens.shape

        band_sequences = tokens.reshape(batch * time_steps * channels, bands, d_model)
        for block in self.band_blocks:
            band_sequences = block(band_sequences)
        tokens = band_sequences.reshape(batch, time_steps, channels, bands, d_model)

        channel_sequences = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch * time_steps * bands, channels, d_model
        )
        for block in self.channel_blocks:
            channel_sequences = block(channel_sequences)
        tokens = channel_sequences.reshape(batch, time_steps, bands, channels, d_model).permute(
            0, 1, 3, 2, 4
        )

        time_sequences = tokens.permute(0, 2, 3, 1, 4).reshape(
            batch * channels * bands, time_steps, d_model
        )
        for block in self.time_blocks:
            time_sequences = block(time_sequences)
        tokens = time_sequences.reshape(batch, channels, bands, time_steps, d_model).permute(
            0, 3, 1, 2, 4
        )

        tokens = self.output_norm(tokens)
        mean = tokens.mean(dim=(1, 2, 3))
        std = torch.sqrt(tokens.var(dim=(1, 2, 3), unbiased=False).clamp_min(1e-6))
        return torch.cat([mean, std], dim=-1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_representation(value))


def parameter_report(model: NoCLSFullGridAxialTransformer) -> dict[str, int]:
    groups = {
        "input_and_embeddings": 0,
        "band_transformer": 0,
        "channel_transformer": 0,
        "time_transformer": 0,
        "classifier": 0,
    }
    for name, parameter in model.named_parameters():
        if name.startswith(("input_encoders", "band_embedding", "channel_embedding", "time_embedding")):
            group = "input_and_embeddings"
        elif name.startswith("band_blocks"):
            group = "band_transformer"
        elif name.startswith("channel_blocks"):
            group = "channel_transformer"
        elif name.startswith("time_blocks"):
            group = "time_transformer"
        else:
            group = "classifier"
        groups[group] += parameter.numel()
    groups["total"] = sum(parameter.numel() for parameter in model.parameters())
    return groups


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    targets = []
    predictions = []
    for value, label in loader:
        value = value.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(value)
        total_loss += float(criterion(logits, label).cpu()) * len(label)
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
    metrics = {
        "loss": total_loss / total,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_classes": int(np.unique(y_pred).size),
        "prediction_histogram": np.bincount(y_pred, minlength=CLASSES).tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
    return metrics, y_true, y_pred


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
    threshold = values.max(initial=0) / 2
    for row in range(CLASSES):
        for column in range(CLASSES):
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def append_metrics(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = (
        "feature",
        "fold",
        "epoch",
        "split",
        "loss",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "predicted_classes",
    )
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_one(
    args: argparse.Namespace,
    feature: str,
    fold: int,
    cache_root: Path,
    device: torch.device,
    metrics_path: Path,
) -> dict[str, Any]:
    source_subjects, target_subjects = official_fold_subjects(fold)
    output = args.run_root / feature / f"fold-{fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    source, target, mean, std, band_sizes, feature_audit = prepare_feature(
        feature, cache_root, source_subjects, target_subjects
    )
    audit = {
        **feature_audit,
        "fold": fold,
        "source_subjects": list(map(int, source_subjects)),
        "target_subjects": list(map(int, target_subjects)),
        "source_trials": int(len(source["y"])),
        "target_trials": int(len(target["y"])),
        "axis_order": ["time", "channel", "band", "native_frequency"],
        "band_sizes": list(band_sizes),
        "target_used_for_gradients": False,
        "target_monitored_every": args.eval_every,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    source_dataset = StructuredFeatureDataset(source["x"], source["y"], mean, std)
    target_dataset = StructuredFeatureDataset(target["x"], target["y"], mean, std)
    source_loader = make_loader(
        source_dataset, args.batch_size, True, args.seed + fold, device, args.num_workers
    )
    source_eval_loader = make_loader(
        source_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers
    )
    target_loader = make_loader(
        target_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers
    )

    seed_everything(args.seed + fold)
    model = NoCLSFullGridAxialTransformer(
        band_sizes,
        args.d_model,
        args.heads,
        args.layers_per_axis,
        args.ffn_ratio,
    ).to(device)
    report = parameter_report(model)
    stage_counts = [report[name] for name in ("band_transformer", "channel_transformer", "time_transformer")]
    if len(set(stage_counts)) != 1:
        raise AssertionError(f"Transformer stages are not parameter-balanced: {report}")
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    final_target_true = None
    final_target_pred = None

    print(
        f"\n[{feature.upper()} fold {fold:02d}] source={len(source_subjects)} subjects "
        f"target={len(target_subjects)} subjects params={report['total']:,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for value, label in source_loader:
            value = value.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(value)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(label)
            train_correct += int((logits.argmax(1) == label).sum().detach().cpu())
            train_count += len(label)

        if epoch % args.eval_every == 0:
            source_metrics, _, _ = evaluate(model, source_eval_loader, device)
            target_metrics, target_true, target_pred = evaluate(model, target_loader, device)
            if epoch == args.epochs:
                final_target_true = target_true
                final_target_pred = target_pred
            rows = []
            for split_name, metrics in (("source", source_metrics), ("target", target_metrics)):
                row = {
                    "feature": feature,
                    "fold": fold,
                    "epoch": epoch,
                    "split": split_name,
                    **{name: metrics[name] for name in (
                        "loss", "accuracy", "balanced_accuracy", "macro_f1", "predicted_classes"
                    )},
                }
                rows.append(row)
                history.append({
                    **row,
                    "per_class_recall": metrics["per_class_recall"],
                    "prediction_histogram": metrics["prediction_histogram"],
                })
                save_confusion(
                    metrics["confusion_matrix"],
                    f"{feature.upper()} fold {fold:02d} epoch {epoch:03d} {split_name}",
                    output / f"epoch-{epoch:03d}_{split_name}_confusion.png",
                )
            append_metrics(metrics_path, rows)
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train={train_correct / train_count:.3f} "
                f"source_f1={source_metrics['macro_f1']:.3f} "
                f"source_bacc={source_metrics['balanced_accuracy']:.3f} "
                f"target_f1={target_metrics['macro_f1']:.3f} "
                f"target_bacc={target_metrics['balanced_accuracy']:.3f} "
                f"target_classes={target_metrics['predicted_classes']} "
                f"class9_recall={target_metrics['per_class_recall'][8]:.3f}"
            )

    if final_target_true is None or final_target_pred is None:
        raise RuntimeError("Final target monitoring result is missing")
    torch.save(
        {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "feature": feature,
            "fold": fold,
            "epoch": args.epochs,
            "band_sizes": band_sizes,
            "source_mean": mean,
            "source_std": std,
            "model_config": {
                "d_model": args.d_model,
                "heads": args.heads,
                "layers_per_axis": args.layers_per_axis,
                "ffn_ratio": args.ffn_ratio,
            },
        },
        output / "model_final_fixed_epoch.pt",
    )
    summary = {
        "feature": feature,
        "fold": fold,
        "epochs": args.epochs,
        "parameter_report": report,
        "final_source": source_metrics,
        "final_target": target_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["final_target_true"] = final_target_true
    summary["final_target_pred"] = final_target_pred

    del model, optimizer, source_dataset, target_dataset, source_loader, source_eval_loader, target_loader
    del source, target
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
    args.run_root.mkdir(parents=True, exist_ok=True)
    cache_root, cache_manifest = find_complete_cache()
    device = resolve_device(args.device)
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = vars(args).copy()
    config["run_root"] = str(args.run_root.resolve())
    config["device_resolved"] = str(device)
    config["cache_root"] = str(cache_root.resolve())
    config["cache_signature"] = cache_manifest.get("signature")
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
    (args.run_root / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path = args.run_root / "metrics_every_n_epochs.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    results = []
    for feature in args.features:
        for fold in args.folds:
            results.append(run_one(args, feature, fold, cache_root, device, metrics_path))

    aggregate: dict[str, Any] = {
        "features": list(args.features),
        "folds": list(args.folds),
        "epochs": args.epochs,
        "results": {},
    }
    for feature in args.features:
        selected = [result for result in results if result["feature"] == feature]
        true = np.concatenate([result.pop("final_target_true") for result in selected])
        pred = np.concatenate([result.pop("final_target_pred") for result in selected])
        target_rows = [result["final_target"] for result in selected]
        aggregate["results"][feature] = {
            "fold_mean": {
                metric: float(np.mean([row[metric] for row in target_rows]))
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            },
            "fold_std": {
                metric: float(np.std([row[metric] for row in target_rows]))
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
            },
            "pooled": {
                "accuracy": float(accuracy_score(true, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
                "predicted_classes": int(np.unique(pred).size),
                "confusion_matrix": confusion_matrix(true, pred, labels=np.arange(CLASSES)).tolist(),
            },
        }
        save_confusion(
            aggregate["results"][feature]["pooled"]["confusion_matrix"],
            f"{feature.upper()} pooled fixed-epoch target",
            args.run_root / f"{feature}_pooled_target_confusion.png",
        )
    (args.run_root / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
