from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmrd.faced import EMOTION_NAMES, VIDEO_LABELS, VIDEOS, official_fold_subjects
from scripts.run_faced_rjsd_shape_magnitude_ablation import (
    BAND_NAMES,
    BAND_SIZES,
    CHANNELS,
    CLASSES,
    EPSILON,
    TIME_STEPS,
    ShapeMagnitudeAxialTransformer,
    load_probabilities,
    save_confusion,
)


@dataclass(frozen=True)
class ReferenceSpec:
    name: str
    mode: str
    support_trials: int
    kappa: float
    seed: int = 42


SPECS = {
    "global": ReferenceSpec("global", "global", 0, 0.0),
    "subject_k1": ReferenceSpec("subject_k1", "subject", 1, 0.0),
    "subject_k2": ReferenceSpec("subject_k2", "subject", 2, 0.0),
    "subject_k4": ReferenceSpec("subject_k4", "subject", 4, 0.0),
    "subject_k8": ReferenceSpec("subject_k8", "subject", 8, 0.0),
    "subject_k16": ReferenceSpec("subject_k16", "subject", 16, 0.0),
    "subject_k27": ReferenceSpec("subject_k27", "subject", 27, 0.0),
    "subject_k27_shrink4": ReferenceSpec("subject_k27_shrink4", "subject", 27, 4.0),
    "subject_k27_shrink16": ReferenceSpec("subject_k27_shrink16", "subject", 27, 16.0),
    "subject_k27_shrink64": ReferenceSpec("subject_k27_shrink64", "subject", 27, 64.0),
    "class_balanced_loto": ReferenceSpec("class_balanced_loto", "class_balanced", 27, 0.0),
    "class_balanced_shrink16": ReferenceSpec("class_balanced_shrink16", "class_balanced", 27, 16.0),
    "class_balanced_shrink64": ReferenceSpec("class_balanced_shrink64", "class_balanced", 27, 64.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACED subject-conditioned oracle reference diagnostics")
    parser.add_argument("--phase", choices=("screen", "deep"), required=True)
    parser.add_argument("--specs", nargs="+", choices=tuple(SPECS), default=tuple(SPECS))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--four-second-cache", type=Path, default=Path("runs/faced_4s_welch_probability_cache"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/faced_subject_reference_oracles"))
    args = parser.parse_args()
    if args.fold < 1 or args.fold > 10:
        parser.error("--fold must be in 1..10")
    if args.phase == "deep" and (args.epochs <= 0 or args.epochs % args.eval_every):
        parser.error("--epochs must be positive and divisible by --eval-every")
    return args


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_global_reference(cache_root: Path, subjects: Sequence[int]) -> list[np.ndarray]:
    sums = [np.zeros((CHANNELS, size), dtype=np.float64) for size in BAND_SIZES]
    count = 0
    for subject in subjects:
        for band, value in enumerate(load_probabilities(cache_root, int(subject))):
            sums[band] += value.sum(axis=(0, 1), dtype=np.float64)
        count += VIDEOS * TIME_STEPS
    output = []
    for value in sums:
        reference = value / count
        reference /= reference.sum(axis=-1, keepdims=True)
        output.append(reference.astype(np.float32))
    return output


def trial_references(
    probabilities: np.ndarray,
    global_reference: np.ndarray,
    labels: np.ndarray,
    subject: int,
    spec: ReferenceSpec,
) -> np.ndarray:
    if spec.mode == "global":
        return np.broadcast_to(global_reference, (VIDEOS,) + global_reference.shape).copy()
    trial_centers = probabilities.mean(axis=1, dtype=np.float32)
    references = []
    for query in range(VIDEOS):
        candidates = np.asarray([index for index in range(VIDEOS) if index != query], dtype=np.int64)
        if spec.mode == "subject":
            if spec.support_trials >= len(candidates):
                support = candidates
            else:
                rng = np.random.default_rng(spec.seed + subject * 10_000 + query * 101 + spec.support_trials)
                support = np.sort(rng.choice(candidates, size=spec.support_trials, replace=False))
            center = trial_centers[support].mean(axis=0, dtype=np.float32)
            evidence = len(support)
        elif spec.mode == "class_balanced":
            class_centers = []
            for label in range(CLASSES):
                support = candidates[labels[candidates] == label]
                if not len(support):
                    raise RuntimeError(f"No class {label} support after excluding query {query}")
                class_centers.append(trial_centers[support].mean(axis=0, dtype=np.float32))
            center = np.stack(class_centers).mean(axis=0, dtype=np.float32)
            evidence = len(candidates)
        else:
            raise ValueError(spec.mode)
        reference = (spec.kappa * global_reference + evidence * center) / (spec.kappa + evidence)
        reference /= reference.sum(axis=-1, keepdims=True)
        references.append(reference.astype(np.float32))
    return np.stack(references)


def materialize(
    cache_root: Path,
    subjects: Sequence[int],
    global_reference: Sequence[np.ndarray],
    spec: ReferenceSpec,
) -> dict[str, np.ndarray]:
    samples = len(subjects) * VIDEOS
    shape = np.zeros((samples, TIME_STEPS, CHANNELS, len(BAND_NAMES), max(BAND_SIZES)), dtype=np.float16)
    magnitude = np.zeros((samples, TIME_STEPS, CHANNELS, len(BAND_NAMES)), dtype=np.float32)
    labels = np.asarray(VIDEO_LABELS, dtype=np.int64)
    for subject_index, subject in enumerate(subjects):
        start = subject_index * VIDEOS
        for band, (p, q_global, size) in enumerate(zip(
            load_probabilities(cache_root, int(subject)), global_reference, BAND_SIZES, strict=True
        )):
            q = trial_references(p, q_global, labels, int(subject), spec)
            midpoint = 0.5 * (p + q[:, None, :, :])
            contribution = 0.5 * (
                p * (np.log(p + EPSILON) - np.log(midpoint + EPSILON))
                + q[:, None, :, :] * (np.log(q[:, None, :, :] + EPSILON) - np.log(midpoint + EPSILON))
            )
            contribution = np.maximum(contribution, 0.0).astype(np.float32)
            r = np.sqrt(contribution, dtype=np.float32)
            m = np.sqrt(np.square(r).sum(axis=-1), dtype=np.float32)
            signed_shape = np.sign(p - q[:, None, :, :]).astype(np.float32) * r / np.maximum(m[..., None], EPSILON)
            shape[start : start + VIDEOS, ..., band, :size] = signed_shape.astype(np.float16)
            magnitude[start : start + VIDEOS, ..., band] = m
    return {
        "shape": shape,
        "magnitude": magnitude,
        "labels": np.tile(labels, len(subjects)),
        "subjects": np.repeat(np.asarray(subjects, dtype=np.int64), VIDEOS),
        "videos": np.tile(np.arange(VIDEOS, dtype=np.int64), len(subjects)),
    }


def pooled_features(payload: dict[str, np.ndarray]) -> np.ndarray:
    shape = payload["shape"].astype(np.float32)
    magnitude = payload["magnitude"]
    parts = [
        shape.mean(axis=(1, 2)).reshape(len(shape), -1),
        shape.std(axis=(1, 2)).reshape(len(shape), -1),
        magnitude.mean(axis=1).reshape(len(shape), -1),
        magnitude.std(axis=1).reshape(len(shape), -1),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def standardize(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = source.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = source.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-7] = 1.0
    return (source - mean) / std, (target - mean) / std


def metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    recall = np.divide(
        np.diag(matrix), matrix.sum(1), out=np.zeros(len(labels), dtype=np.float64), where=matrix.sum(1) > 0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_classes": int(np.unique(y_pred).size),
        "per_class_recall": recall.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def screen_spec(
    args: argparse.Namespace,
    spec: ReferenceSpec,
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
) -> dict[str, Any]:
    source_features = pooled_features(source)
    target_features = pooled_features(target)
    source_features, target_features = standardize(source_features, target_features)
    emotion = LinearSVC(C=1.0, dual="auto", max_iter=20_000, random_state=args.seed)
    emotion.fit(source_features, source["labels"])
    target_prediction = emotion.predict(target_features)
    result = {
        "spec": asdict(spec),
        "feature_dimensions": int(source_features.shape[1]),
        "emotion_target": metrics(target["labels"], target_prediction, np.arange(CLASSES)),
        "source_subject_identity": None,
    }
    identity_specs = {"global", "subject_k27", "subject_k27_shrink16", "class_balanced_loto"}
    if spec.name in identity_specs:
        labels = np.asarray(VIDEO_LABELS)
        held_out_videos = np.asarray([np.flatnonzero(labels == label)[-1] for label in range(CLASSES)])
        train_mask = ~np.isin(source["videos"], held_out_videos)
        test_mask = ~train_mask
        magnitude_features = np.concatenate((
            source["magnitude"].mean(axis=1).reshape(len(source["labels"]), -1),
            source["magnitude"].std(axis=1).reshape(len(source["labels"]), -1),
        ), axis=1)
        identity_train, identity_test = standardize(magnitude_features[train_mask], magnitude_features[test_mask])
        identity = LinearSVC(C=1.0, dual="auto", max_iter=20_000, random_state=args.seed)
        identity.fit(identity_train, source["subjects"][train_mask])
        identity_prediction = identity.predict(identity_test)
        result["source_subject_identity"] = metrics(
            source["subjects"][test_mask], identity_prediction, np.unique(source["subjects"])
        )
        result["subject_probe_train_videos"] = np.flatnonzero(
            ~np.isin(np.arange(VIDEOS), held_out_videos)
        ).tolist()
        result["subject_probe_test_videos"] = held_out_videos.tolist()
    return result


class OracleDataset(Dataset):
    def __init__(self, payload: dict[str, np.ndarray], magnitude_mean: np.ndarray, magnitude_std: np.ndarray):
        self.shape = payload["shape"]
        self.magnitude = payload["magnitude"]
        self.labels = payload["labels"]
        self.magnitude_mean = magnitude_mean
        self.magnitude_std = magnitude_std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        shape = self.shape[index].astype(np.float32)
        magnitude = (self.magnitude[index] - self.magnitude_mean) / self.magnitude_std
        return (
            torch.from_numpy(np.ascontiguousarray(shape)),
            torch.from_numpy(np.ascontiguousarray(magnitude, dtype=np.float32)),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, device: torch.device, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    targets = []
    predictions = []
    for shape, magnitude, label in loader:
        shape = shape.to(device, non_blocking=True)
        magnitude = magnitude.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(shape, magnitude)
        total_loss += float(criterion(logits, label).cpu()) * len(label)
        total += len(label)
        targets.append(label.cpu().numpy())
        predictions.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {"loss": total_loss / total, **metrics(y_true, y_pred, np.arange(CLASSES))}


def run_deep(
    args: argparse.Namespace,
    spec: ReferenceSpec,
    source: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, Any]:
    output = args.run_root / "deep" / spec.name / f"fold-{args.fold:02d}"
    output.mkdir(parents=True, exist_ok=True)
    magnitude_mean = source["magnitude"].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    magnitude_std = source["magnitude"].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    magnitude_std[magnitude_std < 1e-7] = 1.0
    source_dataset = OracleDataset(source, magnitude_mean, magnitude_std)
    target_dataset = OracleDataset(target, magnitude_mean, magnitude_std)
    train_loader = make_loader(source_dataset, args.batch_size, True, args.seed + args.fold, device, args.num_workers)
    source_loader = make_loader(source_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers)
    target_loader = make_loader(target_dataset, args.eval_batch_size, False, args.seed, device, args.num_workers)
    seed_everything(args.seed + args.fold)
    model = ShapeMagnitudeAxialTransformer(
        args.d_model, args.heads, args.layers_per_axis, args.ffn_ratio, True
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    started = time.perf_counter()
    print(f"\n[{spec.name}] params={sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = 0
        count = 0
        for shape, magnitude, label in train_loader:
            shape = shape.to(device, non_blocking=True)
            magnitude = magnitude.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(shape, magnitude)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == label).sum().detach().cpu())
            count += len(label)
        if epoch % args.eval_every == 0:
            source_metrics = evaluate(model, source_loader, device)
            target_metrics = evaluate(model, target_loader, device)
            for split, current in (("source", source_metrics), ("target", target_metrics)):
                history.append({"epoch": epoch, "split": split, **current})
                save_confusion(
                    current["confusion_matrix"],
                    f"{spec.name} epoch {epoch} {split}",
                    output / f"epoch-{epoch:03d}_{split}_confusion.png",
                )
            print(
                f"epoch {epoch:03d}/{args.epochs} train={correct / count:.3f} "
                f"source_f1={source_metrics['macro_f1']:.3f} target_f1={target_metrics['macro_f1']:.3f} "
                f"target_bacc={target_metrics['balanced_accuracy']:.3f} classes={target_metrics['predicted_classes']}"
            )
    result = {
        "spec": asdict(spec),
        "fold": args.fold,
        "epochs": args.epochs,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_trials": len(source_dataset),
        "target_trials": len(target_dataset),
        "final_source": source_metrics,
        "final_target": target_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "spec": asdict(spec),
            "magnitude_mean": magnitude_mean,
            "magnitude_std": magnitude_std,
        },
        output / "model_final.pt",
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model, optimizer, source_dataset, target_dataset, train_loader, source_loader, target_loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    args.four_second_cache = resolve(args.four_second_cache)
    args.run_root = resolve(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    source_subjects, target_subjects = official_fold_subjects(args.fold)
    global_reference = fit_global_reference(args.four_second_cache, source_subjects)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    config = vars(args).copy()
    for key in ("four_second_cache", "run_root"):
        config[key] = str(config[key])
    config["device_resolved"] = str(device)
    config["training"] = {
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "dropout": 0.0,
        "weight_decay": 0.0,
        "scheduler": None,
        "label_smoothing": 0.0,
        "early_stopping": None,
        "gradient_clipping": None,
        "amp": False,
    }
    (args.run_root / f"config_{args.phase}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    for name in args.specs:
        spec = SPECS[name]
        started = time.perf_counter()
        source = materialize(args.four_second_cache, source_subjects, global_reference, spec)
        target = materialize(args.four_second_cache, target_subjects, global_reference, spec)
        if args.phase == "screen":
            result = screen_spec(args, spec, source, target)
            result["elapsed_seconds"] = time.perf_counter() - started
            print(
                f"[{name}] emotion_f1={result['emotion_target']['macro_f1']:.3f} "
                f"emotion_bacc={result['emotion_target']['balanced_accuracy']:.3f} "
                f"subject_acc={result['source_subject_identity']['accuracy']:.3f}"
                if result["source_subject_identity"] is not None else
                f"[{name}] emotion_f1={result['emotion_target']['macro_f1']:.3f} "
                f"emotion_bacc={result['emotion_target']['balanced_accuracy']:.3f} subject_acc=skipped"
            )
        else:
            result = run_deep(args, spec, source, target, device)
        results.append(result)
        del source, target
        gc.collect()

    output = args.run_root / f"{args.phase}_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.phase == "screen":
        rows = [{
            "spec": item["spec"]["name"],
            "mode": item["spec"]["mode"],
            "support_trials": item["spec"]["support_trials"],
            "kappa": item["spec"]["kappa"],
            "target_accuracy": item["emotion_target"]["accuracy"],
            "target_balanced_accuracy": item["emotion_target"]["balanced_accuracy"],
            "target_macro_f1": item["emotion_target"]["macro_f1"],
            "source_subject_accuracy": (
                item["source_subject_identity"]["accuracy"]
                if item["source_subject_identity"] is not None else None
            ),
        } for item in results]
        with (args.run_root / "screen_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
