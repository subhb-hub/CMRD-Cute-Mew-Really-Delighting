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
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset

from .faced import EMOTION_NAMES, SUBJECTS, VIDEO_LABELS, VIDEOS, official_fold_subjects
from .models.subject_adversarial import SubjectAdversarialHCBT
from .training.metrics import classification_metrics
from .training.runtime import seed_everything, select_device


LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AdversarialConfig:
    path: Path
    base_run_root: Path
    run_root: Path
    fold: int
    seed: int
    development_subjects: tuple[int, ...]
    conditions: tuple[str, ...]
    protocols: tuple[str, ...]
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


def load_config(path: str | Path) -> AdversarialConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = config_path.parents[2]
    protocol = raw["protocol"]
    config = AdversarialConfig(
        path=config_path,
        base_run_root=_resolve(repo_root, raw["paths"]["base_run_root"]),
        run_root=_resolve(repo_root, raw["paths"]["run_root"]),
        fold=int(protocol["fold"]),
        seed=int(protocol["seed"]),
        development_subjects=tuple(map(int, protocol["development_subjects"])),
        conditions=tuple(map(str, protocol["conditions"])),
        protocols=tuple(map(str, protocol["evaluation_protocols"])),
        model=dict(raw["model"]),
        training=dict(raw["training"]),
        evidence_label=str(raw["experiment"]["evidence_label"]),
    )
    source, target = official_fold_subjects(config.fold)
    if not set(config.development_subjects) <= set(source) or set(config.development_subjects) & set(target):
        raise ValueError("Development subjects must be outer-fold sources and target-disjoint")
    if any(condition not in SubjectAdversarialHCBT.CONDITIONS for condition in config.conditions):
        raise ValueError("Config contains an unknown adversarial condition")
    allowed_protocols = {"conventional_subject_holdout", "subject_and_stimulus_holdout"}
    if not set(config.protocols) <= allowed_protocols:
        raise ValueError("Config contains an unknown evaluation protocol")
    return config


def stimulus_split() -> dict[str, np.ndarray]:
    """Create disjoint train/dev/test videos inside every emotion class."""
    train: list[int] = []
    development: list[int] = []
    test: list[int] = []
    for label in range(len(EMOTION_NAMES)):
        videos = np.flatnonzero(VIDEO_LABELS == label).tolist()
        if len(videos) < 3:
            raise ValueError("Every FACED emotion needs at least three videos for stimulus isolation")
        train.extend(videos[:-2])
        development.append(videos[-2])
        test.append(videos[-1])
    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "development": np.asarray(development, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
    }


def protocol_videos(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "conventional_subject_holdout":
        all_videos = np.arange(VIDEOS, dtype=np.int64)
        return all_videos, all_videos, all_videos
    if name == "subject_and_stimulus_holdout":
        split = stimulus_split()
        return split["train"], split["development"], split["test"]
    raise KeyError(name)


def _spectra_root(config: AdversarialConfig) -> Path:
    candidates: list[Path] = []
    for manifest in sorted((config.base_run_root / "cache" / "native_spectra").glob("*/manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("all_subjects_complete") and len(payload.get("subjects_complete", [])) == SUBJECTS:
            candidates.append(manifest.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete FACED DE cache, found {candidates}")
    return candidates[0]


def load_examples(
    config: AdversarialConfig,
    subjects: Sequence[int],
    videos: Sequence[int],
    subject_map: dict[int, int] | None,
) -> list[Example]:
    root = _spectra_root(config) / "subjects"
    selected_videos = list(map(int, videos))
    examples: list[Example] = []
    for subject in map(int, subjects):
        with np.load(root / f"sub{subject:03d}.npz", allow_pickle=False) as archive:
            de = np.asarray(archive["de"], dtype=np.float32)
        if de.shape != (VIDEOS, 30, 150) or not np.isfinite(de).all():
            raise ValueError(f"Invalid FACED DE cache for sub{subject:03d}: {de.shape}")
        for video in selected_videos:
            examples.append(Example(
                x=np.ascontiguousarray(de[video]),
                label=int(VIDEO_LABELS[video]),
                raw_subject=subject,
                local_subject=int(subject_map[subject]) if subject_map is not None else -1,
                video=video,
            ))
    return examples


def fit_normalizer(examples: Sequence[Example]) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        raise ValueError("Cannot fit a normalizer without source-training examples")
    values = np.concatenate([example.x.astype(np.float64) for example in examples], axis=0)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class AdversarialDataset(Dataset):
    def __init__(self, examples: Sequence[Example], mean: np.ndarray, std: np.ndarray) -> None:
        self.data = torch.from_numpy(np.stack([
            np.ascontiguousarray((example.x - mean) / std, dtype=np.float32)
            for example in examples
        ]))
        self.labels = torch.as_tensor([example.label for example in examples], dtype=torch.long)
        self.local_subjects = torch.as_tensor(
            [example.local_subject for example in examples], dtype=torch.long
        )
        self.raw_subjects = np.asarray([example.raw_subject for example in examples], dtype=np.int64)
        self.videos = np.asarray([example.video for example in examples], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index], self.local_subjects[index]


class SubjectEmotionBatchSampler(BatchSampler):
    """Deterministic batches sharing emotions across several source subjects."""

    def __init__(
        self,
        dataset: AdversarialDataset,
        subjects_per_batch: int,
        emotions_per_subject: int,
        seed: int,
    ) -> None:
        if subjects_per_batch < 2 or emotions_per_subject < 2:
            raise ValueError("Balanced batches require multiple subjects and emotions")
        subjects = np.unique(dataset.local_subjects.numpy())
        if np.any(subjects < 0):
            raise ValueError("Training dataset needs local source-subject IDs")
        if subjects_per_batch > len(subjects) or emotions_per_subject > len(EMOTION_NAMES):
            raise ValueError("Balanced batch dimensions exceed available subjects/classes")
        self.dataset = dataset
        self.subjects = subjects
        self.subjects_per_batch = int(subjects_per_batch)
        self.emotions_per_subject = int(emotions_per_subject)
        self.seed = int(seed)
        self.epoch = 0
        self.batches = math.ceil(len(dataset) / (subjects_per_batch * emotions_per_subject))
        labels = dataset.labels.numpy()
        local = dataset.local_subjects.numpy()
        self.groups: dict[tuple[int, int], np.ndarray] = {}
        for subject in subjects:
            for label in range(len(EMOTION_NAMES)):
                indices = np.flatnonzero((local == subject) & (labels == label))
                if not len(indices):
                    raise ValueError(f"Missing source subject/emotion pair: {subject}/{label}")
                self.groups[(int(subject), label)] = indices

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.batches):
            subjects = rng.choice(self.subjects, self.subjects_per_batch, replace=False)
            labels = rng.choice(len(EMOTION_NAMES), self.emotions_per_subject, replace=False)
            batch = [
                int(rng.choice(self.groups[(int(subject), int(label))]))
                for subject in subjects for label in labels
            ]
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches


def adversarial_weight(epoch: int, total_epochs: int, warmup_epochs: int, maximum: float) -> float:
    if epoch <= warmup_epochs or maximum <= 0:
        return 0.0
    denominator = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / denominator, 0.0), 1.0)
    return float(maximum * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0))


def _valid_indices(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    return mask.reshape(-1).nonzero(as_tuple=False).squeeze(1).to(device, non_blocking=True)


def _autocast(config: AdversarialConfig, device: torch.device):
    precision = str(config.training.get("precision", "float32"))
    if device.type != "cuda" or precision not in {"float16", "bfloat16"}:
        return nullcontext()
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _loader(dataset: AdversarialDataset, batch_size: int = 128) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate_emotion(
    model: SubjectAdversarialHCBT,
    dataset: AdversarialDataset,
    device: torch.device,
    config: AdversarialConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for data, labels, _ in _loader(dataset):
        mask = torch.ones(data.shape[:2], dtype=torch.bool)
        valid = _valid_indices(mask, device)
        data = data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with _autocast(config, device):
            emotion, _ = model.encode(data, mask, valid)
            logits = model.emotion_classifier(emotion)
        targets.append(labels.numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    y = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    return classification_metrics(y, prediction, len(EMOTION_NAMES)), y, prediction


@torch.no_grad()
def extract_embeddings(
    model: SubjectAdversarialHCBT,
    dataset: AdversarialDataset,
    device: torch.device,
    config: AdversarialConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    emotion_values: list[np.ndarray] = []
    subject_values: list[np.ndarray] = []
    has_subject = model.subject_encoder is not None
    for data, _, _ in _loader(dataset):
        mask = torch.ones(data.shape[:2], dtype=torch.bool)
        valid = _valid_indices(mask, device)
        data = data.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with _autocast(config, device):
            emotion, subject = model.encode(data, mask, valid)
        emotion_values.append(emotion.float().cpu().numpy())
        if subject is not None:
            subject_values.append(subject.float().cpu().numpy())
    return np.concatenate(emotion_values), np.concatenate(subject_values) if has_subject else None


def independent_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    c: float,
) -> float:
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(c), class_weight="balanced", solver="lbfgs", max_iter=2_000,
            random_state=42,
        ),
    )
    estimator.fit(train_x, train_y)
    return float(accuracy_score(test_y, estimator.predict(test_x)))


def probe_embeddings(
    model: SubjectAdversarialHCBT,
    config: AdversarialConfig,
    mean: np.ndarray,
    std: np.ndarray,
    fit_subjects: Sequence[int],
    dev_subjects: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    # These source-only probes are trained only after the emotion checkpoint is
    # locked. Their labels never affect the encoder or checkpoint selection.
    source_map = {subject: index for index, subject in enumerate(fit_subjects)}
    fit_all = AdversarialDataset(
        load_examples(config, fit_subjects, np.arange(VIDEOS), source_map), mean, std
    )
    dev_all = AdversarialDataset(
        load_examples(config, dev_subjects, np.arange(VIDEOS), None), mean, std
    )
    fit_ze, _ = extract_embeddings(model, fit_all, device, config)
    dev_ze, dev_zs = extract_embeddings(model, dev_all, device, config)
    test_videos = stimulus_split()["test"]
    subject_train_mask = ~np.isin(dev_all.videos, test_videos)
    subject_test_mask = np.isin(dev_all.videos, test_videos)
    c = float(config.training.get("probe_c", 1.0))
    result = {
        "subject_id_probe_on_ze": independent_probe(
            dev_ze[subject_train_mask], dev_all.raw_subjects[subject_train_mask],
            dev_ze[subject_test_mask], dev_all.raw_subjects[subject_test_mask], c,
        ),
        "video_id_probe_on_ze": independent_probe(
            fit_ze, fit_all.videos, dev_ze, dev_all.videos, c,
        ),
        "subject_probe_train_samples": int(subject_train_mask.sum()),
        "subject_probe_test_samples": int(subject_test_mask.sum()),
        "video_probe_train_samples": int(len(fit_all)),
        "video_probe_test_samples": int(len(dev_all)),
    }
    if dev_zs is not None:
        result["subject_id_probe_on_zs"] = independent_probe(
            dev_zs[subject_train_mask], dev_all.raw_subjects[subject_train_mask],
            dev_zs[subject_test_mask], dev_all.raw_subjects[subject_test_mask], c,
        )
    del fit_all, dev_all, fit_ze, dev_ze, dev_zs
    gc.collect()
    return result


def bootstrap_subject_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    subjects: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects)
    accuracy: list[float] = []
    macro_f1: list[float] = []
    for _ in range(int(repeats)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
        accuracy.append(float(accuracy_score(y[indices], prediction[indices])))
        macro_f1.append(float(f1_score(y[indices], prediction[indices], average="macro", zero_division=0)))
    return {
        "accuracy_ci95": [float(value) for value in np.quantile(accuracy, [0.025, 0.975])],
        "macro_f1_ci95": [float(value) for value in np.quantile(macro_f1, [0.025, 0.975])],
    }


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


def protocol_hash(config: AdversarialConfig) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fold": config.fold,
        "seed": config.seed,
        "development_subjects": config.development_subjects,
        "conditions": config.conditions,
        "protocols": config.protocols,
        "model": config.model,
        "training": config.training,
        "stimulus_split": {key: value.tolist() for key, value in stimulus_split().items()},
        "target_policy": "load only after source-development checkpoint lock",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _task_root(run_root: Path, protocol: str, condition: str) -> Path:
    return run_root / protocol / condition / "fold-01" / "seed-42"


def paired_target_bootstrap(config: AdversarialConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    repeats = int(config.training["bootstrap_repeats"])
    for protocol in config.protocols:
        records: dict[str, list[dict[str, int]]] = {}
        for condition in config.conditions:
            path = _task_root(config.run_root, protocol, condition) / "target_predictions.csv"
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = [
                    {key: int(row[key]) for key in ("subject", "video", "target", "prediction")}
                    for row in csv.DictReader(stream)
                ]
            rows.sort(key=lambda row: (row["subject"], row["video"]))
            records[condition] = rows
        base_rows = records["B0_emotion_only"]
        keys = [(row["subject"], row["video"], row["target"]) for row in base_rows]
        for condition, rows in records.items():
            other = [(row["subject"], row["video"], row["target"]) for row in rows]
            if other != keys:
                raise ValueError(f"Paired target rows do not align for {protocol}/{condition}")
        subjects = np.asarray([row["subject"] for row in base_rows], dtype=np.int64)
        y = np.asarray([row["target"] for row in base_rows], dtype=np.int64)
        predictions = {
            condition: np.asarray([row["prediction"] for row in rows], dtype=np.int64)
            for condition, rows in records.items()
        }
        unique = np.unique(subjects)
        rng = np.random.default_rng(config.seed)
        by_condition: dict[str, Any] = {}
        for condition in config.conditions[1:]:
            accuracy_differences: list[float] = []
            f1_differences: list[float] = []
            for _ in range(repeats):
                sampled = rng.choice(unique, size=len(unique), replace=True)
                indices = np.concatenate([np.flatnonzero(subjects == subject) for subject in sampled])
                base_prediction = predictions["B0_emotion_only"][indices]
                candidate_prediction = predictions[condition][indices]
                accuracy_differences.append(
                    float(accuracy_score(y[indices], candidate_prediction) - accuracy_score(y[indices], base_prediction))
                )
                f1_differences.append(float(
                    f1_score(y[indices], candidate_prediction, average="macro", zero_division=0)
                    - f1_score(y[indices], base_prediction, average="macro", zero_division=0)
                ))
            point_accuracy = float(
                accuracy_score(y, predictions[condition])
                - accuracy_score(y, predictions["B0_emotion_only"])
            )
            point_f1 = float(
                f1_score(y, predictions[condition], average="macro", zero_division=0)
                - f1_score(y, predictions["B0_emotion_only"], average="macro", zero_division=0)
            )
            by_condition[condition] = {
                "accuracy_difference": point_accuracy,
                "accuracy_difference_ci95": [
                    float(value) for value in np.quantile(accuracy_differences, [0.025, 0.975])
                ],
                "accuracy_probability_greater_than_B0": float(
                    np.mean(np.asarray(accuracy_differences) > 0)
                ),
                "macro_f1_difference": point_f1,
                "macro_f1_difference_ci95": [
                    float(value) for value in np.quantile(f1_differences, [0.025, 0.975])
                ],
                "macro_f1_probability_greater_than_B0": float(
                    np.mean(np.asarray(f1_differences) > 0)
                ),
            }
        result[protocol] = by_condition
    return result


def train_task(
    config: AdversarialConfig,
    protocol: str,
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
    root = run_root or config.run_root
    output = _task_root(root, protocol, condition)
    result_path = output / "result.json"
    if result_path.is_file() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
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
    LOGGER.info("%s %s: loading source fit/development only", protocol, condition)
    train_examples = load_examples(config, fit_subjects, train_videos, subject_map)
    dev_examples = load_examples(config, dev_subjects, dev_videos, None)
    mean, std = fit_normalizer(train_examples)
    train_dataset = AdversarialDataset(train_examples, mean, std)
    dev_dataset = AdversarialDataset(dev_examples, mean, std)
    sampler = SubjectEmotionBatchSampler(
        train_dataset,
        int(config.training["batch_subjects"]),
        int(config.training["batch_emotions"]),
        config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = SubjectAdversarialHCBT(
        condition=condition,
        input_dim=150,
        channels=int(config.model["channels"]),
        classes=int(config.model["classes"]),
        source_subjects=len(fit_subjects),
        max_length=30,
        d_model=int(config.model["d_model"]),
        heads=int(config.model["heads"]),
        layers=int(config.model["layers"]),
        feedforward=int(config.model["feedforward"]),
        dropout=float(config.model["dropout"]),
        subject_hidden=int(config.model["subject_hidden"]),
        subject_dim=int(config.model["subject_dim"]),
    ).to(device)
    adversarial_parameters = (
        list(model.adversarial_subject_classifier.parameters())
        if model.adversarial_subject_classifier is not None else []
    )
    positive_parameters = (
        list(model.positive_subject_classifier.parameters())
        if model.positive_subject_classifier is not None else []
    )
    auxiliary_ids = {id(parameter) for parameter in adversarial_parameters + positive_parameters}
    core_parameters_for_optimizer = [
        parameter for parameter in model.parameters() if id(parameter) not in auxiliary_ids
    ]
    parameter_groups: list[dict[str, Any]] = [{
        "params": core_parameters_for_optimizer,
        "lr": float(config.training["learning_rate"]),
        "group_name": "encoder_emotion",
    }]
    subject_learning_rate = float(
        config.training.get("subject_learning_rate", config.training["learning_rate"])
    )
    if adversarial_parameters:
        parameter_groups.append({
            "params": adversarial_parameters,
            "lr": subject_learning_rate,
            "group_name": "adversarial_subject",
        })
    if positive_parameters:
        parameter_groups.append({
            "params": positive_parameters,
            "lr": subject_learning_rate,
            "group_name": "positive_subject",
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config.training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(config.training["minimum_learning_rate"]),
    )
    emotion_criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.training.get("label_smoothing", 0.0))
    )
    subject_criterion = nn.CrossEntropyLoss()
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
    warmup = int(config.training["adversarial_warmup_epochs"])
    selection_minimum = (
        warmup + 1
        if condition != "B0_emotion_only" and epochs > warmup
        else 1
    )
    for epoch in range(1, epochs + 1):
        model.train()
        coefficient = adversarial_weight(
            epoch, epochs, warmup, float(config.training["adversarial_max_weight"])
        )
        totals = {"loss": 0.0, "emotion_loss": 0.0, "adversarial_loss": 0.0, "positive_loss": 0.0}
        correct = {"emotion": 0, "adversarial": 0, "positive": 0}
        counts = {"examples": 0, "adversarial": 0, "positive": 0}
        gradient_norms: list[float] = []
        for data, labels, subjects in train_loader:
            mask = torch.ones(data.shape[:2], dtype=torch.bool)
            valid = _valid_indices(mask, device)
            data = data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            subjects = subjects.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config, device):
                outputs = model(
                    data, mask, labels=labels, grl_coefficient=coefficient, valid_indices=valid
                )
                emotion_logits = outputs["emotion_logits"]
                assert isinstance(emotion_logits, torch.Tensor)
                emotion_loss = emotion_criterion(emotion_logits, labels)
                total_loss = emotion_loss
                adversarial_loss = emotion_loss.new_zeros(())
                positive_loss = emotion_loss.new_zeros(())
                adversarial_logits = outputs["adversarial_subject_logits"]
                if isinstance(adversarial_logits, torch.Tensor):
                    adversarial_loss = subject_criterion(adversarial_logits, subjects)
                    total_loss = total_loss + adversarial_loss
                    correct["adversarial"] += int((adversarial_logits.argmax(1) == subjects).sum())
                    counts["adversarial"] += int(subjects.shape[0])
                positive_logits = outputs["positive_subject_logits"]
                if isinstance(positive_logits, torch.Tensor):
                    positive_loss = subject_criterion(positive_logits, subjects)
                    total_loss = total_loss + float(config.training["positive_subject_weight"]) * positive_loss
                    correct["positive"] += int((positive_logits.argmax(1) == subjects).sum())
                    counts["positive"] += int(subjects.shape[0])
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip = float(config.training["gradient_clip_norm"])
            core_modules = [model.backbone, model.emotion_encoder, model.emotion_classifier]
            if model.subject_encoder is not None:
                core_modules.append(model.subject_encoder)
            core_parameters = [
                parameter
                for module in core_modules for parameter in module.parameters()
                if parameter.grad is not None
            ]
            gradient = nn.utils.clip_grad_norm_(core_parameters, clip)
            if model.adversarial_subject_classifier is not None:
                nn.utils.clip_grad_norm_(model.adversarial_subject_classifier.parameters(), clip)
            if model.positive_subject_classifier is not None:
                nn.utils.clip_grad_norm_(model.positive_subject_classifier.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            batch = int(labels.shape[0])
            totals["loss"] += float(total_loss.detach()) * batch
            totals["emotion_loss"] += float(emotion_loss.detach()) * batch
            totals["adversarial_loss"] += float(adversarial_loss.detach()) * batch
            totals["positive_loss"] += float(positive_loss.detach()) * batch
            correct["emotion"] += int((emotion_logits.argmax(1) == labels).sum())
            counts["examples"] += batch
            gradient_norms.append(float(gradient.detach().cpu()))
        scheduler.step()
        dev_metrics, _, _ = evaluate_emotion(model, dev_dataset, device, config)
        row = {
            "epoch": epoch,
            "grl_coefficient": coefficient,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "subject_learning_rate": (
                float(optimizer.param_groups[1]["lr"])
                if len(optimizer.param_groups) > 1 else None
            ),
            "train_total_loss": totals["loss"] / counts["examples"],
            "train_emotion_loss": totals["emotion_loss"] / counts["examples"],
            "train_adversarial_subject_loss": totals["adversarial_loss"] / counts["examples"],
            "train_positive_subject_loss": totals["positive_loss"] / counts["examples"],
            "train_emotion_accuracy": correct["emotion"] / counts["examples"],
            "train_adversarial_subject_accuracy": (
                correct["adversarial"] / counts["adversarial"] if counts["adversarial"] else None
            ),
            "train_positive_subject_accuracy": (
                correct["positive"] / counts["positive"] if counts["positive"] else None
            ),
            "source_dev_accuracy": dev_metrics["accuracy"],
            "source_dev_balanced_accuracy": dev_metrics["balanced_accuracy"],
            "source_dev_macro_f1": dev_metrics["macro_f1"],
            "mean_gradient_norm": float(np.mean(gradient_norms)),
        }
        history.append(row)
        _write_csv(output / "training_history.csv", history)
        key = (
            float(dev_metrics["macro_f1"]),
            float(dev_metrics["balanced_accuracy"]),
            -float(epoch),
        )
        if epoch >= selection_minimum and key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save({
                "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "normalization_mean": mean,
                "normalization_std": std,
                "condition": condition,
                "protocol": protocol,
                "best_epoch": best_epoch,
                "config": {"model": config.model, "training": config.training},
                "fit_subjects": fit_subjects,
                "development_subjects": dev_subjects,
                "target_loaded_during_training": False,
            }, checkpoint)
        LOGGER.info(
            "%s %s epoch %02d dev BACC %.4f F1 %.4f GRL %.4f",
            protocol, condition, epoch,
            dev_metrics["balanced_accuracy"], dev_metrics["macro_f1"], coefficient,
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    selected_dev_metrics, _, _ = evaluate_emotion(model, dev_dataset, device, config)
    probes = probe_embeddings(model, config, mean, std, fit_subjects, dev_subjects, device)
    # Target arrays are intentionally loaded only after source-development
    # checkpoint selection and all independent source-only probes are complete.
    target_examples = load_examples(config, target_subjects, target_videos, None)
    target_dataset = AdversarialDataset(target_examples, mean, std)
    target_metrics, target_y, target_prediction = evaluate_emotion(model, target_dataset, device, config)
    target_subject_array = target_dataset.raw_subjects
    bootstrap = bootstrap_subject_metrics(
        target_y, target_prediction, target_subject_array,
        int(config.training["bootstrap_repeats"]), config.seed,
    )
    predictions = [
        {
            "subject": int(example.raw_subject),
            "video": int(example.video),
            "emotion": EMOTION_NAMES[int(example.label)],
            "target": int(y),
            "prediction": int(prediction),
        }
        for example, y, prediction in zip(target_examples, target_y, target_prediction, strict=True)
    ]
    _write_csv(output / "target_predictions.csv", predictions)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "family": "FACED-Class-Conditional-Subject-Adversarial-Light-v1",
        "condition": condition,
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
        "epochs_maximum": epochs,
        "source_selected_epoch": best_epoch,
        "checkpoint_selection": "maximum source-development Macro-F1; earlier epoch wins ties",
        "checkpoint_selection_minimum_epoch": selection_minimum,
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
        "adversarial_random_subject_accuracy": 1.0 / len(fit_subjects),
        "evidence_label": config.evidence_label,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at": utc_now(),
        "diagnostic_smoke": fit_subjects_override is not None,
    }
    _write_json(result_path, result)
    _write_json(output / "COMPLETE.json", {
        "condition": condition,
        "protocol": protocol,
        "protocol_hash": result["protocol_hash"],
        "completed_at": result["completed_at"],
    })
    del model, train_dataset, dev_dataset, target_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_matrix(config: AdversarialConfig, *, force: bool = False) -> dict[str, Any]:
    config.run_root.mkdir(parents=True, exist_ok=True)
    log = logging.FileHandler(config.run_root / "experiment.log", encoding="utf-8")
    log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(log)
    LOGGER.setLevel(logging.INFO)
    rows: list[dict[str, Any]] = []
    try:
        for protocol in config.protocols:
            for condition in config.conditions:
                result = train_task(config, protocol, condition, force=force)
                rows.append({
                    "protocol": protocol,
                    "condition": condition,
                    "best_epoch": result["source_selected_epoch"],
                    "parameters": result["parameter_count"],
                    "dev_balanced_accuracy": result["source_development"]["balanced_accuracy"],
                    "dev_macro_f1": result["source_development"]["macro_f1"],
                    "target_accuracy": result["target_test"]["accuracy"],
                    "target_balanced_accuracy": result["target_test"]["balanced_accuracy"],
                    "target_macro_f1": result["target_test"]["macro_f1"],
                    "subject_probe_ze": result["independent_source_only_probes"]["subject_id_probe_on_ze"],
                    "subject_probe_zs": result["independent_source_only_probes"].get("subject_id_probe_on_zs"),
                    "video_probe_ze": result["independent_source_only_probes"]["video_id_probe_on_ze"],
                    "selected_train_adversarial_subject_accuracy": result["selected_epoch_training"]["train_adversarial_subject_accuracy"],
                    "selected_train_positive_subject_accuracy": result["selected_epoch_training"]["train_positive_subject_accuracy"],
                    "elapsed_seconds": result["elapsed_seconds"],
                })
                _write_csv(config.run_root / "matrix_results.csv", rows)
    finally:
        LOGGER.removeHandler(log)
        log.close()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "protocol_hash": protocol_hash(config),
        "tasks": len(rows),
        "target_used_for_selection": False,
        "rows": rows,
        "paired_target_bootstrap_vs_B0": paired_target_bootstrap(config),
        "completed_at": utc_now(),
    }
    _write_json(config.run_root / "summary.json", summary)
    return summary


def run_smoke(config: AdversarialConfig, *, force: bool = False) -> dict[str, Any]:
    source, target = official_fold_subjects(config.fold)
    dev = list(config.development_subjects[:4])
    fit = [subject for subject in source if subject not in set(config.development_subjects)][:8]
    root = config.run_root.parent / f"{config.run_root.name}_smoke"
    rows = []
    for condition in ("B0_emotion_only", "B2_conditional_subject_grl", "B3_dual_conditional_grl"):
        rows.append(train_task(
            config, "conventional_subject_holdout", condition,
            run_root=root,
            fit_subjects_override=fit,
            dev_subjects_override=dev,
            target_subjects_override=target[:1],
            epochs_override=1,
            force=force,
        ))
    summary = {"status": "complete", "tasks": len(rows), "run_root": str(root), "results": rows}
    _write_json(root / "summary.json", summary)
    return summary


def status(config: AdversarialConfig) -> dict[str, Any]:
    tasks = []
    for protocol in config.protocols:
        for condition in config.conditions:
            result = _task_root(config.run_root, protocol, condition) / "result.json"
            tasks.append({"protocol": protocol, "condition": condition, "complete": result.is_file()})
    return {
        "run_root": str(config.run_root),
        "complete": all(task["complete"] for task in tasks),
        "completed_tasks": sum(task["complete"] for task in tasks),
        "total_tasks": len(tasks),
        "tasks": tasks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FACED lightweight subject-adversarial B0-B3 ablation")
    parser.add_argument("command", choices=("run", "smoke", "status"))
    parser.add_argument("--config", default="configs/faced/subject_adversarial_fold1_light.yaml")
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
