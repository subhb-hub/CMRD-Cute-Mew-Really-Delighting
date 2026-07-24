from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmrd.faced import EMOTION_NAMES, official_fold_subjects
from scripts.run_faced_rjsd_shape_magnitude_ablation import (
    CLASSES,
    ShapeMagnitudeAxialTransformer,
    ShapeMagnitudeDataset,
    VARIANTS,
    fit_reference,
    materialize_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Class geometry of the trained FACED E representation")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--four-second-cache", type=Path, default=Path("runs/faced_4s_welch_probability_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


@torch.no_grad()
def extract(
    model: ShapeMagnitudeAxialTransformer,
    dataset: ShapeMagnitudeDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    representations = []
    labels = []
    model.eval()
    for shape, magnitude, label in loader:
        shape = shape.to(device, non_blocking=True)
        magnitude = magnitude.to(device, non_blocking=True)
        representations.append(model.encode_representation(shape, magnitude).cpu().numpy())
        labels.append(label.numpy())
    return np.concatenate(representations), np.concatenate(labels)


def standardize(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = source.mean(axis=0, dtype=np.float64)
    std = source.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    return ((source - mean) / std).astype(np.float64), ((target - mean) / std).astype(np.float64)


def geometry(source: np.ndarray, source_y: np.ndarray, target: np.ndarray, target_y: np.ndarray) -> dict[str, Any]:
    source, target = standardize(source, target)
    source_centroids = np.stack([source[source_y == label].mean(0) for label in range(CLASSES)])
    target_centroids = np.stack([target[target_y == label].mean(0) for label in range(CLASSES)])
    source_within = np.asarray([
        np.mean(np.sum(np.square(source[source_y == label] - source_centroids[label]), axis=1))
        for label in range(CLASSES)
    ])
    target_within = np.asarray([
        np.mean(np.sum(np.square(target[target_y == label] - target_centroids[label]), axis=1))
        for label in range(CLASSES)
    ])
    source_pair_distance = np.linalg.norm(source_centroids[:, None, :] - source_centroids[None, :, :], axis=-1)
    target_pair_distance = np.linalg.norm(target_centroids[:, None, :] - target_centroids[None, :, :], axis=-1)
    cross_distance = np.linalg.norm(target_centroids[:, None, :] - source_centroids[None, :, :], axis=-1)
    source_fisher = np.square(source_pair_distance) / np.maximum(source_within[:, None] + source_within[None, :], 1e-12)
    target_fisher = np.square(target_pair_distance) / np.maximum(target_within[:, None] + target_within[None, :], 1e-12)

    class_rows = []
    for label in range(CLASSES):
        other = [index for index in range(CLASSES) if index != label]
        nearest_source = min(other, key=lambda index: source_pair_distance[label, index])
        nearest_target = min(other, key=lambda index: target_pair_distance[label, index])
        nearest_cross = int(np.argmin(cross_distance[label]))
        same_cross = float(cross_distance[label, label])
        cross_order = np.argsort(cross_distance[label])
        same_rank = int(np.flatnonzero(cross_order == label)[0]) + 1
        class_rows.append({
            "class_index": label,
            "class_name": EMOTION_NAMES[label],
            "source_within_variance": float(source_within[label]),
            "target_within_variance": float(target_within[label]),
            "source_target_same_class_distance": same_cross,
            "target_to_source_same_class_rank": same_rank,
            "target_nearest_source_class": nearest_cross,
            "target_nearest_source_name": EMOTION_NAMES[nearest_cross],
            "target_nearest_source_distance": float(cross_distance[label, nearest_cross]),
            "source_nearest_other_class": nearest_source,
            "source_nearest_other_name": EMOTION_NAMES[nearest_source],
            "source_nearest_other_distance": float(source_pair_distance[label, nearest_source]),
            "source_nearest_fisher_ratio": float(source_fisher[label, nearest_source]),
            "target_nearest_other_class": nearest_target,
            "target_nearest_other_name": EMOTION_NAMES[nearest_target],
            "target_nearest_other_distance": float(target_pair_distance[label, nearest_target]),
            "target_nearest_fisher_ratio": float(target_fisher[label, nearest_target]),
        })
    return {
        "dimensions": int(source.shape[1]),
        "classes": class_rows,
        "source_pair_distance": source_pair_distance.tolist(),
        "target_pair_distance": target_pair_distance.tolist(),
        "target_to_source_distance": cross_distance.tolist(),
        "source_fisher_ratio": source_fisher.tolist(),
        "target_fisher_ratio": target_fisher.tolist(),
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = resolve(args.checkpoint)
    args.four_second_cache = resolve(args.four_second_cache)
    args.output = resolve(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_subjects, target_subjects = official_fold_subjects(args.fold)
    reference = fit_reference(args.four_second_cache, source_subjects)
    source = materialize_variant(args.four_second_cache, source_subjects, reference, VARIANTS["E"])
    target = materialize_variant(args.four_second_cache, target_subjects, reference, VARIANTS["E"])
    source_dataset = ShapeMagnitudeDataset(
        source,
        checkpoint["shape_mean"],
        checkpoint["shape_std"],
        checkpoint["magnitude_mean"],
        checkpoint["magnitude_std"],
    )
    target_dataset = ShapeMagnitudeDataset(
        target,
        checkpoint["shape_mean"],
        checkpoint["shape_std"],
        checkpoint["magnitude_mean"],
        checkpoint["magnitude_std"],
    )
    model = ShapeMagnitudeAxialTransformer(48, 4, 1, 2, True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    source_representation, source_y = extract(model, source_dataset, args.batch_size, device)
    target_representation, target_y = extract(model, target_dataset, args.batch_size, device)
    if source_representation.shape[1] != 396:
        raise ValueError(f"Expected 396-dimensional E representation, got {source_representation.shape}")
    output = {
        "fold": args.fold,
        "checkpoint": str(args.checkpoint),
        "feature": "E_4s_welch_signed_shape_magnitude",
        "source_trials": int(len(source_y)),
        "target_trials": int(len(target_y)),
        "geometry_standardization": "each representation coordinate standardized from source trials only",
        "shape": geometry(source_representation[:, :96], source_y, target_representation[:, :96], target_y),
        "magnitude": geometry(source_representation[:, 96:], source_y, target_representation[:, 96:], target_y),
        "combined": geometry(source_representation, source_y, target_representation, target_y),
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tenderness = {
        branch: output[branch]["classes"][8]
        for branch in ("shape", "magnitude", "combined")
    }
    print(json.dumps(tenderness, indent=2))


if __name__ == "__main__":
    main()
