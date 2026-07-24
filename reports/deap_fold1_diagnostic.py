from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "runs" / "deap_native_compact_v1_seed42"
FIXED_CACHE = Path(
    r"C:\Users\Lin\Documents\Arbitruam\Dataset\Processed\CMRD\deap"
    r"\de_rjsd_ica_1s_hop1\b84bab9e4f721dbe"
)
FEATURE_CACHE = (
    RUN_ROOT / "feature_cache" / "deap" / "fold-01" / "4b6b63c7a0c31b26"
)
OUTPUT_JSON = PROJECT_ROOT / "reports" / "deap_fold1_diagnostic_data.json"
PROBE_CSV = PROJECT_ROOT / "reports" / "deap_fold1_probe_metrics.csv"
CURVE_CSV = PROJECT_ROOT / "reports" / "deap_fold1_target_curve.csv"

REPRESENTATIONS = {
    "sqrt-JSD": "native_sqrt_jsd",
    "Fisher–Rao PC1": "native_fisher_rao_pca",
}
CLASS_NAMES = {0: "LVLA", 1: "LVHA", 2: "HVLA", 3: "HVHA"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(4), average="macro", zero_division=0)),
    }


def label_counts(labels: Iterable[int]) -> dict[str, int]:
    values = Counter(int(value) for value in labels)
    return {CLASS_NAMES[index]: int(values.get(index, 0)) for index in range(4)}


def eta_squared(features: np.ndarray, groups: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    groups = np.asarray(groups)
    overall = features.mean(axis=0)
    total = np.square(features - overall).sum(axis=0)
    between = np.zeros(features.shape[1], dtype=np.float64)
    for value in np.unique(groups):
        subset = features[groups == value]
        between += subset.shape[0] * np.square(subset.mean(axis=0) - overall)
    return np.divide(between, total, out=np.zeros_like(between), where=total > 1e-12)


def adjusted_eta_squared(values: np.ndarray, sample_count: int, group_count: int) -> np.ndarray:
    """Remove the finite-sample advantage of drivers with more groups."""
    if sample_count <= group_count:
        return np.full_like(values, np.nan, dtype=np.float64)
    return 1.0 - (1.0 - values) * (sample_count - 1) / (sample_count - group_count)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(write_outputs: bool = True) -> dict[str, Any]:
    fold = read_json(FIXED_CACHE / "folds" / "fold-01" / "manifest.json")
    entries: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for entry in fold["groups"][split]:
            entries.append({**entry, "split": split})
    entries.sort(key=lambda item: int(item["source_index"]))

    metadata_by_trial = {
        path.stem: read_json(path)
        for path in (FIXED_CACHE / "trial_metadata").glob("*.json")
    }
    features: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        trial_id = str(entry["trial_id"])
        metadata = metadata_by_trial[trial_id]
        record = metadata["record_metadata"]
        ica = metadata["ica"]
        feature_path = FEATURE_CACHE / "trials" / f"{trial_id}.npz"
        with np.load(feature_path, allow_pickle=False) as archive:
            for name, key in REPRESENTATIONS.items():
                value = np.asarray(archive[key], dtype=np.float32)
                features[name].append(np.concatenate([value.mean(axis=0), value.std(axis=0)]))
        rows.append({
            "trial_id": trial_id,
            "split": entry["split"],
            "subject": int(entry["subject"]),
            "trial": int(entry["trial"]),
            "label": int(entry["label"]),
            "experiment_id": int(record["experiment_id"]),
            "valence": float(record["valence"]),
            "arousal": float(record["arousal"]),
            "near_va_boundary_0_5": bool(
                abs(float(record["valence"]) - 5.0) <= 0.5
                or abs(float(record["arousal"]) - 5.0) <= 0.5
            ),
            "ica_fallback": bool(ica["fit_errors"]),
            "interpolated_channels": len(ica["interpolated_bad_channels"]),
            "excluded_components": len(ica["excluded_components"]),
            "cleaned_std_microvolt": float(ica["cleaned_std_microvolt"]),
        })

    y = np.asarray([row["label"] for row in rows], dtype=np.int64)
    subjects = np.asarray([row["subject"] for row in rows], dtype=np.int64)
    experiment_ids = np.asarray([row["experiment_id"] for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in rows])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    all_source_mask = train_mask | validation_mask

    source_majority = Counter(y[train_mask]).most_common(1)[0][0]
    majority_prediction = np.full(test_mask.sum(), source_majority, dtype=np.int64)
    majority_metrics = metric_row(y[test_mask], majority_prediction)

    probe_rows: list[dict[str, Any]] = []
    quality_probe_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    for representation, values in features.items():
        matrix = np.stack(values)
        for class_weight_name, class_weight in (("unweighted", None), ("balanced", "balanced")):
            for fit_name, fit_mask, evaluation_name, evaluation_mask in (
                ("29 source-train", train_mask, "2 source-validation", validation_mask),
                ("29 source-train", train_mask, "target subject 1", test_mask),
                ("31 all-source", all_source_mask, "target subject 1", test_mask),
            ):
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=5000,
                        class_weight=class_weight,
                        random_state=42,
                    ),
                )
                model.fit(matrix[fit_mask], y[fit_mask])
                predicted = model.predict(matrix[evaluation_mask])
                probe_rows.append({
                    "representation": representation,
                    "classifier": f"logistic-{class_weight_name}",
                    "fit_scope": fit_name,
                    "evaluation": evaluation_name,
                    "n": int(evaluation_mask.sum()),
                    **metric_row(y[evaluation_mask], predicted),
                })

        source_matrix = matrix[train_mask]
        effects = {
            "emotion label": eta_squared(source_matrix, y[train_mask]),
            "subject identity": eta_squared(source_matrix, subjects[train_mask]),
            "video identity": eta_squared(source_matrix, experiment_ids[train_mask]),
        }
        effect_groups = {
            "emotion label": y[train_mask],
            "subject identity": subjects[train_mask],
            "video identity": experiment_ids[train_mask],
        }
        for driver, values_eta in effects.items():
            group_count = int(np.unique(effect_groups[driver]).size)
            adjusted = adjusted_eta_squared(values_eta, int(train_mask.sum()), group_count)
            effect_rows.append({
                "representation": representation,
                "driver": driver,
                "mean_eta_squared": float(values_eta.mean()),
                "median_eta_squared": float(np.median(values_eta)),
                "p90_eta_squared": float(np.quantile(values_eta, 0.9)),
                "mean_adjusted_eta_squared": float(adjusted.mean()),
                "group_count": group_count,
            })

        quality_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42),
        )
        quality_model.fit(matrix[all_source_mask], y[all_source_mask])
        target_prediction = quality_model.predict(matrix[test_mask])
        target_truth = y[test_mask]
        target_fallback = np.asarray([
            row["ica_fallback"] for row in rows if row["split"] == "test"
        ], dtype=bool)
        for subgroup_name, subgroup_mask in (
            ("ICA fallback", target_fallback),
            ("ICA fit succeeded", ~target_fallback),
        ):
            quality_probe_rows.append({
                "representation": representation,
                "subgroup": subgroup_name,
                "n": int(subgroup_mask.sum()),
                "accuracy": float(accuracy_score(
                    target_truth[subgroup_mask], target_prediction[subgroup_mask]
                )),
                "macro_f1_all_classes": float(f1_score(
                    target_truth[subgroup_mask], target_prediction[subgroup_mask],
                    labels=np.arange(4), average="macro", zero_division=0,
                )),
            })

        scaler = StandardScaler().fit(source_matrix)
        for split_name, mask in (("source-validation", validation_mask), ("target subject 1", test_mask)):
            z = scaler.transform(matrix[mask])
            shift_rows.append({
                "representation": representation,
                "split": split_name,
                "mean_absolute_z": float(np.abs(z).mean()),
                "rms_z": float(np.sqrt(np.square(z).mean())),
                "fraction_abs_z_gt_3": float((np.abs(z) > 3.0).mean()),
            })

    train_experiment_majority: dict[int, int] = {}
    train_experiment_agreement: list[float] = []
    for experiment_id in sorted(set(experiment_ids[train_mask])):
        labels = y[train_mask & (experiment_ids == experiment_id)]
        majority, count = Counter(labels).most_common(1)[0]
        train_experiment_majority[int(experiment_id)] = int(majority)
        train_experiment_agreement.append(float(count / labels.size))
    target_video_prediction = np.asarray([
        train_experiment_majority[int(value)] for value in experiment_ids[test_mask]
    ])
    validation_video_prediction = np.asarray([
        train_experiment_majority[int(value)] for value in experiment_ids[validation_mask]
    ])

    curve_path = (
        RUN_ROOT / "deap" / "a_native_sqrt_jsd_base_v2" / "fold-01" / "seed-42"
        / "target_curve.json"
    )
    curve = read_json(curve_path)["curve"]
    history_path = curve_path.parent / "training_history.csv"
    with history_path.open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    curve_rows = [{
        "epoch": int(row["epoch"]),
        "accuracy": float(row["accuracy"]),
        "balanced_accuracy": float(row["balanced_accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "majority_accuracy": majority_metrics["accuracy"],
        "chance_balanced_accuracy": 0.25,
    } for row in curve]

    pca = read_json(FEATURE_CACHE / "fisher_rao_pca_state.json")
    pca_rows = [{
        "band": band,
        "mean_explained_variance_ratio": float(value),
    } for band, value in pca["mean_explained_variance_ratio_by_band"].items()]

    target_rows = [row for row in rows if row["split"] == "test"]
    source_rows = [row for row in rows if row["split"] == "train"]
    source_cleaned_std_by_subject = {
        subject: float(np.median([
            row["cleaned_std_microvolt"] for row in source_rows if row["subject"] == subject
        ]))
        for subject in sorted({row["subject"] for row in source_rows})
    }
    target_cleaned_std = float(np.median([row["cleaned_std_microvolt"] for row in target_rows]))
    source_cleaned_values = np.asarray(list(source_cleaned_std_by_subject.values()))

    report = {
        "run_status": {
            "sqrt_jsd": "interrupted after epoch 190; no final result.json/checkpoint",
            "fisher_rao": "pending; no neural training curve",
            "completed_target_curve_points": len(curve_rows),
        },
        "split": {
            "source_train_trials": int(train_mask.sum()),
            "source_validation_trials": int(validation_mask.sum()),
            "target_trials": int(test_mask.sum()),
            "source_train_label_counts": label_counts(y[train_mask]),
            "source_validation_label_counts": label_counts(y[validation_mask]),
            "target_label_counts": label_counts(y[test_mask]),
        },
        "majority_baseline_target": {
            "predicted_class": CLASS_NAMES[int(source_majority)],
            **majority_metrics,
        },
        "sqrt_jsd_training": {
            "epochs_completed": int(history[-1]["epoch"]),
            "initial_train_loss": float(history[0]["train_loss"]),
            "final_train_loss": float(history[-1]["train_loss"]),
            "relative_loss_reduction": float(
                1.0 - float(history[-1]["train_loss"]) / float(history[0]["train_loss"])
            ),
            "best_target_macro_f1": max(curve_rows, key=lambda row: row["macro_f1"]),
            "best_target_balanced_accuracy": max(
                curve_rows, key=lambda row: row["balanced_accuracy"]
            ),
            "last_target_point": curve_rows[-1],
        },
        "probe_metrics": probe_rows,
        "quality_probe_metrics": quality_probe_rows,
        "effect_sizes": effect_rows,
        "distribution_shift": shift_rows,
        "video_label_consistency": {
            "mean_source_majority_agreement": float(np.mean(train_experiment_agreement)),
            "source_majority_to_validation": metric_row(
                y[validation_mask], validation_video_prediction
            ),
            "source_majority_to_target": metric_row(y[test_mask], target_video_prediction),
        },
        "threshold_sensitivity": {
            "source_train_near_boundary_count": int(sum(
                row["near_va_boundary_0_5"] for row in source_rows
            )),
            "source_train_near_boundary_fraction": float(np.mean([
                row["near_va_boundary_0_5"] for row in source_rows
            ])),
            "target_near_boundary_count": int(sum(
                row["near_va_boundary_0_5"] for row in target_rows
            )),
            "target_near_boundary_fraction": float(np.mean([
                row["near_va_boundary_0_5"] for row in target_rows
            ])),
        },
        "ica_diagnostic": {
            "target_fallback_trials": int(sum(row["ica_fallback"] for row in target_rows)),
            "target_interpolated_trials": int(sum(
                row["interpolated_channels"] > 0 for row in target_rows
            )),
            "target_median_cleaned_std_microvolt": target_cleaned_std,
            "source_subject_median_cleaned_std_microvolt": float(np.median(source_cleaned_values)),
            "target_percentile_among_source_subject_medians": float(
                np.mean(source_cleaned_values <= target_cleaned_std)
            ),
        },
        "fisher_rao_pca": pca_rows,
        "target_curve": curve_rows,
    }
    if write_outputs:
        OUTPUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        write_csv(PROBE_CSV, probe_rows)
        write_csv(CURVE_CSV, curve_rows)
    return report


if __name__ == "__main__":
    result = run_analysis(write_outputs=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
