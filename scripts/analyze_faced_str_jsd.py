from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


C0 = "C0_absolute_de"
C2 = "C2_absolute_de_unsigned_pointwise_jsd"
C3 = "C3_absolute_de_signed_pointwise_jsd"
C4 = "C4_absolute_de_delta_de_signed_pointwise_jsd"
C5 = "C5_absolute_de_pointwise_log_ratio"
C6 = "C6_c4_reference_quality_gate"


def read_predictions(root: Path, protocol: str, temporal: str, condition: str):
    path = root / protocol / temporal / condition / "fold-01" / "seed-42" / "target_predictions.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [{key: int(row[key]) for key in ("subject", "video", "target", "prediction")} for row in csv.DictReader(stream)]
    rows.sort(key=lambda row: (row["subject"], row["video"]))
    return rows


def paired_bootstrap(left, right, repeats: int, seed: int):
    left_keys = [(row["subject"], row["video"], row["target"]) for row in left]
    right_keys = [(row["subject"], row["video"], row["target"]) for row in right]
    if left_keys != right_keys:
        raise ValueError("Paired comparison rows do not align")
    subjects = np.asarray([row["subject"] for row in left], dtype=np.int64)
    y = np.asarray([row["target"] for row in left], dtype=np.int64)
    left_prediction = np.asarray([row["prediction"] for row in left], dtype=np.int64)
    right_prediction = np.asarray([row["prediction"] for row in right], dtype=np.int64)
    unique = np.unique(subjects)
    def subject_confusions(prediction):
        values = np.zeros((len(unique), 9, 9), dtype=np.int64)
        for index, subject in enumerate(unique):
            selected = subjects == subject
            np.add.at(values[index], (y[selected], prediction[selected]), 1)
        return values

    def metrics(confusion):
        accuracy = np.trace(confusion) / max(confusion.sum(), 1)
        true_positive = np.diag(confusion).astype(np.float64)
        denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
        per_class_f1 = np.divide(
            2.0 * true_positive, denominator,
            out=np.zeros_like(true_positive), where=denominator > 0,
        )
        return float(accuracy), float(per_class_f1.mean())

    left_confusions = subject_confusions(left_prediction)
    right_confusions = subject_confusions(right_prediction)
    rng = np.random.default_rng(seed)
    accuracy = np.empty(int(repeats), dtype=np.float64)
    macro_f1 = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        counts = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
        left_metrics = metrics(np.tensordot(counts, left_confusions, axes=1))
        right_metrics = metrics(np.tensordot(counts, right_confusions, axes=1))
        accuracy[repeat] = left_metrics[0] - right_metrics[0]
        macro_f1[repeat] = left_metrics[1] - right_metrics[1]
    point_accuracy = float(accuracy_score(y, left_prediction) - accuracy_score(y, right_prediction))
    point_f1 = float(f1_score(y, left_prediction, average="macro", zero_division=0) - f1_score(y, right_prediction, average="macro", zero_division=0))
    return {
        "accuracy_difference_left_minus_right": point_accuracy,
        "accuracy_difference_ci95": np.quantile(accuracy, [0.025, 0.975]).tolist(),
        "accuracy_probability_left_greater": float(np.mean(accuracy > 0)),
        "macro_f1_difference_left_minus_right": point_f1,
        "macro_f1_difference_ci95": np.quantile(macro_f1, [0.025, 0.975]).tolist(),
        "macro_f1_probability_left_greater": float(np.mean(macro_f1 > 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/faced_str_jsd_fold1_light_seed42")
    parser.add_argument("--repeats", type=int, default=5000)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    direct = {}
    for protocol_index, protocol in enumerate(("conventional_subject_holdout", "subject_and_stimulus_holdout")):
        direct[protocol] = {}
        five = {condition: read_predictions(root, protocol, "five_second_average", condition) for condition in (C0, C2, C3, C4, C5, C6)}
        one = {condition: read_predictions(root, protocol, "one_second_control", condition) for condition in (C0, C4, C5)}
        comparisons = {
            "C4_minus_C5_five_second": (five[C4], five[C5]),
            "C4_minus_C6_five_second": (five[C4], five[C6]),
            "C3_minus_C2_five_second": (five[C3], five[C2]),
            "C4_minus_C0_one_second": (one[C4], one[C0]),
            "C5_minus_C0_one_second": (one[C5], one[C0]),
            "C0_five_minus_one_second": (five[C0], one[C0]),
            "C4_five_minus_one_second": (five[C4], one[C4]),
            "C5_five_minus_one_second": (five[C5], one[C5]),
        }
        for offset, (name, (left, right)) in enumerate(comparisons.items()):
            direct[protocol][name] = paired_bootstrap(
                left, right, args.repeats, 42 + 100 * protocol_index + offset
            )

    per_class = {}
    for protocol in ("conventional_subject_holdout", "subject_and_stimulus_holdout"):
        per_class[protocol] = {}
        for temporal, conditions in {
            "five_second_average": (C0, C2, C3, C4, C5, C6),
            "one_second_control": (C0, C4, C5),
        }.items():
            per_class[protocol][temporal] = {}
            for condition in conditions:
                rows = read_predictions(root, protocol, temporal, condition)
                y = np.asarray([row["target"] for row in rows], dtype=np.int64)
                prediction = np.asarray([row["prediction"] for row in rows], dtype=np.int64)
                per_class[protocol][temporal][condition] = recall_score(
                    y, prediction, labels=np.arange(9), average=None, zero_division=0
                ).tolist()

    audits = []
    all_results = list(root.glob("*/*/*/fold-01/seed-42/result.json"))
    for path in all_results:
        result = json.loads(path.read_text(encoding="utf-8"))
        audits.append({
            "path": str(path.relative_to(root)),
            "target_loaded_during_training": result["target_loaded_during_training"],
            "target_used_for_checkpoint_or_hyperparameter_selection": result["target_used_for_checkpoint_or_hyperparameter_selection"],
            "target_used_for_gradients": result["target_used_for_gradients"],
            "protocol_hash": result["protocol_hash"],
            "parameter_count": result["parameter_count"],
            "jsd_invariant_error": result["feature_audit"]["maximum_signed_jsd_invariant_error"],
        })
    validation = {
        "result_files": len(audits),
        "all_target_isolation_flags_pass": all(
            not row["target_loaded_during_training"]
            and not row["target_used_for_checkpoint_or_hyperparameter_selection"]
            and not row["target_used_for_gradients"] for row in audits
        ),
        "unique_protocol_hashes": sorted({row["protocol_hash"] for row in audits}),
        "unique_parameter_counts": sorted({row["parameter_count"] for row in audits}),
        "maximum_jsd_invariant_error": max(row["jsd_invariant_error"] for row in audits),
        "summary_tasks": summary["tasks"],
    }
    output = {
        "status": "complete",
        "direct_paired_subject_bootstrap": direct,
        "per_class_recall": per_class,
        "validation_audit": validation,
    }
    (root / "analysis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
