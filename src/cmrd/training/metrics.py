from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(targets: np.ndarray, predictions: np.ndarray, classes: int) -> dict[str, object]:
    labels = list(range(classes))
    precision, recall, f1, support = precision_recall_fscore_support(targets, predictions, labels=labels, zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": [
            {"class": label, "precision": float(precision[label]), "recall": float(recall[label]), "f1": float(f1[label]), "support": int(support[label])}
            for label in labels
        ],
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }

