from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> dict[str, object]:
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
    }

