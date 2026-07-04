from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def source_train_validation_split(
    subjects: np.ndarray,
    labels: np.ndarray,
    target_subject: int,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.flatnonzero(subjects == target_subject)
    source = np.flatnonzero(subjects != target_subject)
    if target.size == 0 or source.size == 0:
        raise ValueError(f"Invalid LOSO fold for subject {target_subject}")
    train, validation = train_test_split(
        source,
        test_size=validation_fraction,
        random_state=seed + int(target_subject),
        shuffle=True,
        stratify=labels[source],
    )
    return np.sort(train), np.sort(validation), np.sort(target)
