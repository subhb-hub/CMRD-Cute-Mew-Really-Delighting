from __future__ import annotations

import numpy as np


def fit_source_normalizer(
    data: np.ndarray, mask: np.ndarray, train_indices: np.ndarray, epsilon: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature-wise statistics using only real source-training windows."""
    feature_dim = data.shape[-1]
    total = np.zeros(feature_dim, dtype=np.float64)
    total_sq = np.zeros(feature_dim, dtype=np.float64)
    count = 0
    for index in train_indices:
        real = np.asarray(data[index, mask[index]], dtype=np.float64)
        total += real.sum(axis=0)
        total_sq += np.square(real).sum(axis=0)
        count += real.shape[0]
    if count == 0:
        raise ValueError("No real source-training windows available for normalization")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < epsilon] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_trial(trial: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray((trial - mean) / std, dtype=np.float32)

