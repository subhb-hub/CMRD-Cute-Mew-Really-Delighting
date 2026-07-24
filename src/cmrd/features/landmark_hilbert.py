from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class HilbertAtlasState:
    """Source-fitted channel-band JSD atlas on a padded native grid."""

    center: np.ndarray  # [C,B,F]
    anchors: np.ndarray  # [C,B,K,F]
    frequency_mask: np.ndarray  # [B,F]
    magnitude_mean: np.ndarray  # [C,B]
    magnitude_scale: np.ndarray  # [C,B]
    de_mean: np.ndarray  # [C,B]
    de_scale: np.ndarray  # [C,B]


def normalize_masked(values: np.ndarray, mask: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if data.shape[-1] != valid.shape[-1]:
        raise ValueError(f"Frequency dimension mismatch: {data.shape}, {valid.shape}")
    while valid.ndim < data.ndim:
        valid = valid[None]
    data = np.where(valid, np.maximum(data, epsilon), 0.0)
    return (data / np.maximum(data.sum(axis=-1, keepdims=True), epsilon)).astype(np.float32)


def jsd_numpy(left: np.ndarray, right: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    p = np.maximum(np.asarray(left, dtype=np.float64), epsilon)
    q = np.maximum(np.asarray(right, dtype=np.float64), epsilon)
    midpoint = 0.5 * (p + q)
    value = 0.5 * np.sum(p * (np.log(p) - np.log(midpoint)), axis=-1)
    value += 0.5 * np.sum(q * (np.log(q) - np.log(midpoint)), axis=-1)
    return np.maximum(value, 0.0).astype(np.float32)


def fit_channel_band_atlas(
    candidates: np.ndarray,
    frequency_mask: np.ndarray,
    landmarks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an exact-K subject/trial-balanced farthest-point atlas.

    ``candidates`` is ``[N,C,B,F]``. Callers contribute the same number of
    windows per source trial, making the mean and k-center pool trial-balanced.
    The low-frequency bands are deliberately allowed to have K larger than
    their intrinsic rank; the downstream ridge-Cholesky coordinates handle
    this without silently changing token dimensionality between bands.
    """
    values = np.asarray(candidates, dtype=np.float32)
    mask = np.asarray(frequency_mask, dtype=bool)
    if values.ndim != 4 or mask.ndim != 2 or values.shape[2:] != mask.shape:
        raise ValueError(f"Expected candidates [N,C,B,F] and mask [B,F], got {values.shape}/{mask.shape}")
    if values.shape[0] < 2 or landmarks < 1:
        raise ValueError("At least two candidates and one landmark are required")
    values = normalize_masked(values, mask)
    center = normalize_masked(values.mean(axis=0, dtype=np.float64), mask)
    channels, bands, frequencies = center.shape
    anchors = np.empty((channels, bands, int(landmarks), frequencies), dtype=np.float32)
    for band in range(bands):
        width = int(mask[band].sum())
        band_values = values[:, :, band, :width]
        band_center = center[:, band, :width]
        minimum = jsd_numpy(band_values, band_center[None])
        channel_index = np.arange(channels)
        for landmark in range(int(landmarks)):
            selected = np.argmax(minimum, axis=0)
            chosen = band_values[selected, channel_index]
            anchors[:, band, landmark, :] = 0.0
            anchors[:, band, landmark, :width] = chosen
            minimum = np.minimum(minimum, jsd_numpy(band_values, chosen[None]))
    return center, anchors


def streaming_moments(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not chunks:
        raise ValueError("At least one source chunk is required")
    total = 0
    summed: np.ndarray | None = None
    squared: np.ndarray | None = None
    for chunk in chunks:
        value = np.asarray(chunk, dtype=np.float64)
        if value.ndim != 3:
            raise ValueError(f"Moment chunks must be [T,C,B], got {value.shape}")
        total += value.shape[0]
        current_sum = value.sum(axis=0)
        current_squared = np.square(value).sum(axis=0)
        summed = current_sum if summed is None else summed + current_sum
        squared = current_squared if squared is None else squared + current_squared
    assert summed is not None and squared is not None
    mean = summed / max(total, 1)
    variance = np.maximum(squared / max(total, 1) - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
