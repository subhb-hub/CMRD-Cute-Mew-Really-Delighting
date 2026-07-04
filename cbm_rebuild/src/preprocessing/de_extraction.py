from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .signal_processing import bandpass_filter


DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 31.0),
    "gamma": (31.0, 50.0),
}


def window_starts(num_samples: int, window_size: int, hop_size: int) -> np.ndarray:
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive")
    if num_samples < window_size:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, num_samples - window_size + 1, hop_size, dtype=np.int64)


def extract_de_features(
    signal: np.ndarray,
    fs: float,
    window_seconds: float,
    hop_seconds: float,
    bands: Mapping[str, Sequence[float]] = DEFAULT_BANDS,
    filter_order: int = 4,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Return Gaussian differential entropy with shape [windows, channels * bands]."""
    window_size = int(round(window_seconds * fs))
    hop_size = int(round(hop_seconds * fs))
    starts = window_starts(signal.shape[-1], window_size, hop_size)
    channels = signal.shape[0]
    if starts.size == 0:
        raise ValueError(
            f"Trial has {signal.shape[-1]} samples, shorter than one {window_seconds}s window"
        )
    features = np.empty((starts.size, channels, len(bands)), dtype=np.float32)
    constant = 2.0 * np.pi * np.e
    for band_index, (name, limits) in enumerate(bands.items()):
        if len(limits) != 2:
            raise ValueError(f"Band {name!r} must contain [low_hz, high_hz]")
        filtered = bandpass_filter(signal, fs, float(limits[0]), float(limits[1]), filter_order)
        for window_index, start in enumerate(starts):
            variance = np.var(filtered[:, start : start + window_size], axis=-1, ddof=1)
            features[window_index, :, band_index] = 0.5 * np.log(constant * variance + epsilon)
    if not np.isfinite(features).all():
        raise FloatingPointError("DE extraction produced NaN or infinite values")
    return features.reshape(starts.size, channels * len(bands))

