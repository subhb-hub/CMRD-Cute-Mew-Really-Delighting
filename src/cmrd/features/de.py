from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .signal import bandpass, window_starts


def extract_de(signal: np.ndarray, rate: float, window_seconds: float, hop_seconds: float, bands: Mapping[str, Sequence[float]], filter_order: int, epsilon: float = 1e-8) -> np.ndarray:
    window = int(round(window_seconds * rate))
    hop = int(round(hop_seconds * rate))
    starts = window_starts(signal.shape[-1], window, hop)
    if not starts.size:
        raise ValueError("Trial is shorter than one DE window")
    features = np.empty((starts.size, signal.shape[0], len(bands)), dtype=np.float32)
    constant = 2.0 * np.pi * np.e
    for band_index, (name, limits) in enumerate(bands.items()):
        if len(limits) != 2:
            raise ValueError(f"Band {name} must contain [low, high]")
        filtered = bandpass(signal, rate, float(limits[0]), float(limits[1]), filter_order)
        for window_index, start in enumerate(starts):
            variance = np.var(filtered[:, start:start + window], axis=-1, ddof=1)
            features[window_index, :, band_index] = 0.5 * np.log(constant * variance + epsilon)
    result = features.reshape(starts.size, -1)
    if not np.isfinite(result).all():
        raise FloatingPointError("DE extraction produced non-finite values")
    return result

