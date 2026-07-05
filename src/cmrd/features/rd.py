from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.signal import welch

from .signal import window_starts


def normalize_histograms(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    value = np.clip(np.asarray(values, dtype=np.float32), epsilon, None)
    denominator = value.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise ValueError("Invalid histogram bin sum")
    return value / denominator


def extract_spectral_histograms(signal: np.ndarray, rate: float, window_seconds: float, hop_seconds: float, bands: Mapping[str, Sequence[float]], bins_per_band: int, nfft: int, epsilon: float = 1e-12) -> np.ndarray:
    window = int(round(window_seconds * rate))
    hop = int(round(hop_seconds * rate))
    starts = window_starts(signal.shape[-1], window, hop)
    if not starts.size:
        raise ValueError("Trial is shorter than one RD window")
    if nfft < window:
        raise ValueError("spectral_nfft must be at least the window size")
    output = np.zeros((starts.size, signal.shape[0], len(bands), bins_per_band), dtype=np.float32)
    for window_index, start in enumerate(starts):
        frequencies, psd = welch(signal[:, start:start + window], fs=rate, window="hann", nperseg=window, noverlap=0, nfft=nfft, detrend="constant", scaling="density", axis=-1)
        for band_index, (name, limits) in enumerate(bands.items()):
            low, high = map(float, limits)
            selected = (frequencies >= low) & (frequencies < high)
            if not np.any(selected):
                raise ValueError(f"No spectral samples for band {name}")
            indices = np.floor((frequencies[selected] - low) * bins_per_band / (high - low)).astype(np.int64)
            indices = np.clip(indices, 0, bins_per_band - 1)
            histogram = output[window_index, :, band_index]
            selected_psd = psd[:, selected]
            for bin_index in np.unique(indices):
                histogram[:, bin_index] = selected_psd[:, indices == bin_index].sum(axis=-1)
            denominator = histogram.sum(axis=-1, keepdims=True)
            if np.any(denominator <= epsilon):
                raise FloatingPointError(f"Near-zero spectral power in {name}")
            histogram /= denominator
    return output


def fit_reference(histograms: Iterable[np.ndarray], epsilon: float = 1e-12) -> tuple[np.ndarray, int]:
    total: np.ndarray | None = None
    count = 0
    expected: tuple[int, ...] | None = None
    for histogram in histograms:
        current = normalize_histograms(histogram, epsilon)
        if current.ndim != 4:
            raise ValueError(f"p_hist must be [T,C,B,F], got {current.shape}")
        if expected is None:
            expected = current.shape[1:]
            total = np.zeros(expected, dtype=np.float64)
        elif current.shape[1:] != expected:
            raise ValueError("Inconsistent p_hist shapes")
        assert total is not None
        total += current.sum(axis=0, dtype=np.float64)
        count += current.shape[0]
    if total is None or count == 0:
        raise ValueError("Cannot fit an RD reference from zero source-training windows")
    return normalize_histograms(total / count, epsilon), count


def transform_rd(histogram: np.ndarray, reference: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    p = normalize_histograms(histogram, epsilon)
    q = normalize_histograms(reference, epsilon)
    if p.ndim != 4 or q.ndim != 3 or p.shape[1:] != q.shape:
        raise ValueError(f"Expected P=[T,C,B,F], Q=[C,B,F], got {p.shape}, {q.shape}")
    midpoint = 0.5 * (p + q[None])
    divergence = 0.5 * (p * (np.log(p) - np.log(midpoint))).sum(axis=-1)
    divergence += 0.5 * (q * (np.log(q) - np.log(midpoint))).sum(axis=-1)
    result = np.maximum(divergence, 0).astype(np.float32).reshape(p.shape[0], -1)
    if not np.isfinite(result).all():
        raise FloatingPointError("RD extraction produced non-finite values")
    return result

