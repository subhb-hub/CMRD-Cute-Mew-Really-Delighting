from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def resample(signal: np.ndarray, original_rate: float, target_rate: float) -> np.ndarray:
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("Sampling rates must be positive")
    if np.isclose(original_rate, target_rate):
        return np.asarray(signal, dtype=np.float32, order="C")
    ratio = Fraction(target_rate / original_rate).limit_denominator(10_000)
    return np.asarray(resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1), dtype=np.float32)


def bandpass(signal: np.ndarray, rate: float, low: float, high: float, order: int) -> np.ndarray:
    if not 0 < low < high < rate / 2:
        raise ValueError(f"Invalid band {low}-{high} Hz for sampling rate {rate}")
    sos = butter(order, [low, high], btype="bandpass", fs=rate, output="sos")
    return np.asarray(sosfiltfilt(sos, signal, axis=-1), dtype=np.float32, order="C")


def preprocess_signal(signal: np.ndarray, original_rate: float, target_rate: float, broad_band: list[float], order: int) -> np.ndarray:
    value = resample(signal, original_rate, target_rate)
    return bandpass(value, target_rate, float(broad_band[0]), float(broad_band[1]), order)


def window_starts(samples: int, window: int, hop: int) -> np.ndarray:
    if window <= 0 or hop <= 0:
        raise ValueError("Window and hop sizes must be positive")
    if samples < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, samples - window + 1, hop, dtype=np.int64)

