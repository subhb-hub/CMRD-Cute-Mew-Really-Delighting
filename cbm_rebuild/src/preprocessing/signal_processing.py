from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


def resample_signal(signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rates must be positive")
    if np.isclose(original_fs, target_fs):
        return np.asarray(signal, dtype=np.float32, order="C")
    ratio = Fraction(target_fs / original_fs).limit_denominator(10_000)
    result = resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
    return np.asarray(result, dtype=np.float32, order="C")


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = fs / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Band must satisfy 0 < low < high < Nyquist; got {low_hz}-{high_hz} Hz at fs={fs}"
        )
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    try:
        filtered = sosfiltfilt(sos, signal, axis=-1)
    except ValueError as exc:
        raise ValueError(
            f"Signal with {signal.shape[-1]} samples is too short for {order}th-order "
            f"{low_hz}-{high_hz} Hz zero-phase filtering"
        ) from exc
    return np.asarray(filtered, dtype=np.float32, order="C")


def unified_preprocess(
    signal: np.ndarray,
    original_fs: float,
    target_fs: float,
    broad_band: tuple[float, float],
    filter_order: int,
) -> np.ndarray:
    resampled = resample_signal(signal, original_fs, target_fs)
    return bandpass_filter(resampled, target_fs, broad_band[0], broad_band[1], filter_order)

