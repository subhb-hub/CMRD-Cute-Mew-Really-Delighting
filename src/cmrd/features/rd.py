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


def transform_signed_sqrt_rd(
    histogram: np.ndarray,
    reference: np.ndarray,
    bands_hz: Sequence[Sequence[float]],
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return signed ``sqrt(JSD)`` using the spectral-centroid shift as sign.

    Histogram bins represent frequency positions within each band.  A positive
    value therefore means that the window centroid moved toward higher
    frequencies relative to the source-only reference; a negative value means
    a lower-frequency shift.  Exact centroid ties deterministically use +1 so
    a non-zero divergence is never erased by ``sign(0)``.
    """
    p = normalize_histograms(histogram, epsilon)
    q = normalize_histograms(reference, epsilon)
    if p.ndim != 4 or q.ndim != 3 or p.shape[1:] != q.shape:
        raise ValueError(f"Expected P=[T,C,B,F], Q=[C,B,F], got {p.shape}, {q.shape}")
    if len(bands_hz) != p.shape[2]:
        raise ValueError(f"Expected {p.shape[2]} band limits, got {len(bands_hz)}")

    centers = np.empty((p.shape[2], p.shape[3]), dtype=np.float32)
    for band_index, limits in enumerate(bands_hz):
        if len(limits) != 2:
            raise ValueError(f"Band {band_index} must contain [low, high], got {limits}")
        low, high = map(float, limits)
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError(f"Invalid band limits at index {band_index}: {limits}")
        width = (high - low) / p.shape[3]
        centers[band_index] = low + (np.arange(p.shape[3], dtype=np.float32) + 0.5) * width

    midpoint = 0.5 * (p + q[None])
    divergence = 0.5 * (p * (np.log(p) - np.log(midpoint))).sum(axis=-1)
    divergence += 0.5 * (q * (np.log(q) - np.log(midpoint))).sum(axis=-1)
    p_centroid = (p * centers[None, None]).sum(axis=-1)
    q_centroid = (q * centers[None]).sum(axis=-1)
    sign = np.where(p_centroid >= q_centroid[None], 1.0, -1.0).astype(np.float32)
    result = (sign * np.sqrt(np.maximum(divergence, 0))).astype(np.float32)
    result = result.reshape(p.shape[0], -1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Signed sqrt RD extraction produced non-finite values")
    return result


def native_frequency_grid(
    rate: float,
    window_seconds: float,
    bands: Mapping[str, Sequence[float]],
) -> list[np.ndarray]:
    """Return the unpadded FFT frequencies inside each half-open band.

    A one-second, 200 Hz window therefore uses a 200-point FFT and retains
    3/4/6/17/19 samples for the standard delta/theta/alpha/beta/gamma bands.
    No zero padding or artificial equal-width histogram is introduced.
    """
    window = int(round(float(window_seconds) * float(rate)))
    if window <= 0 or rate <= 0:
        raise ValueError("rate and window_seconds must be positive")
    frequencies = np.fft.rfftfreq(window, d=1.0 / float(rate)).astype(np.float32)
    output: list[np.ndarray] = []
    for name, limits in bands.items():
        if len(limits) != 2:
            raise ValueError(f"Band {name} must contain [low, high]")
        low, high = map(float, limits)
        if not 0.0 <= low < high <= float(rate) / 2.0:
            raise ValueError(f"Invalid band {name}={limits} for rate={rate}")
        selected = frequencies[(frequencies >= low) & (frequencies < high)]
        if selected.size < 2:
            raise ValueError(f"Native grid for {name} requires at least two frequencies")
        output.append(np.ascontiguousarray(selected, dtype=np.float32))
    return output


def extract_native_spectral_distributions(
    signal: np.ndarray,
    rate: float,
    window_seconds: float,
    hop_seconds: float,
    bands: Mapping[str, Sequence[float]],
    epsilon: float = 1e-12,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract normalized PSDs on the native, non-zero-padded FFT grid.

    Returns one ``[T,C,F_b]`` array per frequency band plus the corresponding
    physical frequencies.  Band lengths may differ because every downstream
    transform compresses a band to exactly one scalar.
    """
    value = np.asarray(signal, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"signal must be [C,S], got {value.shape}")
    window = int(round(float(window_seconds) * float(rate)))
    hop = int(round(float(hop_seconds) * float(rate)))
    starts = window_starts(value.shape[-1], window, hop)
    if not starts.size:
        raise ValueError("Signal is shorter than one native-spectrum window")

    # sliding_window_view avoids copying the full trial before scipy applies
    # the Hann taper.  Welch has one segment here and is therefore the same
    # native-grid modified periodogram used by the frozen one-second protocol.
    framed = np.lib.stride_tricks.sliding_window_view(value, window, axis=-1)
    framed = np.moveaxis(framed[:, starts, :], 1, 0)
    frequencies, psd = welch(
        framed,
        fs=float(rate),
        window="hann",
        nperseg=window,
        noverlap=0,
        nfft=window,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    frequency_grid = native_frequency_grid(rate, window_seconds, bands)
    distributions: list[np.ndarray] = []
    for (name, limits), expected in zip(bands.items(), frequency_grid, strict=True):
        low, high = map(float, limits)
        selected = (frequencies >= low) & (frequencies < high)
        actual = np.asarray(frequencies[selected], dtype=np.float32)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"Unexpected native frequency grid for {name}: {actual}")
        band_psd = np.asarray(psd[..., selected], dtype=np.float32)
        denominator = band_psd.sum(axis=-1, keepdims=True, dtype=np.float32)
        if np.any(denominator <= epsilon) or not np.isfinite(denominator).all():
            raise FloatingPointError(f"Near-zero or invalid native spectral power in {name}")
        distribution = np.ascontiguousarray(band_psd / denominator, dtype=np.float32)
        if not np.isfinite(distribution).all():
            raise FloatingPointError(f"Native spectral distribution is non-finite in {name}")
        distributions.append(distribution)
    return distributions, frequency_grid


def _validate_native_inputs(
    distributions: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
) -> tuple[int, int]:
    if not distributions or len(distributions) != len(references):
        raise ValueError("Native distributions and references must have the same non-zero band count")
    time = int(np.asarray(distributions[0]).shape[0])
    channels = int(np.asarray(distributions[0]).shape[1])
    for band, (distribution, reference) in enumerate(zip(distributions, references, strict=True)):
        p = np.asarray(distribution)
        q = np.asarray(reference)
        if p.ndim != 3 or q.ndim != 2 or p.shape[1:] != q.shape:
            raise ValueError(f"Band {band} expected P=[T,C,F], Q=[C,F], got {p.shape}, {q.shape}")
        if p.shape[:2] != (time, channels):
            raise ValueError("All native bands must share time and channel dimensions")
    return time, channels


def transform_native_sqrt_jsd(
    distributions: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return unsigned ``sqrt(JSD)`` with one scalar per channel-band."""
    time, channels = _validate_native_inputs(distributions, references)
    values: list[np.ndarray] = []
    for distribution, reference in zip(distributions, references, strict=True):
        p = normalize_histograms(distribution, epsilon)
        q = normalize_histograms(reference, epsilon)
        midpoint = 0.5 * (p + q[None])
        divergence = 0.5 * (p * (np.log(p) - np.log(midpoint))).sum(axis=-1)
        divergence += 0.5 * (q * (np.log(q) - np.log(midpoint))).sum(axis=-1)
        values.append(np.sqrt(np.maximum(divergence, 0.0)).astype(np.float32))
    result = np.stack(values, axis=-1).reshape(time, channels * len(values)).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Native sqrt-JSD produced non-finite values")
    return result


def transform_native_wasserstein1(
    distributions: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    frequencies: Sequence[np.ndarray],
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return diameter-normalized one-dimensional Wasserstein-1 distances.

    The discrete CDF formula uses the actual frequency spacing.  Dividing by
    the support diameter bounds every band result in ``[0, 1]`` and makes the
    five physical frequency ranges comparable before source-only z-scoring.
    """
    time, channels = _validate_native_inputs(distributions, references)
    if len(frequencies) != len(distributions):
        raise ValueError("One native frequency grid is required per band")
    values: list[np.ndarray] = []
    for band, (distribution, reference, grid) in enumerate(
        zip(distributions, references, frequencies, strict=True)
    ):
        p = normalize_histograms(distribution, epsilon)
        q = normalize_histograms(reference, epsilon)
        support = np.asarray(grid, dtype=np.float32)
        if support.ndim != 1 or support.size != p.shape[-1] or support.size < 2:
            raise ValueError(f"Invalid frequency support for band {band}: {support.shape}")
        spacing = np.diff(support)
        if np.any(spacing <= 0):
            raise ValueError(f"Frequency support must be strictly increasing for band {band}")
        cdf_difference = np.cumsum(p - q[None], axis=-1)[..., :-1]
        distance = (np.abs(cdf_difference) * spacing).sum(axis=-1)
        diameter = float(support[-1] - support[0])
        values.append(np.asarray(distance / diameter, dtype=np.float32))
    result = np.stack(values, axis=-1).reshape(time, channels * len(values)).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result < -1e-7) or np.any(result > 1.0 + 1e-5):
        raise FloatingPointError("Normalized native Wasserstein-1 is outside its finite [0,1] range")
    return result


def fisher_rao_log_map(
    distribution: np.ndarray,
    reference: np.ndarray,
    epsilon: float = 1e-7,
) -> np.ndarray:
    """Map ``P`` to the Fisher-Rao tangent space at ``Q``.

    ``distribution`` is ``[T,C,F]`` and ``reference`` is ``[C,F]``.  The
    standard isometric embedding ``p -> 2*sqrt(p)`` is used, so the geodesic
    norm is ``2*arccos(sum(sqrt(p*q)))``.  Returned ambient tangent vectors
    have the same shape and are orthogonal to ``sqrt(reference)`` up to
    floating-point error.
    """
    p = normalize_histograms(distribution)
    q = normalize_histograms(reference)
    if p.ndim != 3 or q.ndim != 2 or p.shape[1:] != q.shape:
        raise ValueError(f"Expected P=[T,C,F], Q=[C,F], got {p.shape}, {q.shape}")
    root_p = np.sqrt(p).astype(np.float32)
    root_q = np.sqrt(q).astype(np.float32)
    cosine = np.clip((root_p * root_q[None]).sum(axis=-1), 0.0, 1.0)
    angle = np.arccos(cosine).astype(np.float32)
    sine = np.sin(angle).astype(np.float32)
    scale = np.ones_like(angle, dtype=np.float32)
    active = angle > epsilon
    scale[active] = angle[active] / np.maximum(sine[active], epsilon)
    tangent = 2.0 * scale[..., None] * (root_p - cosine[..., None] * root_q[None])
    tangent = np.asarray(tangent, dtype=np.float32)
    if not np.isfinite(tangent).all():
        raise FloatingPointError("Fisher-Rao log map produced non-finite values")
    return tangent


def transform_native_fisher_rao_pca(
    distributions: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    tangent_means: Sequence[np.ndarray],
    components: Sequence[np.ndarray],
) -> np.ndarray:
    """Project each native band onto its source-fitted first tangent PC."""
    time, channels = _validate_native_inputs(distributions, references)
    if not (
        len(tangent_means) == len(components) == len(distributions)
    ):
        raise ValueError("Fisher-Rao PCA state must contain one mean and component per band")
    values: list[np.ndarray] = []
    for band, (distribution, reference, mean, component) in enumerate(
        zip(distributions, references, tangent_means, components, strict=True)
    ):
        tangent = fisher_rao_log_map(distribution, reference)
        center = np.asarray(mean, dtype=np.float32)
        axis = np.asarray(component, dtype=np.float32)
        if center.shape != reference.shape or axis.shape != reference.shape:
            raise ValueError(f"Invalid Fisher-Rao PCA state shape for band {band}")
        values.append(np.sum((tangent - center[None]) * axis[None], axis=-1, dtype=np.float32))
    result = np.stack(values, axis=-1).reshape(time, channels * len(values)).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("Native Fisher-Rao PCA coordinate produced non-finite values")
    return result
