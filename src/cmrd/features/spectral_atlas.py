from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import helmert
from scipy.signal import welch

from cmrd.features.rd import native_frequency_grid, normalize_histograms
from cmrd.features.signal import window_starts


EPSILON = 1e-12


@dataclass(frozen=True)
class LandmarkBandState:
    reference: np.ndarray  # [C,F]
    anchors: np.ndarray  # [C,K,F]
    anchor_to_reference_jsd: np.ndarray  # [C,K]
    whitening: np.ndarray  # [C,K,K]
    eigenvalues: np.ndarray  # [C,K]


@dataclass(frozen=True)
class ProjectionState:
    mean: np.ndarray  # [C,D]
    components: np.ndarray  # [C,D,R]


def extract_native_shape_power(
    signal: np.ndarray,
    rate: float,
    window_seconds: float,
    hop_seconds: float,
    bands: Mapping[str, Sequence[float]],
    epsilon: float = EPSILON,
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """Return native-grid spectral shapes and log band powers.

    Shapes are one ``[T,C,F_b]`` probability array per band.  Log power is
    ``[T,C,B]`` and, together with a shape, reconstructs the corresponding
    native-grid PSD up to floating-point error.
    """
    value = np.asarray(signal, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"signal must be [C,S], got {value.shape}")
    window = int(round(float(window_seconds) * float(rate)))
    hop = int(round(float(hop_seconds) * float(rate)))
    starts = window_starts(value.shape[-1], window, hop)
    if not starts.size:
        raise ValueError("Signal is shorter than one spectral-atlas window")
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
    grids = native_frequency_grid(rate, window_seconds, bands)
    shapes: list[np.ndarray] = []
    powers: list[np.ndarray] = []
    for (name, limits), expected in zip(bands.items(), grids, strict=True):
        low, high = map(float, limits)
        selected = (frequencies >= low) & (frequencies < high)
        actual = np.asarray(frequencies[selected], dtype=np.float32)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"Unexpected native frequency grid for {name}: {actual}")
        band_psd = np.asarray(psd[..., selected], dtype=np.float32)
        total = band_psd.sum(axis=-1, dtype=np.float32)
        if np.any(total <= epsilon) or not np.isfinite(total).all():
            raise FloatingPointError(f"Near-zero or invalid native spectral power in {name}")
        shapes.append(np.ascontiguousarray(band_psd / total[..., None], dtype=np.float32))
        powers.append(np.log(np.maximum(total, epsilon)).astype(np.float32))
    log_power = np.stack(powers, axis=-1).astype(np.float32)
    return shapes, log_power, grids


def _jsd_to_reference(
    distributions: np.ndarray,
    reference: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    p = normalize_histograms(distributions, epsilon)
    q = normalize_histograms(reference, epsilon)
    if p.ndim != 3 or q.ndim != 2 or p.shape[1:] != q.shape:
        raise ValueError(f"Expected P=[N,C,F], Q=[C,F], got {p.shape}, {q.shape}")
    midpoint = 0.5 * (p + q[None])
    value = 0.5 * (p * (np.log(p) - np.log(midpoint))).sum(axis=-1)
    value += 0.5 * (q[None] * (np.log(q[None]) - np.log(midpoint))).sum(axis=-1)
    return np.maximum(value, 0.0).astype(np.float32)


def _jsd_to_anchors(
    distributions: np.ndarray,
    anchors: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    p = normalize_histograms(distributions, epsilon)
    a = normalize_histograms(anchors, epsilon)
    if p.ndim != 3 or a.ndim != 3 or p.shape[1] != a.shape[0] or p.shape[2] != a.shape[2]:
        raise ValueError(f"Expected P=[N,C,F], A=[C,K,F], got {p.shape}, {a.shape}")
    midpoint = 0.5 * (p[:, :, None, :] + a[None])
    value = 0.5 * (
        p[:, :, None, :] * (np.log(p[:, :, None, :]) - np.log(midpoint))
    ).sum(axis=-1)
    value += 0.5 * (
        a[None] * (np.log(a[None]) - np.log(midpoint))
    ).sum(axis=-1)
    return np.maximum(value, 0.0).astype(np.float32)


def fit_landmark_band(
    candidates: np.ndarray,
    cap: int,
    *,
    ridge: float = 1e-6,
    eigenvalue_tolerance: float = 1e-7,
) -> LandmarkBandState:
    """Fit balanced farthest-point JSD landmarks for one frequency band."""
    values = normalize_histograms(candidates)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError(f"candidates must be [N,C,F] with N>=2, got {values.shape}")
    if cap < 1 or ridge <= 0 or eigenvalue_tolerance <= 0:
        raise ValueError("cap, ridge, and eigenvalue_tolerance must be positive")
    coordinates = min(int(cap), int(values.shape[-1]) - 1)
    reference = normalize_histograms(values.mean(axis=0, dtype=np.float64)).astype(np.float32)
    minimum = _jsd_to_reference(values, reference)
    anchors: list[np.ndarray] = []
    channels = values.shape[1]
    channel_indices = np.arange(channels)
    for _ in range(coordinates):
        selected = np.argmax(minimum, axis=0)
        anchor = values[selected, channel_indices].astype(np.float32)
        anchors.append(anchor)
        minimum = np.minimum(minimum, _jsd_to_reference(values, anchor))
    anchor_array = np.stack(anchors, axis=1)  # [C,K,F]
    anchor_to_reference = _jsd_to_reference(
        np.moveaxis(anchor_array, 1, 0), reference
    ).T.astype(np.float32)

    gram = np.empty((channels, coordinates, coordinates), dtype=np.float64)
    for left in range(coordinates):
        pairwise = _jsd_to_anchors(anchor_array[:, left][None], anchor_array)
        gram[:, left, :] = 0.5 * (
            anchor_to_reference[:, left, None]
            + anchor_to_reference
            - pairwise[0]
        )
    gram = 0.5 * (gram + np.swapaxes(gram, -1, -2))
    whitening = np.empty_like(gram, dtype=np.float32)
    eigenvalues = np.empty((channels, coordinates), dtype=np.float32)
    for channel in range(channels):
        values_eig, vectors = np.linalg.eigh(gram[channel])
        order = np.argsort(values_eig)[::-1]
        values_eig = np.maximum(values_eig[order], 0.0)
        vectors = vectors[:, order]
        for column in range(vectors.shape[1]):
            pivot = int(np.argmax(np.abs(vectors[:, column])))
            if vectors[pivot, column] < 0:
                vectors[:, column] *= -1.0
        scale = np.sqrt(np.maximum(values_eig, ridge))
        whitening[channel] = (vectors / scale[None]).astype(np.float32)
        threshold = max(float(values_eig[0]) * eigenvalue_tolerance, ridge)
        eigenvalues[channel] = np.where(values_eig >= threshold, values_eig, 0.0)
    return LandmarkBandState(
        reference=reference,
        anchors=anchor_array,
        anchor_to_reference_jsd=anchor_to_reference,
        whitening=whitening,
        eigenvalues=eigenvalues,
    )


def raw_landmark_coordinates(
    distributions: np.ndarray,
    state: LandmarkBandState,
) -> np.ndarray:
    distance_to_reference = _jsd_to_reference(distributions, state.reference)
    distance_to_anchors = _jsd_to_anchors(distributions, state.anchors)
    return (distance_to_anchors - distance_to_reference[..., None]).astype(np.float32)


def nystrom_landmark_coordinates(
    distributions: np.ndarray,
    state: LandmarkBandState,
) -> np.ndarray:
    distance_to_reference = _jsd_to_reference(distributions, state.reference)
    distance_to_anchors = _jsd_to_anchors(distributions, state.anchors)
    inner_products = 0.5 * (
        distance_to_reference[..., None]
        + state.anchor_to_reference_jsd[None]
        - distance_to_anchors
    )
    return np.einsum("nck,ckr->ncr", inner_products, state.whitening).astype(np.float32)


def ilr_basis(parts: int) -> np.ndarray:
    if parts < 2:
        raise ValueError("An ILR composition requires at least two parts")
    return np.asarray(helmert(parts, full=False).T, dtype=np.float32)


def ilr_coordinates(distributions: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    value = normalize_histograms(distributions, epsilon)
    if value.ndim != 3:
        raise ValueError(f"ILR input must be [N,C,F], got {value.shape}")
    return np.einsum("ncf,fd->ncd", np.log(value), ilr_basis(value.shape[-1])).astype(np.float32)


def full_ilr_power(
    distributions: Sequence[np.ndarray],
    log_power: np.ndarray,
) -> np.ndarray:
    _validate_shapes(distributions, log_power)
    parts = [ilr_coordinates(value) for value in distributions]
    parts.append(np.asarray(log_power, dtype=np.float32))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def full_log_psd(
    distributions: Sequence[np.ndarray],
    log_power: np.ndarray,
    epsilon: float = EPSILON,
) -> np.ndarray:
    _validate_shapes(distributions, log_power)
    parts = [
        np.log(np.maximum(normalize_histograms(value, epsilon), epsilon))
        + log_power[..., band, None]
        for band, value in enumerate(distributions)
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def scalar_jsd_power(
    distributions: Sequence[np.ndarray],
    log_power: np.ndarray,
    states: Sequence[LandmarkBandState],
) -> np.ndarray:
    _validate_state_shapes(distributions, log_power, states)
    radii = [
        np.sqrt(_jsd_to_reference(value, state.reference))[..., None]
        for value, state in zip(distributions, states, strict=True)
    ]
    return np.concatenate([*radii, log_power], axis=-1).astype(np.float32)


def landmark_power(
    distributions: Sequence[np.ndarray],
    log_power: np.ndarray,
    states: Sequence[LandmarkBandState],
    *,
    orthogonalized: bool,
) -> np.ndarray:
    _validate_state_shapes(distributions, log_power, states)
    transform = nystrom_landmark_coordinates if orthogonalized else raw_landmark_coordinates
    parts = [transform(value, state) for value, state in zip(distributions, states, strict=True)]
    return np.concatenate([*parts, log_power], axis=-1).astype(np.float32)


def fit_channel_pca(values: np.ndarray, output_dim: int) -> ProjectionState:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 3 or not 1 <= output_dim <= data.shape[-1]:
        raise ValueError(f"PCA input/output mismatch: {data.shape}, output_dim={output_dim}")
    mean = data.mean(axis=0, dtype=np.float64)
    centered = data.astype(np.float64) - mean[None]
    covariance = np.einsum("ncd,nce->cde", centered, centered) / max(data.shape[0] - 1, 1)
    components = np.empty((data.shape[1], data.shape[2], output_dim), dtype=np.float32)
    for channel in range(data.shape[1]):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance[channel])
        axes = eigenvectors[:, np.argsort(eigenvalues)[::-1][:output_dim]]
        for column in range(axes.shape[1]):
            pivot = int(np.argmax(np.abs(axes[:, column])))
            if axes[pivot, column] < 0:
                axes[:, column] *= -1.0
        components[channel] = axes.astype(np.float32)
    return ProjectionState(mean=mean.astype(np.float32), components=components)


def fit_random_projection(
    channels: int,
    input_dim: int,
    output_dim: int,
    seed: int,
) -> ProjectionState:
    if channels < 1 or not 1 <= output_dim <= input_dim:
        raise ValueError("Invalid random projection dimensions")
    rng = np.random.default_rng(seed)
    components = np.empty((channels, input_dim, output_dim), dtype=np.float32)
    for channel in range(channels):
        matrix = rng.normal(size=(input_dim, output_dim))
        q, _ = np.linalg.qr(matrix, mode="reduced")
        components[channel] = q.astype(np.float32)
    return ProjectionState(
        mean=np.zeros((channels, input_dim), dtype=np.float32),
        components=components,
    )


def apply_projection(values: np.ndarray, state: ProjectionState) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 3 or data.shape[1:] != state.mean.shape:
        raise ValueError(f"Projection input/state mismatch: {data.shape}, {state.mean.shape}")
    return np.einsum("ncd,cdr->ncr", data - state.mean[None], state.components).astype(np.float32)


def capped_dimension(band_sizes: Sequence[int], cap: int, include_power: bool = True) -> int:
    if cap < 1 or any(int(size) < 2 for size in band_sizes):
        raise ValueError("cap must be positive and every band needs at least two bins")
    shape = sum(min(int(cap), int(size) - 1) for size in band_sizes)
    return shape + (len(band_sizes) if include_power else 0)


def full_dimension(band_sizes: Sequence[int]) -> int:
    if any(int(size) < 2 for size in band_sizes):
        raise ValueError("Every band needs at least two bins")
    return sum(int(size) for size in band_sizes)


def _validate_shapes(distributions: Sequence[np.ndarray], log_power: np.ndarray) -> tuple[int, int]:
    if not distributions:
        raise ValueError("At least one frequency band is required")
    power = np.asarray(log_power)
    first = np.asarray(distributions[0])
    if first.ndim != 3 or power.ndim != 3:
        raise ValueError("Distributions and power must be [N,C,F] and [N,C,B]")
    if power.shape != (first.shape[0], first.shape[1], len(distributions)):
        raise ValueError(f"Power shape mismatch: {power.shape}")
    for value in distributions:
        if np.asarray(value).ndim != 3 or np.asarray(value).shape[:2] != first.shape[:2]:
            raise ValueError("All distributions must share N and C")
    return first.shape[0], first.shape[1]


def _validate_state_shapes(
    distributions: Sequence[np.ndarray],
    log_power: np.ndarray,
    states: Sequence[LandmarkBandState],
) -> None:
    _validate_shapes(distributions, log_power)
    if len(distributions) != len(states):
        raise ValueError("One landmark state is required per band")
    for value, state in zip(distributions, states, strict=True):
        if np.asarray(value).shape[1:] != state.reference.shape:
            raise ValueError("Landmark state does not match distribution shape")
