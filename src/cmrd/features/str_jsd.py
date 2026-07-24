from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CONDITIONS = (
    "C0_absolute_de",
    "C1_absolute_de_scalar_jsd",
    "C2_absolute_de_unsigned_pointwise_jsd",
    "C3_absolute_de_signed_pointwise_jsd",
    "C4_absolute_de_delta_de_signed_pointwise_jsd",
    "C5_absolute_de_pointwise_log_ratio",
    "C6_c4_reference_quality_gate",
)


def response_starts(response_average_windows: int) -> np.ndarray:
    """Return zero-based response starts after the five-window early reference."""
    width = int(response_average_windows)
    if width not in {1, 5}:
        raise ValueError("STR-JSD supports only the registered 1-window and 5-window settings")
    return np.arange(5, 30 - width + 1, dtype=np.int64)


def _validate(
    de: np.ndarray,
    spectra: Sequence[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    entropy = np.asarray(de, dtype=np.float32)
    if entropy.shape != (28, 30, 150):
        raise ValueError(f"Expected FACED DE [28,30,150], got {entropy.shape}")
    entropy = entropy.reshape(28, 30, 30, 5)
    native = [np.asarray(value, dtype=np.float32) for value in spectra]
    if len(native) != 5:
        raise ValueError("FACED STR-JSD requires five native frequency bands")
    for band, value in enumerate(native):
        if value.shape[:3] != (28, 30, 30) or value.ndim != 4:
            raise ValueError(f"Invalid native band {band}: {value.shape}")
        sums = value.sum(axis=-1)
        if not np.allclose(sums, 1.0, atol=2e-5):
            raise ValueError(f"Native band {band} is not normalized within band")
    if not np.isfinite(entropy).all() or any(not np.isfinite(value).all() for value in native):
        raise FloatingPointError("STR-JSD inputs contain NaN/Inf")
    return entropy, native


def energy_calibrated_spectra(
    de: np.ndarray,
    spectra: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Recover a relative band-energy scale while preserving native PSD shape.

    Cached DE is ``0.5*log(2*pi*e*variance)``. Therefore ``exp(2*DE)`` is
    proportional to band energy. The common constant cancels from all ratios
    and within-band normalizations used below.
    """
    entropy, native = _validate(de, spectra)
    output = []
    for band, distribution in enumerate(native):
        energy = np.exp(2.0 * entropy[..., band : band + 1], dtype=np.float32)
        output.append(np.ascontiguousarray(energy * distribution, dtype=np.float32))
    return output


def reference_quality(
    de: np.ndarray,
    spectra: Sequence[np.ndarray],
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return early-reference instability ``u`` as [video,channel,band]."""
    calibrated = energy_calibrated_spectra(de, spectra)
    values = [
        np.var(np.log(value[:, :5] + epsilon), axis=1, dtype=np.float32).mean(axis=-1)
        for value in calibrated
    ]
    quality = np.stack(values, axis=-1).astype(np.float32)
    if not np.isfinite(quality).all() or np.any(quality < 0):
        raise FloatingPointError("Invalid early-reference quality statistic")
    return quality


def build_condition_features(
    de: np.ndarray,
    spectra: Sequence[np.ndarray],
    condition: str,
    response_average_windows: int,
    *,
    gate_scales: Sequence[float] | None = None,
    epsilon: float = 1e-12,
    return_audit: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Build one capacity-matched STR-JSD condition as [video,time,feature].

    Every channel-band token has the same registered slots in every condition:
    ``[absolute_DE, delta_log_energy, scalar_JSD, native_vector...]``. Inactive
    slots are exactly zero, so all ablations use an identical model and
    parameter count.
    """
    if condition not in CONDITIONS:
        raise KeyError(condition)
    width = int(response_average_windows)
    starts = response_starts(width)
    calibrated = energy_calibrated_spectra(de, spectra)
    quality = reference_quality(de, spectra, epsilon)
    scales = None
    if condition == "C6_c4_reference_quality_gate":
        if gate_scales is None or len(gate_scales) != 5:
            raise ValueError("C6 requires five source-fitted reference-quality scales")
        scales = np.asarray(gate_scales, dtype=np.float32)
        if np.any(scales <= 0) or not np.isfinite(scales).all():
            raise ValueError("C6 gate scales must be finite and positive")

    blocks: list[np.ndarray] = []
    maximum_invariant_error = 0.0
    for band, raw in enumerate(calibrated):
        reference_raw = raw[:, :5].mean(axis=1, dtype=np.float32)
        response_raw = np.stack(
            [raw[:, start : start + width].mean(axis=1, dtype=np.float32) for start in starts],
            axis=1,
        )
        q_energy = reference_raw.sum(axis=-1, keepdims=True, dtype=np.float32)
        p_energy = response_raw.sum(axis=-1, keepdims=True, dtype=np.float32)
        q = reference_raw / np.maximum(q_energy, epsilon)
        p = response_raw / np.maximum(p_energy, epsilon)
        q = q[:, None]
        midpoint = 0.5 * (p + q)
        contribution = 0.5 * (
            p * (np.log(p + epsilon) - np.log(midpoint + epsilon))
            + q * (np.log(q + epsilon) - np.log(midpoint + epsilon))
        )
        contribution = np.maximum(contribution, 0.0).astype(np.float32)
        unsigned = np.sqrt(contribution, dtype=np.float32)
        signed = np.sign(p - q).astype(np.float32) * unsigned
        invariant_error = np.max(np.abs(np.sum(np.square(signed), axis=-1) - contribution.sum(axis=-1)))
        maximum_invariant_error = max(maximum_invariant_error, float(invariant_error))

        absolute_de = 0.5 * np.log(np.maximum(p_energy, epsilon))
        delta_log_energy = np.log(np.maximum(p_energy, epsilon)) - np.log(np.maximum(q_energy[:, None], epsilon))
        scalar_jsd = contribution.sum(axis=-1, keepdims=True, dtype=np.float32)
        log_ratio = np.log(response_raw + epsilon) - np.log(reference_raw[:, None] + epsilon)

        shape = np.zeros_like(p, dtype=np.float32)
        delta_slot = np.zeros_like(delta_log_energy, dtype=np.float32)
        scalar_slot = np.zeros_like(scalar_jsd, dtype=np.float32)
        if condition == "C1_absolute_de_scalar_jsd":
            scalar_slot = scalar_jsd
        elif condition == "C2_absolute_de_unsigned_pointwise_jsd":
            shape = unsigned
        elif condition == "C3_absolute_de_signed_pointwise_jsd":
            shape = signed
        elif condition in {"C4_absolute_de_delta_de_signed_pointwise_jsd", "C6_c4_reference_quality_gate"}:
            delta_slot = delta_log_energy
            shape = signed
        elif condition == "C5_absolute_de_pointwise_log_ratio":
            shape = log_ratio.astype(np.float32)

        if scales is not None:
            gate = np.exp(-quality[..., band] / scales[band], dtype=np.float32)[:, None, :, None]
            delta_slot = delta_slot * gate
            shape = shape * gate
        block = np.concatenate([absolute_de, delta_slot, scalar_slot, shape], axis=-1)
        blocks.append(np.ascontiguousarray(block, dtype=np.float32))

    structured = np.concatenate(blocks, axis=-1)  # [V,T,C,sum(3+F_b)]
    output = np.ascontiguousarray(structured.reshape(28, len(starts), -1), dtype=np.float32)
    if not np.isfinite(output).all():
        raise FloatingPointError(f"Non-finite STR-JSD features for {condition}")
    if return_audit:
        return output, {
            "maximum_signed_jsd_invariant_error": maximum_invariant_error,
            "minimum_feature": float(output.min()),
            "maximum_feature": float(output.max()),
        }
    return output
