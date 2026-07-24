from __future__ import annotations

import unittest

import numpy as np
import torch

from cmrd.features.str_jsd import CONDITIONS, build_condition_features, reference_quality, response_starts
from cmrd.models.str_jsd import STRJSDHCBT


def _synthetic(seed: int = 7):
    rng = np.random.default_rng(seed)
    de = rng.normal(2.0, 0.2, size=(28, 30, 150)).astype(np.float32)
    spectra = []
    for size in (3, 4, 6, 16, 17):
        value = rng.gamma(2.0, 1.0, size=(28, 30, 30, size)).astype(np.float32)
        value /= value.sum(axis=-1, keepdims=True)
        spectra.append(value)
    return de, spectra


class STRJSDTests(unittest.TestCase):
    def test_registered_temporal_lengths_and_condition_shapes(self):
        de, spectra = _synthetic()
        self.assertEqual(response_starts(5).tolist(), list(range(5, 26)))
        self.assertEqual(response_starts(1).tolist(), list(range(5, 30)))
        for condition in CONDITIONS:
            gate = [1.0] * 5 if condition.startswith("C6") else None
            features = build_condition_features(de, spectra, condition, 5, gate_scales=gate)
            self.assertEqual(features.shape, (28, 21, 30 * (3 * 5 + 46)))
            self.assertTrue(np.isfinite(features).all())

    def test_signed_pointwise_field_exactly_recovers_scalar_jsd(self):
        de, spectra = _synthetic()
        c1 = build_condition_features(de, spectra, "C1_absolute_de_scalar_jsd", 5)
        c3, audit = build_condition_features(
            de, spectra, "C3_absolute_de_signed_pointwise_jsd", 5, return_audit=True
        )
        c1 = c1.reshape(28, 21, 30, 61)
        c3 = c3.reshape(28, 21, 30, 61)
        offset = 0
        for size in (3, 4, 6, 16, 17):
            scalar = c1[..., offset + 2]
            signed = c3[..., offset + 3 : offset + 3 + size]
            self.assertTrue(np.allclose(np.square(signed).sum(axis=-1), scalar, atol=2e-6))
            offset += size + 3
        self.assertLess(audit["maximum_signed_jsd_invariant_error"], 2e-6)

    def test_early_reference_is_same_trial_and_changes_only_that_trial(self):
        de, spectra = _synthetic()
        original = build_condition_features(
            de, spectra, "C4_absolute_de_delta_de_signed_pointwise_jsd", 5
        )
        changed_de = de.copy()
        changed_spectra = [value.copy() for value in spectra]
        changed_de[0, :5] += 0.8
        changed_spectra[0][0, :5] = changed_spectra[0][0, :5, :, ::-1]
        changed = build_condition_features(
            changed_de, changed_spectra, "C4_absolute_de_delta_de_signed_pointwise_jsd", 5
        )
        self.assertFalse(np.allclose(original[0], changed[0]))
        self.assertTrue(np.array_equal(original[1:], changed[1:]))

    def test_c0_ignores_reference_and_c6_gate_only_attenuates_relative_slots(self):
        de, spectra = _synthetic()
        c0 = build_condition_features(de, spectra, "C0_absolute_de", 5)
        changed_de = de.copy()
        changed_de[:, :5] += 1.0
        changed_c0 = build_condition_features(changed_de, spectra, "C0_absolute_de", 5)
        self.assertTrue(np.allclose(c0, changed_c0))
        c4 = build_condition_features(de, spectra, "C4_absolute_de_delta_de_signed_pointwise_jsd", 5)
        c6 = build_condition_features(
            de, spectra, "C6_c4_reference_quality_gate", 5, gate_scales=[0.1] * 5
        )
        shaped_c4 = c4.reshape(28, 21, 30, 61)
        shaped_c6 = c6.reshape(28, 21, 30, 61)
        offset = 0
        for size in (3, 4, 6, 16, 17):
            self.assertTrue(np.allclose(shaped_c4[..., offset], shaped_c6[..., offset]))
            self.assertTrue(np.all(np.abs(shaped_c6[..., offset + 1 : offset + 3 + size]) <= np.abs(shaped_c4[..., offset + 1 : offset + 3 + size]) + 1e-7))
            offset += 3 + size

    def test_reference_quality_is_nonnegative_and_trial_local(self):
        de, spectra = _synthetic()
        quality = reference_quality(de, spectra)
        self.assertEqual(quality.shape, (28, 30, 5))
        self.assertTrue(np.isfinite(quality).all())
        self.assertTrue(np.all(quality >= 0))

    def test_capacity_matched_model_forward_and_backward(self):
        model = STRJSDHCBT(
            channels=30, band_sizes=[6, 7, 9, 19, 20], classes=9,
            max_length=21, d_model=16, heads=4, layers=1,
            feedforward=32, dropout=0.0,
        )
        data = torch.randn(2, 21, 1830)
        mask = torch.ones(2, 21, dtype=torch.bool)
        logits = model(data, mask)
        self.assertEqual(logits.shape, (2, 9))
        logits.sum().backward()
        # VectorBandHCBT intentionally bypasses the scalar-only value embedding
        # retained by its shared backbone; all active encoders/classifier must
        # nevertheless receive gradients.
        self.assertTrue(all(parameter.grad is not None for parameter in model.backbone.band_encoders.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in model.classifier.parameters()))


if __name__ == "__main__":
    unittest.main()
