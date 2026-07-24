from __future__ import annotations

import unittest

import numpy as np
import torch

from cmrd.faced_psd_jsd_experiment import (
    BAND_SIZES,
    _lr_factor,
    inner_cv_splits,
    smoke_split,
)
from cmrd.faced_psd_jsd_locked_test import fixed_epoch_from_inner_cv
from cmrd.models.faced_psd_jsd import (
    NativeBandChannelTemporalTransformer,
    NativeBandFlattenTemporalTransformer,
    PaddedCNNTemporalTransformer,
    parameter_count,
)


class FacedPSDJSDSplitTests(unittest.TestCase):
    def test_smoke_split_never_uses_outer_target(self) -> None:
        split = smoke_split(1, 42, 8, 4)
        self.assertEqual(len(split.train_subjects), 8)
        self.assertEqual(len(split.development_subjects), 4)
        self.assertFalse(set(split.train_subjects) & set(split.outer_target_subjects))
        self.assertFalse(set(split.development_subjects) & set(split.outer_target_subjects))

    def test_inner_cv_covers_outer_source_once_as_development(self) -> None:
        splits = inner_cv_splits(1, 42, 3)
        target = set(splits[0].outer_target_subjects)
        development = [subject for split in splits for subject in split.development_subjects]
        self.assertEqual(len(development), len(set(development)))
        self.assertEqual(set(development), set(range(123)) - target)
        for split in splits:
            self.assertFalse(set(split.train_subjects) & set(split.development_subjects))
            self.assertFalse((set(split.train_subjects) | set(split.development_subjects)) & target)


class FacedPSDJSDModelTests(unittest.TestCase):
    def test_small_models_forward_and_capacity(self) -> None:
        mask = torch.zeros(5, 17)
        for band, size in enumerate(BAND_SIZES):
            mask[band, :size] = 1
        cnn = PaddedCNNTemporalTransformer(frequency_mask=mask)
        native = NativeBandChannelTemporalTransformer()
        flatten = NativeBandFlattenTemporalTransformer()
        value = torch.randn(2, 30, 30, 5, 17)
        with torch.no_grad():
            self.assertEqual(tuple(cnn(value).shape), (2, 9))
            self.assertEqual(tuple(native(value).shape), (2, 9))
            self.assertEqual(tuple(flatten(value).shape), (2, 9))
        self.assertLess(parameter_count(cnn), 2_000_000)
        self.assertLess(parameter_count(native), 2_000_000)
        self.assertLess(parameter_count(flatten), 2_000_000)

    def test_native_band_model_ignores_padded_frequency_values(self) -> None:
        torch.manual_seed(3)
        model = NativeBandChannelTemporalTransformer(dropout=0.0).eval()
        value = torch.randn(2, 4, 30, 5, 17)
        changed = value.clone()
        for band, size in enumerate(BAND_SIZES):
            changed[..., band, size:] = torch.randn_like(changed[..., band, size:]) * 1000
        with torch.no_grad():
            original = model(value)
            modified = model(changed)
        torch.testing.assert_close(original, modified, rtol=0.0, atol=0.0)

    def test_native_flatten_model_ignores_padded_frequency_values(self) -> None:
        torch.manual_seed(4)
        model = NativeBandFlattenTemporalTransformer(dropout=0.0).eval()
        value = torch.randn(2, 4, 30, 5, 17)
        changed = value.clone()
        for band, size in enumerate(BAND_SIZES):
            changed[..., band, size:] = torch.randn_like(changed[..., band, size:]) * 1000
        with torch.no_grad():
            original = model(value)
            modified = model(changed)
        torch.testing.assert_close(original, modified, rtol=0.0, atol=0.0)

    def test_warmup_cosine_schedule_has_expected_boundaries(self) -> None:
        self.assertAlmostEqual(_lr_factor(0, 100, 10, 0.01), 0.1)
        self.assertAlmostEqual(_lr_factor(10, 100, 10, 0.01), 1.0)
        self.assertAlmostEqual(_lr_factor(100, 100, 10, 0.01), 0.01)

    def test_locked_epoch_is_source_only_inner_cv_median(self) -> None:
        self.assertEqual(fixed_epoch_from_inner_cv([23, 19, 18]), 19)
        with self.assertRaises(ValueError):
            fixed_epoch_from_inner_cv([])


if __name__ == "__main__":
    unittest.main()
