from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmrd.config import ExperimentConfig, load_config
from cmrd.data.splits import subject_loso_split
from cmrd.features.de import extract_de
from cmrd.features.rd import extract_spectral_histograms, fit_reference, transform_rd
from cmrd.models import PlainTransformer
from cmrd.training.engine import fit_normalizer
from cmrd.data.records import TrialSample


class CoreTests(unittest.TestCase):
    def test_all_configs_and_override(self) -> None:
        for dataset in ("seed", "seediv"):
            for feature in ("de", "rd"):
                config = load_config(ROOT / "configs" / dataset / f"{feature}.yaml", ["training.learning_rate=0.0002"], feature)
                self.assertEqual(config.dataset, dataset)
                self.assertEqual(config.feature, feature)
                self.assertEqual(config.raw["training"]["learning_rate"], 0.0002)
                self.assertEqual(config.data_root, ROOT.parent / "Dataset")

    def test_signature_changes_with_preprocessing(self) -> None:
        original = load_config(ROOT / "configs" / "seed" / "de.yaml")
        changed = load_config(ROOT / "configs" / "seed" / "de.yaml", ["signal.window_seconds=2.0"])
        self.assertNotEqual(original.preprocessing_signature(), changed.preprocessing_signature())

    def test_de_shape_and_finite_values(self) -> None:
        rng = np.random.default_rng(2)
        signal = rng.normal(size=(62, 800)).astype(np.float32)
        bands = {"d": [1, 4], "t": [4, 8], "a": [8, 14], "b": [14, 31], "g": [31, 50]}
        result = extract_de(signal, 200, 1, 1, bands, 4)
        self.assertEqual(result.shape, (4, 310))
        self.assertTrue(np.isfinite(result).all())

    def test_rd_formula_shape_normalization_and_identity(self) -> None:
        rng = np.random.default_rng(3)
        signal = rng.normal(size=(2, 400)).astype(np.float32)
        bands = {"d": [1, 4], "t": [4, 8], "a": [8, 14], "b": [14, 31], "g": [31, 50]}
        histogram = extract_spectral_histograms(signal, 200, 1, 1, bands, 8, 256)
        self.assertEqual(histogram.shape, (2, 2, 5, 8))
        np.testing.assert_allclose(histogram.sum(axis=-1), 1.0, atol=1e-5)
        reference, count = fit_reference([histogram])
        self.assertEqual(count, 2)
        result = transform_rd(histogram, reference)
        self.assertEqual(result.shape, (2, 10))
        identity = transform_rd(reference[None], reference)
        np.testing.assert_allclose(identity, 0.0, atol=1e-7)

    def test_subject_split_has_no_leakage_and_is_reproducible(self) -> None:
        subjects = np.repeat(np.arange(1, 16), 3)
        first = subject_loso_split(subjects, 4, 2, 2026)
        second = subject_loso_split(subjects, 4, 2, 2026)
        self.assertEqual(first, second)
        self.assertEqual(len(first.train_subjects), 12)
        self.assertEqual(len(first.validation_subjects), 2)
        self.assertFalse(set(first.train_subjects) & set(first.validation_subjects))
        self.assertNotIn(4, first.train_subjects + first.validation_subjects)

    def test_normalizer_uses_only_supplied_source_trials(self) -> None:
        source = [TrialSample(np.asarray([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32), 0, 1, 1, 1, 0)]
        target = TrialSample(np.full((5, 2), 10_000, dtype=np.float32), 0, 2, 1, 1, 1)
        mean, std = fit_normalizer(source)
        np.testing.assert_allclose(mean, [2.0, 4.0])
        np.testing.assert_allclose(std, [1.0, 1.0])
        self.assertGreater(float(target.x.mean()), float(mean.mean()))

    def test_transformer_ignores_padding_values(self) -> None:
        torch.manual_seed(4)
        model = PlainTransformer(6, 3, 5, 8, 2, 1, 16, 0.0).eval()
        data = torch.randn(2, 5, 6)
        mask = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
        changed = data.clone()
        changed[~mask] = torch.randn_like(changed[~mask]) * 1000
        with torch.no_grad():
            first = model(data, mask)
            second = model(changed, mask)
        torch.testing.assert_close(first, second)

    def test_transformer_uses_legacy_wide_classifier_head(self) -> None:
        model = PlainTransformer(6, 3, 5, 8, 2, 1, 16, 0.2)
        self.assertIsInstance(model.classifier[0], torch.nn.Linear)
        self.assertEqual((model.classifier[0].in_features, model.classifier[0].out_features), (8, 32))
        self.assertIsInstance(model.classifier[1], torch.nn.ReLU)
        self.assertIsInstance(model.classifier[2], torch.nn.Dropout)
        self.assertIsInstance(model.classifier[3], torch.nn.Linear)
        self.assertEqual((model.classifier[3].in_features, model.classifier[3].out_features), (32, 3))


if __name__ == "__main__":
    unittest.main()
