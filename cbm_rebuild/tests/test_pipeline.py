from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loso_split import source_train_validation_split
from src.models.plain_transformer import PlainTransformer
from src.preprocessing.de_extraction import DEFAULT_BANDS, extract_de_features
from src.preprocessing.normalization import fit_source_normalizer
from src.preprocessing.padding import pad_trials


class PipelineTests(unittest.TestCase):
    def test_de_shape_is_windows_by_310(self) -> None:
        rng = np.random.default_rng(7)
        signal = rng.normal(size=(62, 400)).astype(np.float32)
        features = extract_de_features(signal, 200, 1.0, 1.0, DEFAULT_BANDS)
        self.assertEqual(features.shape, (2, 310))
        self.assertTrue(np.isfinite(features).all())

    def test_padding_and_mask(self) -> None:
        data, mask, lengths = pad_trials(
            [np.ones((2, 310), dtype=np.float32), np.ones((4, 310), dtype=np.float32)]
        )
        self.assertEqual(data.shape, (2, 4, 310))
        np.testing.assert_array_equal(lengths, [2, 4])
        np.testing.assert_array_equal(mask[0], [True, True, False, False])
        self.assertTrue(np.all(data[0, 2:] == 0))

    def test_normalizer_ignores_target_and_padding(self) -> None:
        data = np.zeros((3, 3, 2), dtype=np.float32)
        mask = np.asarray([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool)
        data[0, :2] = [[1, 3], [3, 5]]
        data[0, 2] = 10_000
        data[1, 0] = [5, 7]
        data[2] = 1_000_000
        mean, _ = fit_source_normalizer(data, mask, np.asarray([0, 1]))
        np.testing.assert_allclose(mean, [3, 5])

    def test_loso_never_places_target_in_source(self) -> None:
        subjects = np.repeat(np.arange(1, 5), 12)
        labels = np.tile(np.arange(3), 16)
        train, validation, test = source_train_validation_split(subjects, labels, 3, 0.2, 42)
        self.assertTrue(np.all(subjects[test] == 3))
        self.assertTrue(np.all(subjects[train] != 3))
        self.assertTrue(np.all(subjects[validation] != 3))

    def test_transformer_padding_values_do_not_change_logits(self) -> None:
        torch.manual_seed(3)
        model = PlainTransformer(6, 3, max_length=5, d_model=8, nhead=2, num_layers=1, dim_feedforward=16, dropout=0.0)
        model.eval()
        mask = torch.tensor([[True, True, False, False, False]])
        first = torch.randn(1, 5, 6)
        second = first.clone()
        second[:, 2:] = torch.randn_like(second[:, 2:]) * 10_000
        with torch.no_grad():
            logits_first = model(first, mask)
            logits_second = model(second, mask)
        torch.testing.assert_close(logits_first, logits_second, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
