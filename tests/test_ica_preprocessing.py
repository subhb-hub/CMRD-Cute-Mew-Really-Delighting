from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from preprocess_seediv_de_rjsd_ica import (
    _cleaning_signature_payload,
    _output_family,
    _signature,
    _spectral_nfft,
    extract_de_and_phist,
)


class IcaPreprocessingTests(unittest.TestCase):
    def test_four_second_window_with_one_second_hop(self) -> None:
        rng = np.random.default_rng(31)
        signal = rng.normal(size=(2, 2000)).astype(np.float32)
        de, histogram = extract_de_and_phist(signal, window_seconds=4.0, hop_seconds=1.0)
        self.assertEqual(de.shape, (7, 2, 5))
        self.assertEqual(histogram.shape, (7, 2, 5, 32))
        np.testing.assert_allclose(histogram.sum(axis=-1), 1.0, atol=1e-5)
        self.assertEqual(_spectral_nfft(4.0), 1024)

    def test_output_names_preserve_old_default_and_add_four_second_family(self) -> None:
        self.assertEqual(_output_family(1.0, 0.5), "de_rjsd_ica_1s_hop05")
        self.assertEqual(_output_family(4.0, 1.0), "de_rjsd_ica_4s_hop1")

    def test_cleaning_signature_does_not_depend_on_window(self) -> None:
        common = {
            "dataset": "SEED-IV",
            "raw_dir": "raw",
            "channels": 62,
            "channel_names": ["Cz"],
            "sampling_rate": 200.0,
            "ica": {"random_state": 97},
            "mne_version": "test",
        }
        one_second = {**common, "window_seconds": 1.0, "hop_seconds": 0.5, "welch": {"nfft": 512}}
        four_seconds = {**common, "window_seconds": 4.0, "hop_seconds": 1.0, "welch": {"nfft": 1024}}
        self.assertEqual(
            _signature(_cleaning_signature_payload(one_second)),
            _signature(_cleaning_signature_payload(four_seconds)),
        )


if __name__ == "__main__":
    unittest.main()
