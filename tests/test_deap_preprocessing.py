from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmrd.config import load_config
from cmrd.data.deap import (
    DEAP_SUBJECTS,
    DEAP_TRIALS_PER_SUBJECT,
    _resample_to_target,
    deap_label,
    normalize_deap_event_codes,
    pair_deap_video_events,
    read_deap_ratings,
)


class DeapPreprocessingTests(unittest.TestCase):
    def test_deap_config_declares_formal_shape_and_protocol(self) -> None:
        config = load_config(ROOT / "configs" / "deap" / "rd.yaml", expected_feature="rd")
        self.assertEqual(config.dataset, "deap")
        self.assertEqual(config.raw["dataset"]["subjects"], 32)
        self.assertEqual(config.raw["dataset"]["channels"], 32)
        self.assertEqual(config.raw["dataset"]["label_target"], "quadrant")
        self.assertEqual(config.raw["signal"]["target_rate"], 200)
        self.assertEqual(config.raw["signal"]["hop_seconds"], 1.0)

    def test_label_mapping_uses_declared_threshold_and_quadrant_order(self) -> None:
        low_low = {"valence": 4.99, "arousal": 4.99}
        low_high = {"valence": 4.99, "arousal": 5.0}
        high_low = {"valence": 5.0, "arousal": 4.99}
        high_high = {"valence": 5.0, "arousal": 5.0}
        self.assertEqual([deap_label(row, "quadrant") for row in (low_low, low_high, high_low, high_high)], [0, 1, 2, 3])
        self.assertEqual(deap_label(high_low, "valence"), 1)
        self.assertEqual(deap_label(low_high, "arousal"), 1)

    def test_video_events_produce_exact_sixty_second_cuts(self) -> None:
        rows = []
        for trial in range(DEAP_TRIALS_PER_SUBJECT):
            start = 10_000 + trial * 40_000
            rows.extend(([start, 0, 4], [start + 30_786, 0, 5]))
        pairs = pair_deap_video_events(np.asarray(rows), 512.0)
        self.assertEqual(len(pairs), 40)
        self.assertTrue(all(stop - start == 60 * 512 for start, stop, _ in pairs))

    def test_extra_end_events_are_not_paired_by_position(self) -> None:
        rows = []
        for trial in range(DEAP_TRIALS_PER_SUBJECT):
            start = 10_000 + trial * 50_000
            rows.extend(([start, 0, 4], [start + 30_786, 0, 5]))
        rows.insert(25, [rows[24][0] + 29_696, 0, 5])
        pairs = pair_deap_video_events(np.asarray(sorted(rows, key=lambda row: row[0])), 512.0)
        self.assertEqual(len(pairs), 40)
        self.assertTrue(all(abs(observed - stop) < 100 for _, stop, observed in pairs))

    def test_biosemi_high_status_bits_are_removed(self) -> None:
        events = np.asarray([[100, 65281, 65284], [200, 65284, 65285]], dtype=np.int64)
        normalized = normalize_deap_event_codes(events)
        np.testing.assert_array_equal(normalized[:, 1:], [[1, 4], [4, 5]])

    def test_resampling_preserves_channel_count_and_exact_duration(self) -> None:
        rng = np.random.default_rng(91)
        signal = rng.normal(scale=1e-6, size=(36, 60 * 512))
        result = _resample_to_target(signal, 512.0, 200.0)
        self.assertEqual(result.shape, (36, 60 * 200))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_rating_reader_requires_complete_subject_trial_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.zip"
            buffer = io.StringIO()
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(["Participant_id", "Trial", "Experiment_id", "Start_time", "Valence", "Arousal", "Dominance", "Liking", "Familiarity"])
            for subject in range(1, DEAP_SUBJECTS + 1):
                for trial in range(1, DEAP_TRIALS_PER_SUBJECT + 1):
                    writer.writerow([subject, trial, trial, trial * 1000, 5.5, 4.5, 6.0, 7.0, 3])
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("participant_ratings.csv", buffer.getvalue())
            ratings = read_deap_ratings(path)
        self.assertEqual(len(ratings), 1280)
        self.assertEqual(ratings[(32, 40)]["valence"], 5.5)


if __name__ == "__main__":
    unittest.main()
