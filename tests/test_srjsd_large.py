from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmrd.config import load_config
from cmrd.data.records import TrialSample
from cmrd.features.rd import transform_signed_sqrt_rd
from cmrd.fixed_protocol import build_model, fit_exploratory_monitored_source_model
from cmrd.srjsd_large_runner import EXPECTED_CONDITIONS, declared_tasks, experiment_settings


class SignedSqrtRjsdTests(unittest.TestCase):
    def test_signed_sqrt_jsd_direction_symmetry_and_tie(self) -> None:
        bands = [[1.0, 5.0]]
        low = np.asarray([[[[0.70, 0.20, 0.08, 0.02]]]], dtype=np.float32)
        high = np.asarray([[[[0.02, 0.08, 0.20, 0.70]]]], dtype=np.float32)
        high_from_low = transform_signed_sqrt_rd(high, low[0], bands)
        low_from_high = transform_signed_sqrt_rd(low, high[0], bands)
        self.assertGreater(float(high_from_low[0, 0]), 0.0)
        self.assertLess(float(low_from_high[0, 0]), 0.0)
        self.assertAlmostEqual(abs(float(high_from_low[0, 0])), abs(float(low_from_high[0, 0])), places=6)
        np.testing.assert_allclose(transform_signed_sqrt_rd(low, low[0], bands), 0.0, atol=1e-7)

        # Both distributions have the same centroid but differ in shape. The
        # deterministic +1 tie must preserve the non-zero JSD magnitude.
        outer = np.asarray([[[[0.4, 0.1, 0.1, 0.4]]]], dtype=np.float32)
        inner = np.asarray([[[0.1, 0.4, 0.4, 0.1]]], dtype=np.float32)
        tied = transform_signed_sqrt_rd(outer, inner, bands)
        self.assertGreater(float(tied[0, 0]), 0.0)

    def test_real_shape_remains_310_dimensions(self) -> None:
        rng = np.random.default_rng(42)
        histogram = rng.random((3, 62, 5, 32), dtype=np.float32)
        reference = rng.random((62, 5, 32), dtype=np.float32)
        bands = [[1, 4], [4, 8], [8, 14], [14, 31], [31, 50]]
        output = transform_signed_sqrt_rd(histogram, reference, bands)
        self.assertEqual(output.shape, (3, 310))
        self.assertTrue(np.isfinite(output).all())


class SrjsdLargeRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "srjsd_large" / "seed_v1.yaml")

    def test_frozen_matrix_declares_75_tasks_per_dataset(self) -> None:
        settings = experiment_settings(self.config)
        tasks = declared_tasks(self.config, "protocol")
        self.assertEqual(tuple(settings["conditions"]), EXPECTED_CONDITIONS)
        self.assertEqual(len(tasks), 75)
        self.assertEqual(len({task["task_id"] for task in tasks}), 75)
        self.assertEqual([task["condition"] for task in tasks[:5]], list(EXPECTED_CONDITIONS))
        self.assertEqual(settings["max_epochs"], 200)
        self.assertEqual(settings["target_monitor_interval"], 10)

    def test_large_model_has_more_capacity_than_base(self) -> None:
        settings = experiment_settings(self.config)
        counts = {}
        for name in ("base", "large"):
            architecture = settings["architectures"][name]
            model = build_model({
                "name": "hierarchical_attention",
                **architecture,
            }, input_dim=310, classes=3, max_length=4)
            counts[name] = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(counts["large"], counts["base"] * 4)
        self.assertGreater(counts["large"], 4_000_000)

    def test_monitored_training_keeps_fixed_final_checkpoint(self) -> None:
        rng = np.random.default_rng(7)
        source = [
            TrialSample(rng.normal(size=(2 + index % 2, 310)).astype(np.float32), index % 3, index + 1, 1, index + 1, index)
            for index in range(6)
        ]
        target = [
            TrialSample(rng.normal(size=(2, 310)).astype(np.float32), index % 3, 15, 1, index + 1, index)
            for index in range(3)
        ]
        model = {
            "name": "hierarchical_attention",
            "d_model": 8,
            "heads": 2,
            "layers": 1,
            "feedforward": 16,
            "dropout": 0.0,
        }
        training = {
            "deterministic": True,
            "batch_size": 3,
            "learning_rate": 2e-4,
            "minimum_learning_rate": 1e-6,
            "warmup_fraction": 0.1,
            "weight_decay": 0.01,
            "label_smoothing": 0.0,
            "gradient_clip_norm": 1.0,
            "gradient_accumulation_steps": 1,
            "locked_epochs": 2,
            "target_monitor_interval": 1,
            "precision": "float32",
            "num_workers": 0,
            "persistent_workers": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "final.pt"
            result = fit_exploratory_monitored_source_model(
                source, target, model, training, 3, torch.device("cpu"), checkpoint_path
            )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.assertEqual(result["final_epoch"], 2)
        self.assertEqual([row["epoch"] for row in result["target_curve"]], [1, 2])
        self.assertTrue(checkpoint["target_monitoring_during_training"])
        self.assertEqual(checkpoint["checkpoint_selection"], "fixed_final_epoch_only")


if __name__ == "__main__":
    unittest.main()
