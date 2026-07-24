from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from cmrd.config import load_config
from cmrd.faced_relative_runner import (
    CONDITIONS,
    VECTOR_CONDITIONS,
    _declared_tasks,
    _supervised_components_by_band,
    relative_settings,
)
from cmrd.features.rd import (
    fit_balanced_multiclass_lda_from_moments,
    transform_native_fisher_rao_supervised,
    transform_native_frequency_point_rjsd,
    transform_native_sqrt_jsd,
)
from cmrd.models import (
    FrequencyPointChannelBandTransformer,
    VectorBandHierarchicalChannelTransformer,
)


ROOT = Path(__file__).resolve().parents[1]


class FrequencyPointRjsdTests(unittest.TestCase):
    def test_frequency_contributions_reconstruct_band_jsd_and_keep_direction(self) -> None:
        reference = np.asarray([
            [0.2, 0.3, 0.5],
            [0.4, 0.4, 0.2],
        ], dtype=np.float32)
        distribution = np.asarray([
            [[0.4, 0.2, 0.4], [0.2, 0.5, 0.3]],
            [[0.1, 0.3, 0.6], [0.5, 0.3, 0.2]],
        ], dtype=np.float32)
        pointwise = transform_native_frequency_point_rjsd(
            [distribution], [reference]
        ).reshape(2, 2, 3)
        compact = transform_native_sqrt_jsd(
            [distribution], [reference]
        ).reshape(2, 2)
        np.testing.assert_allclose(
            np.square(pointwise).sum(axis=-1),
            np.square(compact),
            rtol=2e-5,
            atol=2e-7,
        )
        expected_sign = np.sign(distribution - reference[None])
        self.assertTrue(np.all(np.sign(pointwise)[pointwise != 0] == expected_sign[pointwise != 0]))


class SupervisedFisherProjectionTests(unittest.TestCase):
    def test_balanced_lda_moments_return_deterministic_separating_axes(self) -> None:
        rng = np.random.default_rng(7)
        class_samples = [
            rng.normal(loc=[-2.0, 0.0, 0.0], scale=0.2, size=(40, 3)),
            rng.normal(loc=[2.0, 0.0, 0.0], scale=0.2, size=(30, 3)),
            rng.normal(loc=[0.0, 2.0, 0.0], scale=0.2, size=(20, 3)),
        ]
        counts = np.asarray([len(value) for value in class_samples])
        sums = np.stack([value.sum(axis=0)[None] for value in class_samples])
        crosses = np.stack([
            np.einsum("nf,ng->fg", value, value)[None]
            for value in class_samples
        ])
        center, axes, captured = fit_balanced_multiclass_lda_from_moments(
            counts, sums, crosses, components=2, regularization=1e-3
        )
        self.assertEqual(center.shape, (1, 3))
        self.assertEqual(axes.shape, (1, 3, 2))
        self.assertEqual(captured.shape, (1, 2))
        projected_means = np.stack([
            (value.mean(axis=0) - center[0]) @ axes[0]
            for value in class_samples
        ])
        self.assertGreater(np.min([
            np.linalg.norm(projected_means[i] - projected_means[j])
            for i in range(3) for j in range(i + 1, 3)
        ]), 1.0)

    def test_supervised_fisher_transform_concatenates_components_by_band(self) -> None:
        p1 = np.asarray([[[0.2, 0.3, 0.5]], [[0.4, 0.2, 0.4]]], dtype=np.float32)
        q1 = np.asarray([[0.3, 0.3, 0.4]], dtype=np.float32)
        p2 = np.asarray([[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.3, 0.2, 0.1]]], dtype=np.float32)
        q2 = np.full((1, 4), 0.25, dtype=np.float32)
        centers = [np.zeros_like(q1), np.zeros_like(q2)]
        axes = [
            np.asarray([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]], dtype=np.float32),
            np.asarray([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
        ]
        result = transform_native_fisher_rao_supervised(
            [p1, p2], [q1, q2], centers, axes
        )
        self.assertEqual(result.shape, (2, 4))
        self.assertTrue(np.isfinite(result).all())

    def test_supervised_fisher_transform_allows_variable_components(self) -> None:
        p1 = np.asarray([[[0.2, 0.3, 0.5]], [[0.4, 0.2, 0.4]]], dtype=np.float32)
        q1 = np.asarray([[0.3, 0.3, 0.4]], dtype=np.float32)
        p2 = np.asarray([[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.3, 0.2, 0.1]]], dtype=np.float32)
        q2 = np.full((1, 4), 0.25, dtype=np.float32)
        centers = [np.zeros_like(q1), np.zeros_like(q2)]
        axes = [
            np.eye(3, dtype=np.float32)[None],
            np.eye(4, dtype=np.float32)[None, :, :2],
        ]
        result = transform_native_fisher_rao_supervised(
            [p1, p2], [q1, q2], centers, axes
        )
        self.assertEqual(result.shape, (2, 5))
        self.assertTrue(np.isfinite(result).all())


class FrequencyPointModelTests(unittest.TestCase):
    def test_frequency_encoder_returns_regular_hcbt_attention(self) -> None:
        model = FrequencyPointChannelBandTransformer(
            input_dim=2 * (2 + 3),
            channels=2,
            band_sizes=[2, 3],
            classes=3,
            max_length=4,
            d_model=8,
            channel_heads=2,
            temporal_heads=2,
            temporal_layers=1,
            feedforward=16,
            dropout=0.0,
            frequency_hidden=4,
        ).eval()
        data = torch.randn(2, 4, 10)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        with torch.no_grad():
            logits, attention = model(data, mask, return_attention=True)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(attention["band"].shape, (2, 4, 2, 2))
        self.assertEqual(attention["channel"].shape, (2, 4, 2))

    def test_vector_band_model_preserves_physical_band_tokens(self) -> None:
        model = VectorBandHierarchicalChannelTransformer(
            input_dim=2 * (2 + 3),
            channels=2,
            band_sizes=[2, 3],
            classes=3,
            max_length=4,
            d_model=8,
            channel_heads=2,
            temporal_heads=2,
            temporal_layers=1,
            feedforward=16,
            dropout=0.0,
        ).eval()
        data = torch.randn(2, 4, 10)
        mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        with torch.no_grad():
            logits, attention = model(data, mask, return_attention=True)
        self.assertEqual(logits.shape, (2, 3))
        self.assertEqual(attention["band"].shape, (2, 4, 2, 2))
        self.assertEqual(attention["channel"].shape, (2, 4, 2))
        self.assertEqual(attention["band"][~mask].count_nonzero().item(), 0)


class FacedRelativeProtocolTests(unittest.TestCase):
    def test_config_declares_two_monitored_conditions(self) -> None:
        config = load_config(ROOT / "configs" / "faced" / "relative_supervised_monitor.yaml")
        settings = relative_settings(config)
        self.assertEqual(tuple(settings["conditions"]), tuple(CONDITIONS))
        self.assertEqual(settings["monitor_interval"], 10)
        self.assertEqual(settings["supervised_components"], 2)
        self.assertEqual(settings["feature_storage_dtype"], "float16")

    def test_full_matrix_has_ten_folds_by_two_representations(self) -> None:
        tasks = _declared_tasks(list(range(1, 11)), list(CONDITIONS))
        self.assertEqual(len(tasks), 20)
        self.assertEqual(len({task["task_id"] for task in tasks}), 20)

    def test_vector_config_uses_full_per_band_supervised_dimensions(self) -> None:
        config = load_config(ROOT / "configs" / "faced" / "vector_preserving_monitor.yaml")
        settings = relative_settings(config)
        self.assertEqual(tuple(settings["conditions"]), tuple(VECTOR_CONDITIONS))
        self.assertEqual(_supervised_components_by_band(config), [3, 4, 6, 8, 8])
        self.assertEqual(config.raw["training"]["learning_rate"], 1e-3)
        self.assertTrue(config.raw["training"]["class_balanced_loss"])

    def test_vector_powershell_entrypoint_defaults_to_fold_one(self) -> None:
        script = (ROOT / "scripts" / "run_faced_vector_preserving.ps1").read_text(encoding="utf-8")
        self.assertIn("[int[]]$Folds = @(1)", script)
        for stage in ("Smoke", "VectorRJSD", "FisherFull", "Status", "Summarize"):
            self.assertIn(f'"{stage}"', script)

    def test_powershell_entrypoint_exposes_independent_stages(self) -> None:
        script = (ROOT / "scripts" / "run_faced_relative_supervised.ps1").read_text(encoding="utf-8")
        for stage in (
            "Validate", "Lock", "PrepareFeatures", "Smoke", "PointRJSD",
            "FisherSupervised", "Status", "Summarize",
        ):
            self.assertIn(f'"{stage}"', script)
        self.assertIn("conda run --no-capture-output", script)


if __name__ == "__main__":
    unittest.main()
