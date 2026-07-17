from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmrd.config import load_config
from cmrd.features.rd import (
    extract_native_spectral_distributions,
    fisher_rao_log_map,
    native_frequency_grid,
    transform_native_fisher_rao_pca,
    transform_native_sqrt_jsd,
    transform_native_wasserstein1,
)
from cmrd.native_compact_runner import (
    EXPECTED_CONDITIONS,
    _fit_pca_state,
    _fit_reference_state,
    declared_tasks,
    experiment_settings,
)


BANDS = {
    "delta": [1.0, 4.0],
    "theta": [4.0, 8.0],
    "alpha": [8.0, 14.0],
    "beta": [14.0, 31.0],
    "gamma": [31.0, 50.0],
}


class NativeSpectralFeatureTests(unittest.TestCase):
    def test_one_second_native_grid_has_no_zero_padding(self) -> None:
        grid = native_frequency_grid(200.0, 1.0, BANDS)
        self.assertEqual([values.size for values in grid], [3, 4, 6, 17, 19])
        np.testing.assert_array_equal(grid[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(grid[-1], np.arange(31.0, 50.0))

    def test_native_extraction_normalizes_each_band(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(2, 400)).astype(np.float32)
        distributions, frequencies = extract_native_spectral_distributions(
            signal, 200.0, 1.0, 1.0, BANDS
        )
        self.assertEqual([value.shape for value in distributions], [
            (2, 2, 3), (2, 2, 4), (2, 2, 6), (2, 2, 17), (2, 2, 19),
        ])
        self.assertEqual([value.size for value in frequencies], [3, 4, 6, 17, 19])
        for value in distributions:
            np.testing.assert_allclose(value.sum(axis=-1), 1.0, atol=1e-6)

    def test_sqrt_jsd_and_wasserstein_keep_one_scalar_per_band(self) -> None:
        rng = np.random.default_rng(7)
        frequencies = native_frequency_grid(200.0, 1.0, BANDS)
        distributions = []
        references = []
        for grid in frequencies:
            value = rng.random((3, 62, grid.size), dtype=np.float32)
            value /= value.sum(axis=-1, keepdims=True)
            reference = rng.random((62, grid.size), dtype=np.float32)
            reference /= reference.sum(axis=-1, keepdims=True)
            distributions.append(value)
            references.append(reference)
        sqrt_jsd = transform_native_sqrt_jsd(distributions, references)
        wasserstein = transform_native_wasserstein1(distributions, references, frequencies)
        self.assertEqual(sqrt_jsd.shape, (3, 310))
        self.assertEqual(wasserstein.shape, (3, 310))
        self.assertTrue(np.isfinite(sqrt_jsd).all())
        self.assertTrue(np.logical_and(wasserstein >= 0.0, wasserstein <= 1.0).all())

    def test_normalized_wasserstein_uses_support_diameter(self) -> None:
        low = np.asarray([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        high = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        distance = transform_native_wasserstein1([low], [high], [np.asarray([1.0, 2.0, 3.0])])
        self.assertAlmostEqual(float(distance[0, 0]), 1.0, places=6)
        identity = transform_native_wasserstein1([low], [low[0]], [np.asarray([1.0, 2.0, 3.0])])
        np.testing.assert_allclose(identity, 0.0, atol=1e-7)

    def test_fisher_rao_log_map_is_tangent_and_projection_is_compact(self) -> None:
        rng = np.random.default_rng(11)
        distributions = []
        references = []
        means = []
        components = []
        for size in (3, 4, 6, 17, 19):
            value = rng.random((2, 62, size), dtype=np.float32)
            value /= value.sum(axis=-1, keepdims=True)
            reference = rng.random((62, size), dtype=np.float32)
            reference /= reference.sum(axis=-1, keepdims=True)
            tangent = fisher_rao_log_map(value, reference)
            radial = (tangent * np.sqrt(reference)[None]).sum(axis=-1)
            np.testing.assert_allclose(radial, 0.0, atol=2e-6)
            expected_norm = 2.0 * np.arccos(np.clip(
                (np.sqrt(value) * np.sqrt(reference)[None]).sum(axis=-1), 0.0, 1.0
            ))
            np.testing.assert_allclose(
                np.linalg.norm(tangent, axis=-1), expected_norm, atol=3e-5
            )
            component = rng.normal(size=(62, size)).astype(np.float32)
            component /= np.linalg.norm(component, axis=-1, keepdims=True)
            distributions.append(value)
            references.append(reference)
            means.append(np.zeros_like(reference))
            components.append(component)
        output = transform_native_fisher_rao_pca(
            distributions, references, means, components
        )
        self.assertEqual(output.shape, (2, 310))
        self.assertTrue(np.isfinite(output).all())


class NativeCompactRunnerTests(unittest.TestCase):
    def test_streamed_source_only_reference_and_pca_state(self) -> None:
        config = load_config(ROOT / "configs" / "native_compact" / "seed_v1.yaml")
        frequencies = native_frequency_grid(200.0, 1.0, BANDS)
        entries = [
            {"trial_id": f"source-{index}", "subject": index + 2}
            for index in range(3)
        ]
        rng = np.random.default_rng(123)
        by_trial = {}
        for entry in entries:
            values = []
            for grid in frequencies:
                distribution = rng.random((4, 62, grid.size), dtype=np.float32)
                distribution /= distribution.sum(axis=-1, keepdims=True)
                values.append(distribution)
            by_trial[entry["trial_id"]] = values

        def fake_native(_config, _audit, entry):
            return by_trial[entry["trial_id"]], frequencies

        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "cmrd.native_compact_runner._native_distributions", side_effect=fake_native
        ):
            cache = Path(directory)
            references = _fit_reference_state(config, {}, entries, cache, "test-protocol")
            pca = _fit_pca_state(config, {}, entries, references, cache, "test-protocol")

        self.assertEqual(len(pca["components"]), 5)
        for reference, component in zip(
            references["fisher_rao"], pca["components"], strict=True
        ):
            np.testing.assert_allclose(np.linalg.norm(component, axis=-1), 1.0, atol=2e-5)
            np.testing.assert_allclose(
                (component * np.sqrt(reference)).sum(axis=-1), 0.0, atol=2e-5
            )

    def test_config_is_fold1_base_v2_with_three_conditions(self) -> None:
        config = load_config(ROOT / "configs" / "native_compact" / "seed_v1.yaml")
        settings = experiment_settings(config)
        tasks = declared_tasks(config, "protocol")
        self.assertEqual(tuple(settings["conditions"]), EXPECTED_CONDITIONS)
        self.assertEqual(len(tasks), 3)
        self.assertEqual({task["fold"] for task in tasks}, {1})
        self.assertEqual({task["architecture"] for task in tasks}, {"base"})
        self.assertEqual(settings["max_epochs"], 200)
        self.assertEqual(settings["target_monitor_interval"], 10)

    def test_powershell_defaults_to_cmrd_conda_environment(self) -> None:
        script = (ROOT / "scripts" / "run_native_compact.ps1").read_text(encoding="utf-8")
        self.assertIn('[string]$CondaEnv = "cmrd"', script)


if __name__ == "__main__":
    unittest.main()
