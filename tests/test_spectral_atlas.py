from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmrd.config import load_config
from cmrd.features.spectral_atlas import (
    apply_projection,
    capped_dimension,
    extract_native_shape_power,
    fit_channel_pca,
    fit_landmark_band,
    fit_random_projection,
    full_dimension,
    full_ilr_power,
    full_log_psd,
    landmark_power,
    raw_landmark_coordinates,
    scalar_jsd_power,
)
from cmrd.spectral_atlas_runner import (
    FACED_PROTOCOLS,
    MODELS,
    _base_valid,
    all_conditions,
    experiment_settings,
    split_groups,
    synthetic_smoke,
)


BANDS = {
    "delta": [1.0, 4.0],
    "theta": [4.0, 8.0],
    "alpha": [8.0, 14.0],
    "beta": [14.0, 31.0],
    "gamma": [31.0, 50.0],
}


class SpectralAtlasFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.sizes = [3, 4, 6, 17, 19]
        self.shapes = []
        for size in self.sizes:
            value = rng.random((20, 3, size), dtype=np.float32)
            value /= value.sum(axis=-1, keepdims=True)
            self.shapes.append(value)
        self.power = rng.normal(size=(20, 3, 5)).astype(np.float32)

    def test_native_shape_power_has_expected_grid_and_normalization(self) -> None:
        rng = np.random.default_rng(7)
        signal = rng.normal(size=(2, 400)).astype(np.float32)
        shapes, log_power, grids = extract_native_shape_power(
            signal, 200.0, 1.0, 1.0, BANDS
        )
        self.assertEqual([value.shape for value in shapes], [
            (2, 2, 3), (2, 2, 4), (2, 2, 6), (2, 2, 17), (2, 2, 19),
        ])
        self.assertEqual(log_power.shape, (2, 2, 5))
        self.assertEqual([grid.size for grid in grids], self.sizes)
        for value in shapes:
            np.testing.assert_allclose(value.sum(axis=-1), 1.0, atol=2e-6)
        self.assertTrue(np.isfinite(log_power).all())

    def test_dimension_caps_do_not_exceed_full_psd(self) -> None:
        self.assertEqual(full_dimension([3, 4, 6, 16, 17]), 46)
        self.assertEqual(capped_dimension([3, 4, 6, 16, 17], 1), 10)
        self.assertEqual(capped_dimension([3, 4, 6, 16, 17], 2), 15)
        self.assertEqual(capped_dimension([3, 4, 6, 16, 17], 4), 22)
        self.assertEqual(capped_dimension([3, 4, 6, 16, 17], 8), 31)

    def test_landmark_coordinates_are_finite_and_rank_capped(self) -> None:
        states = [fit_landmark_band(value, 8) for value in self.shapes]
        self.assertEqual([state.anchors.shape[1] for state in states], [2, 3, 5, 8, 8])
        raw = landmark_power(self.shapes, self.power, states, orthogonalized=False)
        nystrom = landmark_power(self.shapes, self.power, states, orthogonalized=True)
        scalar = scalar_jsd_power(self.shapes, self.power, states)
        self.assertEqual(raw.shape, (20, 3, 31))
        self.assertEqual(nystrom.shape, raw.shape)
        self.assertEqual(scalar.shape, (20, 3, 10))
        self.assertTrue(np.isfinite(raw).all())
        self.assertTrue(np.isfinite(nystrom).all())
        first = raw_landmark_coordinates(states[0].reference[None], states[0])
        np.testing.assert_allclose(
            first[0], states[0].anchor_to_reference_jsd, atol=2e-6
        )

    def test_full_information_and_matched_projection_dimensions(self) -> None:
        full_ilr = full_ilr_power(self.shapes, self.power)
        log_psd = full_log_psd(self.shapes, self.power)
        self.assertEqual(full_ilr.shape, (20, 3, 49))
        self.assertEqual(log_psd.shape, (20, 3, 49))
        output_dim = capped_dimension(self.sizes, 4)
        pca = fit_channel_pca(full_ilr, output_dim)
        random = fit_random_projection(3, 49, output_dim, 42)
        self.assertEqual(apply_projection(full_ilr, pca).shape, (20, 3, output_dim))
        self.assertEqual(apply_projection(full_ilr, random).shape, (20, 3, output_dim))


class SpectralAtlasRunnerTests(unittest.TestCase):
    def test_base_validation_returns_builtin_bool_for_json_safe_counting(self) -> None:
        band_names = list(BANDS)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trial.npz"
            arrays = {
                name: np.full((2, 3, size), 1.0 / size, dtype=np.float16)
                for name, size in zip(band_names, (3, 4, 6, 17, 19), strict=True)
            }
            np.savez_compressed(
                path,
                **arrays,
                log_power=np.zeros((2, 3, 5), dtype=np.float32),
                de=np.zeros((2, 3, 5), dtype=np.float32),
            )

            valid = _base_valid(path, band_names, channels=3)

        self.assertIs(type(valid), bool)
        self.assertIs(type(sum([valid, valid])), int)

    def test_configs_declare_two_datasets_and_complete_matrix(self) -> None:
        faced = load_config(ROOT / "configs" / "spectral_atlas" / "faced_v1.yaml")
        seediv = load_config(ROOT / "configs" / "spectral_atlas" / "seediv_v1.yaml")
        faced_settings = experiment_settings(faced)
        seediv_settings = experiment_settings(seediv)
        self.assertEqual(faced_settings["protocols"], FACED_PROTOCOLS)
        self.assertEqual(seediv_settings["protocols"], ("loso",))
        self.assertEqual(faced_settings["models"], MODELS)
        self.assertEqual(faced_settings["conditions"], all_conditions())
        self.assertEqual(seediv_settings["monitor_interval"], 10)

    def test_synthetic_smoke_builds_every_condition(self) -> None:
        config = load_config(ROOT / "configs" / "spectral_atlas" / "seediv_v1.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            result = synthetic_smoke(config, Path(temporary))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(set(result["conditions"]), set(all_conditions()))
        self.assertTrue(all(value[0] > 0 for value in result["conditions"].values()))
        self.assertEqual(result["training_smoke"]["epochs"], 2)
        self.assertEqual(result["training_smoke"]["monitor_points"], 2)

    def test_faced_strict_groups_isolate_subjects_and_videos(self) -> None:
        config = load_config(ROOT / "configs" / "spectral_atlas" / "faced_v1.yaml")
        groups = split_groups(config, 1, "subject_and_stimulus_holdout")
        subjects = {
            name: {int(entry["subject"]) for entry in entries}
            for name, entries in groups.items()
        }
        videos = {
            name: {int(entry["video"]) for entry in entries}
            for name, entries in groups.items()
        }
        self.assertFalse(subjects["train"] & subjects["development"])
        self.assertFalse(subjects["train"] & subjects["test"])
        self.assertFalse(subjects["development"] & subjects["test"])
        self.assertFalse(videos["train"] & videos["development"])
        self.assertFalse(videos["train"] & videos["test"])
        self.assertFalse(videos["development"] & videos["test"])
        self.assertEqual(len(subjects["train"]), 99)
        self.assertEqual(len(subjects["development"]), 12)
        self.assertEqual(len(subjects["test"]), 12)

    def test_powershell_uses_cmrd_and_exposes_safe_stages(self) -> None:
        text = (ROOT / "scripts" / "run_spectral_atlas.ps1").read_text(encoding="utf-8")
        self.assertIn('[string]$CondaEnv = "cmrd"', text)
        for stage in ("Validate", "Smoke", "PrepareBase", "Pilot", "Core", "Full", "Status"):
            self.assertIn(f'"{stage}"', text)
        self.assertIn("--no-resume", text)
        self.assertIn("nystrom_landmark_power_cap4", text)


if __name__ == "__main__":
    unittest.main()
