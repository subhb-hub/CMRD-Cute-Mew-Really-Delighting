from __future__ import annotations

import unittest

import numpy as np
import torch

from cmrd.faced import VIDEO_LABELS
from cmrd.features.landmark_hilbert import fit_channel_band_atlas
from cmrd.models.slst import ARCHITECTURES, JSDHilbertTokenizer, StructuredLandmarkSpectralTransformer
from cmrd.slst_runner import SubjectBalancedBatchSampler, _rotation_videos, source_coordinate_diagnostics


def atlas(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = np.zeros((5, 7), dtype=bool)
    widths = (3, 4, 5, 6, 7)
    for band, width in enumerate(widths):
        mask[band, :width] = True
    candidates = rng.random((40, 3, 5, 7), dtype=np.float32) * mask[None, None]
    candidates /= candidates.sum(axis=-1, keepdims=True)
    center, anchors = fit_channel_band_atlas(candidates, mask, landmarks=8)
    return candidates, center, anchors


def tokenizer(feature_mode: str = "A6_hilbert_landmark", learnable: bool = False, direction_rank: int = 4) -> JSDHilbertTokenizer:
    _, center, anchors = atlas()
    mask = np.zeros((5, 7), dtype=bool)
    for band, width in enumerate((3, 4, 5, 6, 7)):
        mask[band, :width] = True
    return JSDHilbertTokenizer(
        torch.from_numpy(center),
        torch.from_numpy(anchors),
        torch.from_numpy(mask),
        torch.zeros(3, 5),
        torch.ones(3, 5),
        torch.zeros(3, 5),
        torch.ones(3, 5),
        feature_mode=feature_mode,
        learnable_landmarks=learnable,
        gram_ridge=1e-4,
        direction_rank=direction_rank,
    )


class LandmarkAtlasTests(unittest.TestCase):
    def test_exact_k_is_kept_even_when_band_rank_is_small(self) -> None:
        _, center, anchors = atlas()
        self.assertEqual(center.shape, (3, 5, 7))
        self.assertEqual(anchors.shape, (3, 5, 8, 7))
        np.testing.assert_allclose(anchors[:, 0, :, 3:], 0.0)
        np.testing.assert_allclose(anchors.sum(axis=-1), 1.0, atol=1e-6)

    def test_hilbert_tokens_and_learnable_gradient_are_finite(self) -> None:
        candidates, _, _ = atlas()
        model = tokenizer(learnable=True)
        shape = torch.from_numpy(candidates[:8].reshape(2, 4, 3, 5, 7))
        magnitude = torch.randn(2, 4, 3, 5)
        de = torch.randn_like(magnitude)
        tokens, diagnostics = model(shape, magnitude, de, return_diagnostics=True)
        self.assertEqual(tokens.shape, (2, 4, 3, 5, 10))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertTrue(torch.all(diagnostics["orthogonal_residual"] >= 0))
        regularization = model.regularization(shape, torch.ones(2, 4, dtype=torch.bool))
        (tokens.square().mean() + sum(regularization.values())).backward()
        self.assertIsNotNone(model.landmark_logits.grad)
        self.assertTrue(torch.isfinite(model.landmark_logits.grad).all())

    def test_explicit_scalar_and_low_rank_direction_modes(self) -> None:
        candidates, _, _ = atlas()
        shape = torch.from_numpy(candidates[:8].reshape(2, 4, 3, 5, 7))
        magnitude = torch.randn(2, 4, 3, 5)
        de = torch.randn_like(magnitude)
        expected_dimensions = {
            "H0_scalar_explicit": 2,
            "H1_raw_inner_explicit": 10,
            "H2_pca_lowrank_explicit": 6,
            "H3_hilbert_lowrank_explicit": 6,
            "H4_stable_hilbert_lowrank_explicit": 6,
            "H5_hilbert_full_explicit": 10,
            "H6_stable_hilbert_lowrank_residual": 7,
        }
        for feature_mode, dimension in expected_dimensions.items():
            with self.subTest(feature_mode=feature_mode):
                model = tokenizer(feature_mode, learnable=feature_mode == "H4_stable_hilbert_lowrank_explicit")
                tokens, diagnostics = model(shape, magnitude, de, return_diagnostics=True)
                self.assertEqual(tokens.shape, (2, 4, 3, 5, dimension))
                self.assertTrue(torch.isfinite(tokens).all())
                if feature_mode != "H0_scalar_explicit":
                    self.assertIn("gram_eigenvalues", diagnostics)
                    self.assertTrue(torch.isfinite(diagnostics["direction_coordinates"]).all())

    def test_new_scalar_is_exactly_the_legacy_scalar_baseline(self) -> None:
        candidates, _, _ = atlas()
        shape = torch.from_numpy(candidates[:8].reshape(2, 4, 3, 5, 7))
        magnitude = torch.randn(2, 4, 3, 5)
        de = torch.randn_like(magnitude)
        legacy = tokenizer("A3_scalar_rjsd")(shape, magnitude, de)
        explicit = tokenizer("H0_scalar_explicit")(shape, magnitude, de)
        torch.testing.assert_close(legacy, explicit)

    def test_rank_override_and_source_coordinate_diagnostics(self) -> None:
        candidates, _, _ = atlas()
        shape = torch.from_numpy(candidates[:8].reshape(2, 4, 3, 5, 7))
        magnitude = torch.randn(2, 4, 3, 5)
        de = torch.randn_like(magnitude)
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
        model = StructuredLandmarkSpectralTransformer(
            tokenizer("H4_stable_hilbert_lowrank_explicit", direction_rank=2),
            channels=3, bands=5, classes=4, max_length=4, architecture="B4_slst",
            d_model=16, band_heads=4, channel_heads=4, temporal_heads=4,
            band_layers=1, channel_layers=1, temporal_layers=1, feedforward=32, dropout=0.0,
        )
        tokens = model.tokenizer(shape, magnitude, de)
        self.assertEqual(tokens.shape[-1], 4)
        diagnostics = source_coordinate_diagnostics(
            model,
            [{"shape": shape, "magnitude": magnitude, "de": de, "mask": mask}],
            torch.device("cpu"),
            ("delta", "theta", "alpha", "beta", "gamma"),
            1,
        )
        self.assertEqual(diagnostics["status"], "complete")
        self.assertEqual(diagnostics["direction_rank"], 2)
        self.assertEqual(len(diagnostics["bands"]), 5)
        self.assertTrue(all(np.isfinite(row["orthogonal_residual_over_d0"]) for row in diagnostics["bands"]))


class SLSTModelTests(unittest.TestCase):
    def test_all_architecture_ablations_accept_structured_trials(self) -> None:
        candidates, _, _ = atlas()
        shape = torch.from_numpy(candidates[:8].reshape(2, 4, 3, 5, 7))
        magnitude = torch.randn(2, 4, 3, 5)
        de = torch.randn_like(magnitude)
        mask = torch.tensor([[True, True, True, True], [True, True, True, False]])
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = StructuredLandmarkSpectralTransformer(
                    tokenizer(), channels=3, bands=5, classes=4, max_length=4,
                    architecture=architecture, d_model=16, band_heads=4,
                    channel_heads=4, temporal_heads=4, band_layers=1,
                    channel_layers=1, temporal_layers=1, feedforward=32, dropout=0.0,
                )
                logits = model(shape, magnitude, de, mask)
                self.assertEqual(logits.shape, (2, 4))
                self.assertTrue(torch.isfinite(logits).all())


class ProtocolAndSamplerTests(unittest.TestCase):
    def test_three_faced_rotations_are_disjoint_within_each_emotion(self) -> None:
        observed_tests: dict[int, set[int]] = {label: set() for label in range(9)}
        for rotation in range(3):
            split = _rotation_videos(rotation)
            self.assertFalse(set(split["train"]) & set(split["development"]))
            self.assertFalse(set(split["train"]) & set(split["test"]))
            self.assertFalse(set(split["development"]) & set(split["test"]))
            for video in split["test"]:
                observed_tests[int(VIDEO_LABELS[video])].add(int(video))
        self.assertTrue(all(len(videos) == 3 for videos in observed_tests.values()))

    def test_subject_balanced_batches_contain_multiple_subjects(self) -> None:
        entries = [{"subject": subject} for subject in range(4) for _ in range(3)]
        sampler = SubjectBalancedBatchSampler(entries, subjects_per_batch=4, trials_per_subject=1, seed=42)
        for batch in sampler:
            subjects = {int(entries[index]["subject"]) for index in batch}
            self.assertGreaterEqual(len(subjects), 2)


if __name__ == "__main__":
    unittest.main()
