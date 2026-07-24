from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from cmrd.faced import VIDEO_LABELS
from cmrd.faced_normative_probe import (
    build_features,
    fit_atlas,
    heldout_videos,
    load_probe_config,
    route_atlas,
)


ROOT = Path(__file__).resolve().parents[1]


class FacedNormativeProbeTests(unittest.TestCase):
    def test_config_keeps_development_out_of_outer_target(self) -> None:
        config = load_probe_config(ROOT / "configs" / "faced" / "normative_probe_fold1.yaml")
        self.assertEqual(config.fold, 1)
        self.assertEqual(config.development_subjects, tuple(range(12, 24)))
        self.assertEqual(config.pseudo_reference_windows, 5)

    def test_heldout_policy_selects_one_video_per_emotion(self) -> None:
        videos = heldout_videos()
        self.assertEqual(len(videos), 9)
        self.assertEqual(VIDEO_LABELS[videos].tolist(), list(range(9)))

    def test_shrinkage_endpoints_have_expected_interpretation(self) -> None:
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(3, 4, 5))
        response = rng.normal(size=(3, 4, 5))
        atlas = rng.normal(size=(3, 4, 5))
        paired = build_features("A1_paired_pseudo_brde", reference, response)
        alpha_one = build_features(
            "A5_shrink_pseudo_brde", reference, response, atlas=atlas, alpha=1.0
        )
        alpha_zero = build_features(
            "A5_shrink_pseudo_brde", reference, response, atlas=atlas, alpha=0.0
        )
        self.assertTrue(np.allclose(paired, alpha_one))
        self.assertTrue(np.allclose(alpha_zero, response - atlas))

    def test_soft_atlas_weights_sum_to_one(self) -> None:
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(8, 6, 5))
        state = fit_atlas(reference, list(range(8)), 3, 42)
        atlas, weights = route_atlas(reference, state)
        self.assertEqual(atlas.shape, reference.shape)
        self.assertEqual(weights.shape, (8, 6, 3))
        self.assertTrue(np.allclose(weights.sum(axis=-1), 1.0))
        self.assertEqual(sum(state.subject_cluster_counts), 8)


if __name__ == "__main__":
    unittest.main()
