from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from cmrd.faced import VIDEO_LABELS
from cmrd.faced_adversarial_runner import (
    AdversarialDataset,
    Example,
    SubjectEmotionBatchSampler,
    adversarial_weight,
    load_config,
    stimulus_split,
)
from cmrd.models.subject_adversarial import SubjectAdversarialHCBT, gradient_reverse


ROOT = Path(__file__).resolve().parents[1]


class GradientReverseTests(unittest.TestCase):
    def test_gradient_is_reversed_and_scaled(self) -> None:
        value = torch.tensor([2.0], requires_grad=True)
        loss = gradient_reverse(value, 0.5).square().sum()
        loss.backward()
        self.assertTrue(torch.allclose(value.grad, torch.tensor([-2.0])))

    def test_schedule_warms_up_then_approaches_maximum(self) -> None:
        self.assertEqual(adversarial_weight(1, 15, 3, 0.1), 0.0)
        self.assertEqual(adversarial_weight(3, 15, 3, 0.1), 0.0)
        self.assertGreater(adversarial_weight(4, 15, 3, 0.1), 0.0)
        self.assertGreater(adversarial_weight(15, 15, 3, 0.1), 0.099)


class SubjectAdversarialModelTests(unittest.TestCase):
    def _forward(self, condition: str):
        model = SubjectAdversarialHCBT(
            condition=condition,
            input_dim=150,
            channels=30,
            classes=9,
            source_subjects=8,
            max_length=3,
            d_model=32,
            heads=4,
            layers=1,
            feedforward=64,
            dropout=0.0,
            subject_hidden=16,
            subject_dim=12,
        )
        data = torch.randn(4, 3, 150)
        mask = torch.ones(4, 3, dtype=torch.bool)
        labels = torch.tensor([0, 1, 2, 3])
        return model, model(data, mask, labels=labels, grl_coefficient=0.1)

    def test_b0_has_only_emotion_outputs(self) -> None:
        _, outputs = self._forward("B0_emotion_only")
        self.assertEqual(outputs["emotion_logits"].shape, (4, 9))
        self.assertIsNone(outputs["adversarial_subject_logits"])
        self.assertIsNone(outputs["subject_embedding"])

    def test_b2_has_class_conditional_subject_logits(self) -> None:
        model, outputs = self._forward("B2_conditional_subject_grl")
        self.assertTrue(model.is_conditional)
        self.assertEqual(outputs["adversarial_subject_logits"].shape, (4, 8))
        self.assertIsNone(outputs["positive_subject_logits"])

    def test_b3_separates_emotion_and_subject_latents(self) -> None:
        _, outputs = self._forward("B3_dual_conditional_grl")
        self.assertEqual(outputs["emotion_embedding"].shape, (4, 32))
        self.assertEqual(outputs["subject_embedding"].shape, (4, 12))
        self.assertEqual(outputs["positive_subject_logits"].shape, (4, 8))
        self.assertEqual(outputs["adversarial_subject_logits"].shape, (4, 8))


class FacedAdversarialProtocolTests(unittest.TestCase):
    def test_light_config_locks_b0_to_b3_and_source_development(self) -> None:
        config = load_config(ROOT / "configs" / "faced" / "subject_adversarial_fold1_light.yaml")
        self.assertEqual(config.fold, 1)
        self.assertEqual(config.development_subjects, tuple(range(12, 24)))
        self.assertEqual(config.conditions, SubjectAdversarialHCBT.CONDITIONS)
        self.assertEqual(config.training["epochs"], 15)

        v2 = load_config(ROOT / "configs" / "faced" / "subject_adversarial_fold1_light_v2.yaml")
        self.assertEqual(v2.conditions, SubjectAdversarialHCBT.CONDITIONS)
        self.assertEqual(v2.training["epochs"], 30)
        self.assertEqual(v2.training["adversarial_warmup_epochs"], 8)
        self.assertGreater(v2.training["subject_learning_rate"], v2.training["learning_rate"])

    def test_stimulus_split_is_disjoint_and_covers_every_emotion(self) -> None:
        split = stimulus_split()
        self.assertFalse(set(split["train"]) & set(split["development"]))
        self.assertFalse(set(split["train"]) & set(split["test"]))
        self.assertFalse(set(split["development"]) & set(split["test"]))
        self.assertEqual(VIDEO_LABELS[split["development"]].tolist(), list(range(9)))
        self.assertEqual(VIDEO_LABELS[split["test"]].tolist(), list(range(9)))
        self.assertEqual(len(split["train"]), 10)

    def test_balanced_sampler_contains_multiple_subjects_and_emotions(self) -> None:
        examples = []
        for subject in range(4):
            for label in range(9):
                examples.append(Example(
                    np.full((2, 3), subject + label, dtype=np.float32),
                    label, subject, subject, label,
                ))
        dataset = AdversarialDataset(examples, np.zeros(3, np.float32), np.ones(3, np.float32))
        sampler = SubjectEmotionBatchSampler(dataset, 2, 3, 42)
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 6)
        self.assertEqual(len(np.unique(dataset.local_subjects.numpy()[batch])), 2)
        self.assertEqual(len(np.unique(dataset.labels.numpy()[batch])), 3)


if __name__ == "__main__":
    unittest.main()
