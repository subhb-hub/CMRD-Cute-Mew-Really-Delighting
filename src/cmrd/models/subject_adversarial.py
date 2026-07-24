from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F

from .hierarchical_attention import HierarchicalChannelBandTransformer


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = float(coefficient)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


def gradient_reverse(value: torch.Tensor, coefficient: float) -> torch.Tensor:
    return _GradientReverse.apply(value, float(coefficient))


class _Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class SubjectAdversarialHCBT(nn.Module):
    """HCBT with optional marginal/conditional subject adversary and dual latents.

    Conditions are deliberately limited to the first diagnostic ablation:

    ``B0_emotion_only``
        Emotion classifier only.
    ``B1_marginal_subject_grl``
        Marginal subject adversary on the emotion representation.
    ``B2_conditional_subject_grl``
        Class-conditional subject adversary using z_e x onehot(y).
    ``B3_dual_conditional_grl``
        Separate emotion/subject latents, a positive subject head on z_s, and
        a class-conditional adversary on z_e.
    """

    CONDITIONS = (
        "B0_emotion_only",
        "B1_marginal_subject_grl",
        "B2_conditional_subject_grl",
        "B3_dual_conditional_grl",
    )

    def __init__(
        self,
        *,
        condition: str,
        input_dim: int,
        channels: int,
        classes: int,
        source_subjects: int,
        max_length: int,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 3,
        feedforward: int = 512,
        dropout: float = 0.15,
        subject_hidden: int = 128,
        subject_dim: int = 64,
    ) -> None:
        super().__init__()
        if condition not in self.CONDITIONS:
            raise ValueError(f"Unknown subject-adversarial condition: {condition}")
        if source_subjects < 2:
            raise ValueError("Subject adversarial training requires at least two source subjects")
        self.condition = condition
        self.classes = int(classes)
        self.source_subjects = int(source_subjects)
        self.d_model = int(d_model)
        self.backbone = HierarchicalChannelBandTransformer(
            input_dim=int(input_dim),
            channels=int(channels),
            classes=int(classes),
            max_length=int(max_length),
            d_model=int(d_model),
            channel_heads=int(heads),
            temporal_heads=int(heads),
            temporal_layers=int(layers),
            feedforward=int(feedforward),
            dropout=float(dropout),
        )
        # The established HCBT performs temporal pooling immediately before
        # this module. Replacing only its classifier preserves the full encoder.
        base_emotion_classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()

        if condition == "B3_dual_conditional_grl":
            self.emotion_encoder = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
            )
            self.subject_encoder: nn.Module | None = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, subject_dim),
                nn.GELU(),
            )
            emotion_dim = int(d_model)
            self.positive_subject_classifier: nn.Module | None = _Classifier(
                int(subject_dim), int(subject_hidden), int(source_subjects), float(dropout)
            )
        else:
            self.emotion_encoder = nn.Identity()
            self.subject_encoder = None
            emotion_dim = int(d_model)
            self.positive_subject_classifier = None

        # Reuse the established classifier, including its initialization, so B0
        # is the exact HCBT architecture rather than a merely similar baseline.
        self.emotion_classifier = base_emotion_classifier
        if condition == "B1_marginal_subject_grl":
            adversarial_dim = emotion_dim
        elif condition in {"B2_conditional_subject_grl", "B3_dual_conditional_grl"}:
            adversarial_dim = emotion_dim * int(classes)
        else:
            adversarial_dim = 0
        self.adversarial_subject_classifier: nn.Module | None = (
            _Classifier(adversarial_dim, int(subject_hidden), int(source_subjects), float(dropout))
            if adversarial_dim
            else None
        )

    @property
    def is_conditional(self) -> bool:
        return self.condition in {"B2_conditional_subject_grl", "B3_dual_conditional_grl"}

    def encode(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        valid_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        shared = self.backbone(data, mask, valid_indices=valid_indices)
        if not isinstance(shared, torch.Tensor):
            raise TypeError("HCBT encoder unexpectedly returned attention metadata")
        emotion = self.emotion_encoder(shared)
        subject = self.subject_encoder(shared) if self.subject_encoder is not None else None
        return emotion, subject

    def _conditional_input(self, emotion: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 1 or labels.shape[0] != emotion.shape[0]:
            raise ValueError("Conditional subject adversary requires one emotion label per trial")
        onehot = F.one_hot(labels, num_classes=self.classes).to(emotion.dtype)
        return torch.einsum("bd,bc->bcd", emotion, onehot).reshape(emotion.shape[0], -1)

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        grl_coefficient: float = 0.0,
        valid_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        emotion, subject = self.encode(data, mask, valid_indices)
        emotion_logits = self.emotion_classifier(emotion)
        adversarial_logits: torch.Tensor | None = None
        if self.adversarial_subject_classifier is not None:
            if self.is_conditional:
                if labels is None:
                    raise ValueError("Conditional adversarial forward requires source emotion labels")
                adversarial_input = self._conditional_input(emotion, labels)
            else:
                adversarial_input = emotion
            adversarial_logits = self.adversarial_subject_classifier(
                gradient_reverse(adversarial_input, grl_coefficient)
            )
        positive_logits = (
            self.positive_subject_classifier(subject)
            if self.positive_subject_classifier is not None and subject is not None
            else None
        )
        return {
            "emotion_logits": emotion_logits,
            "emotion_embedding": emotion,
            "subject_embedding": subject,
            "adversarial_subject_logits": adversarial_logits,
            "positive_subject_logits": positive_logits,
        }
