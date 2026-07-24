from __future__ import annotations

import torch
from torch import nn

from .hierarchical_attention import VectorBandHierarchicalChannelTransformer


class STRJSDHCBT(nn.Module):
    """Capacity-matched vector-band HCBT for Early-State Temporal Reference features."""

    def __init__(
        self,
        *,
        channels: int,
        band_sizes: list[int] | tuple[int, ...],
        classes: int,
        max_length: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = VectorBandHierarchicalChannelTransformer(
            input_dim=int(channels) * sum(map(int, band_sizes)),
            channels=int(channels),
            band_sizes=band_sizes,
            classes=int(classes),
            max_length=int(max_length),
            d_model=int(d_model),
            channel_heads=int(heads),
            temporal_heads=int(heads),
            temporal_layers=int(layers),
            feedforward=int(feedforward),
            dropout=float(dropout),
        )
        classifier = self.backbone.backbone.classifier
        self.backbone.backbone.classifier = nn.Identity()
        self.classifier = classifier

    def encode(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        valid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value = self.backbone(data, mask, valid_indices=valid_indices)
        if not isinstance(value, torch.Tensor):
            raise TypeError("STR-JSD backbone unexpectedly returned attention metadata")
        return value

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        valid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.classifier(self.encode(data, mask, valid_indices))
