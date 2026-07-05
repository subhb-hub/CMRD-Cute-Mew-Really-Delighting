from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.encoding[:, : value.shape[1]]


class PlainTransformer(nn.Module):
    def __init__(self, input_dim: int, classes: int, max_length: int, d_model: int, nhead: int, layers: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPosition(d_model, max_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, classes))

    def forward(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if data.ndim != 3 or mask.shape != data.shape[:2]:
            raise ValueError(f"Expected data [B,T,F] and mask [B,T], got {data.shape}, {mask.shape}")
        encoded = self.position(self.projection(data))
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)

