from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds {self.encoding.shape[1]}")
        return self.dropout(x + self.encoding[:, : x.shape[1]])


class PlainTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        max_length: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = SinusoidalPositionalEncoding(d_model, max_length, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype is not torch.bool:
            mask = mask.bool()
        if x.shape[:2] != mask.shape:
            raise ValueError(f"x and mask shapes are incompatible: {x.shape}, {mask.shape}")
        encoded = self.position(self.input_projection(x))
        encoded = self.encoder(encoded, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)
