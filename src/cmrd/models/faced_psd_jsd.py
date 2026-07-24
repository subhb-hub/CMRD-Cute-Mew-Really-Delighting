from __future__ import annotations

import torch
from torch import nn


class PaddedCNNTemporalTransformer(nn.Module):
    """Small capacity-matched version of the notebook Conv3d baseline."""

    def __init__(
        self,
        *,
        classes: int = 9,
        channels: int = 30,
        bands: int = 5,
        frequency_bins: int = 17,
        time_steps: int = 30,
        cnn_channels: tuple[int, int, int] = (8, 16, 32),
        d_model: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 256,
        dropout: float = 0.05,
        frequency_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if (channels, bands, frequency_bins) != (30, 5, 17):
            raise ValueError("The current FACED CNN geometry requires [30,5,17]")
        c1, c2, c3 = map(int, cnn_channels)
        if frequency_mask is None:
            frequency_mask = torch.ones(bands, frequency_bins)
        if tuple(frequency_mask.shape) != (bands, frequency_bins):
            raise ValueError(f"Expected frequency mask {(bands, frequency_bins)}")
        self.register_buffer(
            "frequency_mask",
            frequency_mask.float()[None, None, None, :, :],
            persistent=False,
        )
        self.d_model = int(d_model)
        self.time_steps = int(time_steps)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, c1 // 4), c1),
            nn.GELU(),
            nn.MaxPool3d((2, 1, 2)),
            nn.Dropout3d(dropout),
            nn.Conv3d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, c2 // 4), c2),
            nn.GELU(),
            nn.MaxPool3d((2, 1, 2)),
            nn.Dropout3d(dropout),
            nn.Conv3d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(max(1, c3 // 4), c3),
            nn.GELU(),
        )
        self.window_projection = nn.Linear(c3 * 7 * 5 * 4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, time_steps + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, classes),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or tuple(value.shape[2:]) != (30, 5, 17):
            raise ValueError(f"Expected [B,T,30,5,17], got {tuple(value.shape)}")
        batch, time = value.shape[:2]
        if time > self.time_steps:
            raise ValueError(f"time={time} exceeds configured maximum {self.time_steps}")
        windows = value.reshape(batch * time, 30, 5, 17).unsqueeze(1)
        windows = windows * self.frequency_mask.to(windows.dtype)
        tokens = self.window_projection(self.cnn(windows).flatten(1))
        tokens = tokens.reshape(batch, time, self.d_model)
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        sequence = sequence + self.position[:, : time + 1]
        return self.classifier(self.temporal(sequence)[:, 0])


class NativeBandChannelTemporalTransformer(nn.Module):
    """Encode each native band without padded-bin or channel-grid assumptions.

    Input remains the cached padded tensor for efficient batching, but each
    physical band is sliced to its native 3/4/6/16/17 frequency points before
    entering a band-specific encoder. Channel relationships are learned with
    self-attention rather than convolution over the arbitrary channel index.
    """

    def __init__(
        self,
        *,
        band_sizes: tuple[int, ...] = (3, 4, 6, 16, 17),
        classes: int = 9,
        channels: int = 30,
        time_steps: int = 30,
        d_model: int = 64,
        heads: int = 4,
        channel_layers: int = 1,
        temporal_layers: int = 2,
        feedforward: int = 256,
        frequency_hidden: int = 32,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        sizes = tuple(map(int, band_sizes))
        if sizes != (3, 4, 6, 16, 17):
            raise ValueError(f"Expected FACED native band sizes, got {sizes}")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.band_sizes = sizes
        self.channels = int(channels)
        self.time_steps = int(time_steps)
        self.d_model = int(d_model)
        self.frequency_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(size),
                    nn.Linear(size, frequency_hidden),
                    nn.GELU(),
                    nn.Linear(frequency_hidden, d_model),
                )
                for size in sizes
            ]
        )
        self.band_embedding = nn.Parameter(torch.empty(len(sizes), d_model))
        self.channel_embedding = nn.Parameter(torch.empty(channels, d_model))
        self.band_norm = nn.LayerNorm(d_model)
        self.band_score = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1, bias=False)
        )
        channel_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_transformer = nn.TransformerEncoder(
            channel_layer, num_layers=channel_layers, enable_nested_tensor=False
        )
        self.channel_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )
        self.position = nn.Parameter(torch.zeros(1, time_steps, d_model))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers, enable_nested_tensor=False
        )
        self.temporal_score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, classes),
        )
        nn.init.normal_(self.band_embedding, std=0.02)
        nn.init.normal_(self.channel_embedding, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or tuple(value.shape[2:]) != (self.channels, 5, 17):
            raise ValueError(
                f"Expected [B,T,{self.channels},5,17], got {tuple(value.shape)}"
            )
        batch, time = value.shape[:2]
        if time > self.time_steps:
            raise ValueError(f"time={time} exceeds configured maximum {self.time_steps}")
        band_tokens = torch.stack(
            [
                encoder(value[..., band, :size])
                for band, (size, encoder) in enumerate(
                    zip(self.band_sizes, self.frequency_encoders, strict=True)
                )
            ],
            dim=-2,
        )
        band_tokens = self.band_norm(
            band_tokens
            + self.band_embedding[None, None, None, :, :]
            + self.channel_embedding[None, None, :, None, :]
        )
        band_weights = torch.softmax(self.band_score(band_tokens), dim=-2)
        channel_tokens = torch.sum(band_tokens * band_weights, dim=-2)
        channel_tokens = channel_tokens.reshape(
            batch * time, self.channels, self.d_model
        )
        channel_tokens = self.channel_transformer(channel_tokens)
        channel_weights = torch.softmax(self.channel_score(channel_tokens), dim=1)
        time_tokens = torch.sum(channel_tokens * channel_weights, dim=1)
        time_tokens = time_tokens.reshape(batch, time, self.d_model)
        time_tokens = time_tokens + self.position[:, :time]
        time_tokens = self.temporal_transformer(time_tokens)
        time_weights = torch.softmax(self.temporal_score(time_tokens), dim=1)
        trial = torch.sum(time_tokens * time_weights, dim=1)
        return self.classifier(trial)


class NativeBandFlattenTemporalTransformer(nn.Module):
    """Preserve fixed channel-band identity before temporal modelling.

    Each band is still sliced to its native number of frequency points.  In
    contrast to :class:`NativeBandChannelTemporalTransformer`, this variant
    does not average band or channel tokens before classification.  A learned
    window projection sees the complete, ordered channel-by-band grid, which
    removes the symmetric pooling path that can settle at the class prior on
    heterogeneous multi-subject batches.
    """

    def __init__(
        self,
        *,
        band_sizes: tuple[int, ...] = (3, 4, 6, 16, 17),
        classes: int = 9,
        channels: int = 30,
        time_steps: int = 30,
        d_model: int = 64,
        heads: int = 4,
        temporal_layers: int = 2,
        feedforward: int = 256,
        frequency_hidden: int = 32,
        band_dim: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        sizes = tuple(map(int, band_sizes))
        if sizes != (3, 4, 6, 16, 17):
            raise ValueError(f"Expected FACED native band sizes, got {sizes}")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.band_sizes = sizes
        self.channels = int(channels)
        self.time_steps = int(time_steps)
        self.d_model = int(d_model)
        self.frequency_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(size, frequency_hidden),
                    nn.GELU(),
                    nn.Linear(frequency_hidden, band_dim),
                    nn.GELU(),
                )
                for size in sizes
            ]
        )
        window_features = channels * len(sizes) * band_dim
        self.window_projection = nn.Sequential(
            nn.Linear(window_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position = nn.Parameter(torch.zeros(1, time_steps + 1, d_model))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, classes),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or tuple(value.shape[2:]) != (self.channels, 5, 17):
            raise ValueError(
                f"Expected [B,T,{self.channels},5,17], got {tuple(value.shape)}"
            )
        batch, time = value.shape[:2]
        if time > self.time_steps:
            raise ValueError(f"time={time} exceeds configured maximum {self.time_steps}")
        band_tokens = torch.cat(
            [
                encoder(value[..., band, :size])
                for band, (size, encoder) in enumerate(
                    zip(self.band_sizes, self.frequency_encoders, strict=True)
                )
            ],
            dim=-1,
        )
        time_tokens = self.window_projection(band_tokens.flatten(start_dim=2))
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, time_tokens], dim=1)
        sequence = sequence + self.position[:, : time + 1]
        trial = self.temporal_transformer(sequence)[:, 0]
        return self.classifier(trial)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
