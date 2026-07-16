from __future__ import annotations

import torch
from torch import nn

from .transformer import SinusoidalPosition


class HierarchicalChannelBandTransformer(nn.Module):
    """Attend over feature bands, EEG channels, and time in that order.

    The regular training pipeline stores each window as ``channels * bands``.
    This module accepts that flattened representation as ``[B, T, D]`` as well
    as an already structured ``[B, T, C, F]`` tensor.
    """

    def __init__(
        self,
        input_dim: int,
        channels: int,
        classes: int,
        max_length: int,
        d_model: int = 64,
        channel_heads: int = 4,
        temporal_heads: int = 4,
        temporal_layers: int = 3,
        feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or channels <= 0 or input_dim % channels != 0:
            raise ValueError(f"input_dim={input_dim} must be divisible by channels={channels}")
        if d_model <= 0 or d_model % channel_heads != 0 or d_model % temporal_heads != 0:
            raise ValueError("d_model must be positive and divisible by both attention head counts")

        self.input_dim = int(input_dim)
        self.channels = int(channels)
        self.feature_slots = self.input_dim // self.channels
        self.d_model = int(d_model)

        # A scalar EEG feature becomes a token. Learned slot and channel
        # identities keep equal-valued bands/electrodes distinguishable.
        self.value_embedding = nn.Linear(1, d_model)
        self.band_embedding = nn.Parameter(torch.empty(self.feature_slots, d_model))
        self.channel_embedding = nn.Parameter(torch.empty(channels, d_model))
        self.band_norm = nn.LayerNorm(d_model)
        self.band_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )
        self.band_dropout = nn.Dropout(dropout)

        # One self-attention block relates all channels independently at every
        # time step, followed by a learned channel pooling distribution.
        self.channel_norm1 = nn.LayerNorm(d_model)
        self.channel_attention = nn.MultiheadAttention(
            d_model, channel_heads, dropout=dropout, batch_first=True
        )
        self.channel_norm2 = nn.LayerNorm(d_model)
        self.channel_feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, d_model),
        )
        self.channel_dropout = nn.Dropout(dropout)
        self.channel_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )

        self.position = SinusoidalPosition(d_model, max_length)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=temporal_heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=temporal_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, classes),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.band_embedding, std=0.02)
        nn.init.normal_(self.channel_embedding, std=0.02)

    def _structure_input(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"Expected mask [B,T], got {mask.shape}")
        if data.ndim == 3:
            if data.shape[:2] != mask.shape or data.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected flattened data [B,T,{self.input_dim}] matching mask, "
                    f"got {data.shape}, {mask.shape}"
                )
            return data.reshape(*data.shape[:2], self.channels, self.feature_slots)
        if data.ndim == 4:
            expected = (*mask.shape, self.channels, self.feature_slots)
            if tuple(data.shape) != expected:
                raise ValueError(f"Expected structured data {expected}, got {tuple(data.shape)}")
            return data
        raise ValueError(f"Expected data [B,T,D] or [B,T,C,F], got {data.shape}")

    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
        valid_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        structured = self._structure_input(data, mask)
        batch, time, _, _ = structured.shape

        # [B,T,C,F] -> [B,T,C,F,d], then learned pooling over F.
        band_tokens = self.value_embedding(structured.unsqueeze(-1))
        band_tokens = band_tokens + self.band_embedding[None, None, None, :, :]
        band_tokens = band_tokens + self.channel_embedding[None, None, :, None, :]
        band_tokens = self.band_dropout(self.band_norm(band_tokens))
        band_weights = torch.softmax(self.band_score(band_tokens).squeeze(-1), dim=-1)
        channel_tokens = torch.sum(band_tokens * band_weights.unsqueeze(-1), dim=-2)

        # Only valid windows enter channel MHA, avoiding wasted work on padding.
        flat_channels = channel_tokens.reshape(batch * time, self.channels, self.d_model)
        if valid_indices is None:
            # Compatibility path for direct CPU model calls. Production
            # loaders compute these indices before moving the mask to CUDA,
            # avoiding a dynamic-shape GPU nonzero and its synchronization.
            valid_indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        elif valid_indices.device != flat_channels.device:
            raise ValueError("valid_indices must be on the same device as data")
        valid_channels = flat_channels.index_select(0, valid_indices)
        normalized = self.channel_norm1(valid_channels)
        related, _ = self.channel_attention(
            normalized, normalized, normalized, need_weights=False
        )
        valid_channels = valid_channels + self.channel_dropout(related)
        valid_channels = valid_channels + self.channel_dropout(
            self.channel_feedforward(self.channel_norm2(valid_channels))
        )
        flat_related = flat_channels.new_zeros(flat_channels.shape).index_copy(
            0, valid_indices, valid_channels
        )
        channel_tokens = flat_related.reshape(batch, time, self.channels, self.d_model)

        channel_weights = torch.softmax(self.channel_score(channel_tokens).squeeze(-1), dim=-1)
        channel_weights = channel_weights * mask.unsqueeze(-1).to(channel_weights.dtype)
        time_tokens = torch.sum(channel_tokens * channel_weights.unsqueeze(-1), dim=-2)

        time_tokens = self.position(time_tokens)
        time_tokens = self.temporal_transformer(
            time_tokens, src_key_padding_mask=~mask
        )
        valid = mask.unsqueeze(-1).to(time_tokens.dtype)
        trial_representation = (time_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(trial_representation)

        if not return_attention:
            return logits
        band_weights = band_weights * mask[:, :, None, None].to(band_weights.dtype)
        return logits, {
            "band": band_weights,
            "channel": channel_weights,
        }
