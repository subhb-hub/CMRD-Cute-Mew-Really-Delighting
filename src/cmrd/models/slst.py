from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .transformer import SinusoidalPosition


FEATURE_MODES = (
    "A0_magnitude",
    "A1_de",
    "A2_full_shape",
    "A3_scalar_rjsd",
    "A4_raw_landmark",
    "A5_centered_landmark",
    "A6_hilbert_landmark",
    "H0_scalar_explicit",
    "H1_raw_inner_explicit",
    "H2_pca_lowrank_explicit",
    "H3_hilbert_lowrank_explicit",
    "H4_stable_hilbert_lowrank_explicit",
    "H5_hilbert_full_explicit",
    "H6_stable_hilbert_lowrank_residual",
)
ARCHITECTURES = ("B0_flatten_mlp", "B1_flatten_temporal", "B2_band_temporal", "B3_channel_temporal", "B4_slst")


def _jsd(left: torch.Tensor, right: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    p = left.clamp_min(epsilon)
    q = right.clamp_min(epsilon)
    midpoint = 0.5 * (p + q)
    value = 0.5 * torch.sum(p * (torch.log(p) - torch.log(midpoint)), dim=-1)
    value = value + 0.5 * torch.sum(q * (torch.log(q) - torch.log(midpoint)), dim=-1)
    return value.clamp_min(0.0)


class JSDHilbertTokenizer(nn.Module):
    """Build differentiable magnitude/shape tokens from a source-only atlas."""

    def __init__(
        self,
        center: torch.Tensor,
        anchors: torch.Tensor,
        frequency_mask: torch.Tensor,
        magnitude_mean: torch.Tensor,
        magnitude_scale: torch.Tensor,
        de_mean: torch.Tensor,
        de_scale: torch.Tensor,
        *,
        feature_mode: str = "A6_hilbert_landmark",
        learnable_landmarks: bool = False,
        random_landmarks: bool = False,
        gram_ridge: float = 1e-4,
        direction_rank: int = 4,
        eigenvalue_floor_ratio: float = 1e-3,
        effective_rank_tolerance: float = 1e-6,
        diversity_margin: float = 0.1,
        coverage_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if feature_mode not in FEATURE_MODES:
            raise ValueError(f"Unknown feature mode {feature_mode}")
        center = torch.as_tensor(center, dtype=torch.float32)
        anchors = torch.as_tensor(anchors, dtype=torch.float32)
        mask = torch.as_tensor(frequency_mask, dtype=torch.bool)
        if center.ndim != 3 or anchors.ndim != 4 or anchors.shape[:2] != center.shape[:2] or anchors.shape[-1] != center.shape[-1]:
            raise ValueError(f"Atlas shape mismatch: {center.shape}/{anchors.shape}")
        if mask.shape != center.shape[1:]:
            raise ValueError(f"Frequency mask mismatch: {mask.shape}/{center.shape}")
        if gram_ridge <= 0 or direction_rank < 1 or coverage_temperature <= 0 or eigenvalue_floor_ratio <= 0 or effective_rank_tolerance <= 0:
            raise ValueError("gram_ridge, eigenvalue floor/rank tolerance, and coverage temperature must be positive")
        self.feature_mode = feature_mode
        self.learnable_landmarks = bool(learnable_landmarks)
        self.gram_ridge = float(gram_ridge)
        self.configured_direction_rank = int(direction_rank)
        self.eigenvalue_floor_ratio = float(eigenvalue_floor_ratio)
        self.effective_rank_tolerance = float(effective_rank_tolerance)
        self.diversity_margin = float(diversity_margin)
        self.coverage_temperature = float(coverage_temperature)
        self.register_buffer("center", center)
        self.register_buffer("frequency_mask", mask)
        self.register_buffer("initial_anchors", anchors)
        self.register_buffer("magnitude_mean", torch.as_tensor(magnitude_mean, dtype=torch.float32))
        self.register_buffer("magnitude_scale", torch.as_tensor(magnitude_scale, dtype=torch.float32).clamp_min(1e-6))
        self.register_buffer("de_mean", torch.as_tensor(de_mean, dtype=torch.float32))
        self.register_buffer("de_scale", torch.as_tensor(de_scale, dtype=torch.float32).clamp_min(1e-6))
        if learnable_landmarks:
            if random_landmarks:
                generator = torch.Generator().manual_seed(19_907)
                logits = 0.05 * torch.randn(anchors.shape, generator=generator)
            else:
                logits = torch.log(anchors.clamp_min(1e-6))
            self.landmark_logits = nn.Parameter(logits)
        else:
            self.register_buffer("fixed_anchors", anchors)

    @property
    def landmarks(self) -> int:
        return int(self.initial_anchors.shape[-2])

    @property
    def output_dim(self) -> int:
        selected_rank = self.direction_rank
        return {
            "A0_magnitude": 1,
            "A1_de": 1,
            "A2_full_shape": 1 + int(self.center.shape[-1]),
            "A3_scalar_rjsd": 2,
            "A4_raw_landmark": 1 + self.landmarks,
            "A5_centered_landmark": 1 + self.landmarks,
            "A6_hilbert_landmark": 2 + self.landmarks,
            "H0_scalar_explicit": 2,
            "H1_raw_inner_explicit": 2 + self.landmarks,
            "H2_pca_lowrank_explicit": 2 + selected_rank,
            "H3_hilbert_lowrank_explicit": 2 + selected_rank,
            "H4_stable_hilbert_lowrank_explicit": 2 + selected_rank,
            "H5_hilbert_full_explicit": 2 + self.landmarks,
            "H6_stable_hilbert_lowrank_residual": 3 + selected_rank,
        }[self.feature_mode]

    @property
    def direction_rank(self) -> int:
        if self.feature_mode in {
            "H2_pca_lowrank_explicit",
            "H3_hilbert_lowrank_explicit",
            "H4_stable_hilbert_lowrank_explicit",
            "H6_stable_hilbert_lowrank_residual",
        }:
            return min(self.configured_direction_rank, self.landmarks)
        return self.landmarks

    def current_anchors(self) -> torch.Tensor:
        if not self.learnable_landmarks:
            return self.fixed_anchors
        mask = self.frequency_mask[None, :, None, :].expand_as(self.landmark_logits)
        logits = self.landmark_logits.masked_fill(~mask, -30.0)
        value = torch.softmax(logits, dim=-1) * mask.to(logits.dtype)
        return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def set_landmarks_trainable(self, trainable: bool) -> None:
        if self.learnable_landmarks:
            self.landmark_logits.requires_grad_(bool(trainable))

    def landmark_parameters(self) -> list[nn.Parameter]:
        return [self.landmark_logits] if self.learnable_landmarks else []

    def _standardized_magnitude(self, magnitude: torch.Tensor) -> torch.Tensor:
        return (magnitude - self.magnitude_mean[None, None]) / self.magnitude_scale[None, None]

    def _standardized_de(self, de: torch.Tensor) -> torch.Tensor:
        return (de - self.de_mean[None, None]) / self.de_scale[None, None]

    def _distances(self, shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = self.current_anchors()
        d0 = _jsd(shape, self.center[None, None])
        dk = _jsd(shape.unsqueeze(-2), anchors[None, None])
        return anchors, d0, dk

    def _gram_geometry(
        self, anchors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_d0 = _jsd(anchors, self.center.unsqueeze(-2))
        pairwise = _jsd(anchors.unsqueeze(-2), anchors.unsqueeze(-3))
        gram = 0.5 * (anchor_d0.unsqueeze(-1) + anchor_d0.unsqueeze(-2) - pairwise)
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        eigenvalues = eigenvalues.flip(-1).clamp_min(0.0)
        eigenvectors = eigenvectors.flip(-1)
        return anchor_d0, pairwise, eigenvalues, eigenvectors

    def _direction_coordinates(
        self,
        shape: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        anchors, d0, dk = self._distances(shape)
        anchor_d0, pairwise, eigenvalues, eigenvectors = self._gram_geometry(anchors)
        inner = 0.5 * (d0.unsqueeze(-1) + anchor_d0[None, None] - dk)
        rank = self.direction_rank
        if mode == "raw":
            coordinates = inner
            denominator = torch.ones_like(eigenvalues)
        else:
            vectors = eigenvectors[..., :rank]
            coordinates = torch.einsum("ntcbk,cbkr->ntcbr", inner, vectors)
            selected = eigenvalues[..., :rank]
            if mode == "pca":
                denominator = torch.ones_like(selected)
            elif mode == "hilbert":
                denominator = selected.clamp_min(self.gram_ridge).sqrt()
            elif mode == "stable_hilbert":
                floor = self.eigenvalue_floor_ratio * eigenvalues[..., :1]
                denominator = torch.maximum(selected, floor).clamp_min(self.gram_ridge).sqrt()
            else:
                raise ValueError(f"Unknown direction coordinate mode {mode}")
            coordinates = coordinates / denominator[None, None]
        residual = torch.relu(d0 - torch.sum(coordinates.square(), dim=-1))
        return coordinates, residual, {
            "distance_to_center": d0,
            "distance_to_anchors": dk,
            "pairwise_anchor_jsd": pairwise,
            "gram_eigenvalues": eigenvalues,
            "coordinate_denominator": denominator,
            "direction_coordinates": coordinates,
            "orthogonal_residual": residual,
        }

    def _hilbert(self, shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        anchors, d0, dk = self._distances(shape)
        anchor_d0 = _jsd(anchors, self.center.unsqueeze(-2))
        pairwise = _jsd(anchors.unsqueeze(-2), anchors.unsqueeze(-3))
        gram = 0.5 * (anchor_d0.unsqueeze(-1) + anchor_d0.unsqueeze(-2) - pairwise)
        gram = 0.5 * (gram + gram.transpose(-1, -2))
        identity = torch.eye(self.landmarks, dtype=gram.dtype, device=gram.device)
        cholesky = torch.linalg.cholesky(gram + self.gram_ridge * identity)
        inner = 0.5 * (d0.unsqueeze(-1) + anchor_d0[None, None] - dk)
        coordinates = torch.linalg.solve_triangular(
            cholesky[None, None], inner.unsqueeze(-1), upper=False
        ).squeeze(-1)
        residual = torch.relu(d0 - torch.sum(coordinates.square(), dim=-1))
        return coordinates, residual, {
            "distance_to_center": d0,
            "distance_to_anchors": dk,
            "pairwise_anchor_jsd": pairwise,
        }

    def forward(
        self,
        shape: torch.Tensor,
        magnitude: torch.Tensor,
        de: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if shape.ndim != 5 or magnitude.shape != shape.shape[:-1] or de.shape != magnitude.shape:
            raise ValueError(f"Expected q=[B,T,C,Band,F], magnitude/de=[B,T,C,Band], got {shape.shape}/{magnitude.shape}/{de.shape}")
        magnitude_z = self._standardized_magnitude(magnitude)
        diagnostics: dict[str, torch.Tensor] = {}
        if self.feature_mode == "A0_magnitude":
            tokens = magnitude_z.unsqueeze(-1)
        elif self.feature_mode == "A1_de":
            tokens = self._standardized_de(de).unsqueeze(-1)
        elif self.feature_mode == "A2_full_shape":
            valid = self.frequency_mask[None, None, None].to(shape.dtype)
            tokens = torch.cat((magnitude_z.unsqueeze(-1), shape * valid), dim=-1)
        else:
            anchors, d0, dk = self._distances(shape)
            if self.feature_mode == "A3_scalar_rjsd":
                tokens = torch.cat((magnitude_z.unsqueeze(-1), torch.sqrt(d0.clamp_min(0.0)).unsqueeze(-1)), dim=-1)
            elif self.feature_mode == "A4_raw_landmark":
                tokens = torch.cat((magnitude_z.unsqueeze(-1), dk), dim=-1)
            elif self.feature_mode == "A5_centered_landmark":
                tokens = torch.cat((magnitude_z.unsqueeze(-1), d0.unsqueeze(-1) - dk), dim=-1)
            elif self.feature_mode == "A6_hilbert_landmark":
                coordinates, residual, hilbert = self._hilbert(shape)
                diagnostics.update(hilbert)
                diagnostics["orthogonal_residual"] = residual
                tokens = torch.cat((magnitude_z.unsqueeze(-1), coordinates, residual.unsqueeze(-1)), dim=-1)
            elif self.feature_mode == "H0_scalar_explicit":
                tokens = torch.cat((magnitude_z.unsqueeze(-1), torch.sqrt(d0.clamp_min(0.0)).unsqueeze(-1)), dim=-1)
            else:
                direction_mode = {
                    "H1_raw_inner_explicit": "raw",
                    "H2_pca_lowrank_explicit": "pca",
                    "H3_hilbert_lowrank_explicit": "hilbert",
                    "H4_stable_hilbert_lowrank_explicit": "stable_hilbert",
                    "H5_hilbert_full_explicit": "hilbert",
                    "H6_stable_hilbert_lowrank_residual": "stable_hilbert",
                }[self.feature_mode]
                coordinates, residual, geometry = self._direction_coordinates(shape, direction_mode)
                diagnostics.update(geometry)
                pieces = [magnitude_z.unsqueeze(-1), torch.sqrt(d0.clamp_min(0.0)).unsqueeze(-1), coordinates]
                if self.feature_mode == "H6_stable_hilbert_lowrank_residual":
                    pieces.append(torch.sqrt(residual.clamp_min(0.0)).unsqueeze(-1))
                tokens = torch.cat(pieces, dim=-1)
            diagnostics.setdefault("distance_to_center", d0)
            diagnostics.setdefault("distance_to_anchors", dk)
            diagnostics["anchors"] = anchors
        if return_diagnostics:
            return tokens, diagnostics
        return tokens

    def regularization(self, shape: torch.Tensor, time_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        zero = shape.new_zeros(())
        if not self.learnable_landmarks:
            return {"anchor": zero, "diversity": zero, "coverage": zero}
        anchors = self.current_anchors()
        anchor_loss = _jsd(anchors, self.initial_anchors).mean()
        pairwise = _jsd(anchors.unsqueeze(-2), anchors.unsqueeze(-3))
        off_diagonal = ~torch.eye(self.landmarks, dtype=torch.bool, device=shape.device)
        distances = torch.sqrt(pairwise.clamp_min(1e-12))[..., off_diagonal]
        diversity = torch.relu(self.diversity_margin - distances).square().mean()
        coverage_distance = _jsd(shape.unsqueeze(-2), anchors[None, None])
        temperature = self.coverage_temperature
        coverage_values = -temperature * (
            torch.logsumexp(-coverage_distance / temperature, dim=-1) - math.log(self.landmarks)
        )
        if time_mask is None:
            coverage = coverage_values.mean()
        else:
            valid = time_mask[:, :, None, None].to(coverage_values.dtype)
            coverage = (coverage_values * valid).sum() / (
                valid.sum().clamp_min(1.0) * coverage_values.shape[-1] * coverage_values.shape[-2]
            )
        return {"anchor": anchor_loss, "diversity": diversity, "coverage": coverage}

    @torch.no_grad()
    def atlas_diagnostics(self) -> dict[str, Any]:
        anchors = self.current_anchors()
        _, pairwise, eigenvalues, _ = self._gram_geometry(anchors)
        off_diagonal = ~torch.eye(self.landmarks, dtype=torch.bool, device=anchors.device)
        positive_floor = self.effective_rank_tolerance * eigenvalues[..., :1]
        effective_rank = (eigenvalues > positive_floor).sum(dim=-1).to(torch.float32)
        minimum = eigenvalues[..., -1]
        condition = eigenvalues[..., 0] / minimum.clamp_min(self.gram_ridge)
        return {
            "mean_jsd_drift": float(_jsd(anchors, self.initial_anchors).mean().cpu()),
            "mean_probability_l1_drift": float(torch.abs(anchors - self.initial_anchors).sum(dim=-1).mean().cpu()),
            "mean_pairwise_jsd": float(pairwise[..., off_diagonal].mean().cpu()),
            "minimum_pairwise_jsd": float(pairwise[..., off_diagonal].min().cpu()),
            "minimum_gram_eigenvalue": float(minimum.min().cpu()),
            "median_gram_condition": float(condition.median().cpu()),
            "mean_effective_rank": float(effective_rank.mean().cpu()),
        }


class AttentionPool(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dimension, dimension), nn.Tanh(), nn.Linear(dimension, 1, bias=False))

    def forward(self, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(value).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        return torch.sum(value * weights.unsqueeze(-1), dim=-2)


def _encoder(d_model: int, heads: int, layers: int, feedforward: int, dropout: float) -> nn.Module:
    if layers == 0:
        return nn.Identity()
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class FactorizedCoordinateEncoder(nn.Module):
    def __init__(self, input_dim: int, channels: int, bands: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.empty(input_dim, d_model))
        self.channel = nn.Parameter(torch.empty(channels, input_dim, d_model))
        self.band = nn.Parameter(torch.empty(bands, input_dim, d_model))
        self.channel_embedding = nn.Parameter(torch.empty(channels, d_model))
        self.band_embedding = nn.Parameter(torch.empty(bands, d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.shared)
        nn.init.normal_(self.channel, std=0.005)
        nn.init.normal_(self.band, std=0.005)
        nn.init.normal_(self.channel_embedding, std=0.02)
        nn.init.normal_(self.band_embedding, std=0.02)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"Coordinate tokens must be [B,T,C,Band,D], got {value.shape}")
        shared = torch.einsum("ntcbi,io->ntcbo", value, self.shared)
        channel = torch.einsum("ntcbi,cio->ntcbo", value, self.channel)
        band = torch.einsum("ntcbi,bio->ntcbo", value, self.band)
        output = shared + channel + band + self.bias
        output = output + self.channel_embedding[None, None, :, None]
        output = output + self.band_embedding[None, None, None]
        return self.dropout(self.norm(output))


class StructuredLandmarkSpectralTransformer(nn.Module):
    """Band -> channel -> time model with explicit structure ablations."""

    def __init__(
        self,
        tokenizer: JSDHilbertTokenizer,
        *,
        channels: int,
        bands: int,
        classes: int,
        max_length: int,
        architecture: str = "B4_slst",
        d_model: int = 128,
        band_heads: int = 4,
        channel_heads: int = 8,
        temporal_heads: int = 8,
        band_layers: int = 2,
        channel_layers: int = 2,
        temporal_layers: int = 3,
        feedforward: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture {architecture}")
        if any(d_model % value for value in (band_heads, channel_heads, temporal_heads)):
            raise ValueError("d_model must be divisible by every attention head count")
        self.tokenizer = tokenizer
        self.channels = int(channels)
        self.bands = int(bands)
        self.architecture = architecture
        self.coordinate_encoder = FactorizedCoordinateEncoder(tokenizer.output_dim, channels, bands, d_model, dropout)
        use_band = architecture in {"B2_band_temporal", "B4_slst"}
        use_channel = architecture in {"B3_channel_temporal", "B4_slst"}
        use_flatten = architecture in {"B0_flatten_mlp", "B1_flatten_temporal"}
        self.band_transformer = _encoder(d_model, band_heads, band_layers if use_band else 0, feedforward, dropout)
        self.band_pool = AttentionPool(d_model)
        self.channel_transformer = _encoder(d_model, channel_heads, channel_layers if use_channel else 0, feedforward, dropout)
        self.channel_pool = AttentionPool(d_model)
        self.flatten_projection: nn.Module | None = nn.Sequential(
            nn.LayerNorm(channels * bands * d_model),
            nn.Linear(channels * bands * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        ) if use_flatten else None
        self.position = SinusoidalPosition(d_model, max_length)
        self.temporal_transformer = _encoder(
            d_model, temporal_heads, temporal_layers if architecture != "B0_flatten_mlp" else 0,
            feedforward, dropout
        )
        self.temporal_pool = AttentionPool(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, classes),
        )

    def _time_tokens(self, embedded: torch.Tensor) -> torch.Tensor:
        batch, time, channels, bands, dimension = embedded.shape
        if self.architecture in {"B0_flatten_mlp", "B1_flatten_temporal"}:
            assert self.flatten_projection is not None
            return self.flatten_projection(embedded.reshape(batch, time, channels * bands * dimension))
        if self.architecture in {"B2_band_temporal", "B4_slst"}:
            band_input = embedded.reshape(batch * time * channels, bands, dimension)
            embedded = self.band_transformer(band_input).reshape(batch, time, channels, bands, dimension)
        channel_tokens = self.band_pool(embedded)
        if self.architecture in {"B3_channel_temporal", "B4_slst"}:
            channel_input = channel_tokens.reshape(batch * time, channels, dimension)
            channel_tokens = self.channel_transformer(channel_input).reshape(batch, time, channels, dimension)
        return self.channel_pool(channel_tokens)

    def encode(
        self,
        shape: torch.Tensor,
        magnitude: torch.Tensor,
        de: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        coordinates = self.tokenizer(shape, magnitude, de)
        assert isinstance(coordinates, torch.Tensor)
        time_tokens = self._time_tokens(self.coordinate_encoder(coordinates))
        if self.architecture != "B0_flatten_mlp":
            time_tokens = self.temporal_transformer(self.position(time_tokens), src_key_padding_mask=~mask)
            return self.temporal_pool(time_tokens, mask)
        valid = mask.unsqueeze(-1).to(time_tokens.dtype)
        return (time_tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        shape: torch.Tensor,
        magnitude: torch.Tensor,
        de: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.encode(shape, magnitude, de, mask))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def configuration(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "feature_mode": self.tokenizer.feature_mode,
            "landmarks": self.tokenizer.landmarks,
            "direction_rank": self.tokenizer.direction_rank,
            "eigenvalue_floor_ratio": self.tokenizer.eigenvalue_floor_ratio,
            "learnable_landmarks": self.tokenizer.learnable_landmarks,
            "parameters": self.parameter_count(),
        }
