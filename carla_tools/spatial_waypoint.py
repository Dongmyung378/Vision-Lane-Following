"""ResNet spatial controls and a compact attention waypoint predictor.

Both experimental heads receive the same 60 projected CNN locations. The
original pooled ResNet remains a separate, unchanged reference implementation.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .waypoint_model import WaypointModel, WaypointModelError, model_contract as resnet_contract

VARIANTS = ("resnet", "spatial", "hybrid")
TOKEN_DIM = 128
GRID = (6, 10)
ATTENTION_HEADS = 4
ENCODER_LAYERS = 2
FEEDFORWARD_DIM = 384


def spatial_model_contract(variant, config=None):
    if variant not in VARIANTS:
        raise WaypointModelError("spatial-unknown-model-variant")
    contract = copy.deepcopy(resnet_contract(config))
    contract["comparison_variant"] = variant
    if variant == "resnet":
        return contract
    contract["model_version"] = "vlf-resnet18-" + variant + "-waypoint-v1"
    contract["architecture"] = "resnet18-60-spatial-tokens-" + variant
    contract["encoder"]["pooling"] = {"method": "none", "output_grid": list(GRID)}
    contract["head"] = {
        "tokens": 60, "token_dim": TOKEN_DIM, "projection": "Linear(512,128)-LayerNorm",
        "position": "fixed-2d-sinusoidal" if variant == "hybrid" else "row-major-flatten-order",
        "aggregation": "flatten-MLP" if variant == "spatial" else "self-attention-and-waypoint-cross-attention",
        "hidden_features": 64, "dropout": contract["head"]["dropout"],
        "head_precision": "float32", "output_projection_initialization": "zeros",
    }
    if variant == "hybrid":
        contract["head"].update(attention_heads=ATTENTION_HEADS, encoder_layers=ENCODER_LAYERS,
                                query_self_attention_blocks=1, query_cross_attention_blocks=1,
                                feedforward_dim=FEEDFORWARD_DIM,
                                attention_dropout=0.1, waypoint_queries=5,
                                normalization="pre-LayerNorm", attention_backend="SDPA-MATH-float32")
    return contract


def fixed_2d_position():
    """Row-major positions, with distinct x/y sine and cosine coordinates."""
    quarter = TOKEN_DIM // 4
    frequency = torch.exp(-math.log(10000.0) * torch.arange(quarter) / quarter)
    rows, cols = torch.meshgrid(torch.arange(GRID[0]), torch.arange(GRID[1]), indexing="ij")
    x, y = cols.flatten()[:, None] * frequency, rows.flatten()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)[None]


class AttentionBlock(nn.Module):
    """Pre-normalized attention with an explicit float32 math implementation."""

    def __init__(self, *, cross=False):
        super().__init__()
        self.cross = cross
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.memory_norm = nn.LayerNorm(TOKEN_DIM) if cross else None
        self.query = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.key = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.value = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.attention_output = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.ff_norm = nn.LayerNorm(TOKEN_DIM)
        self.ff = nn.Sequential(nn.Linear(TOKEN_DIM, FEEDFORWARD_DIM), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(FEEDFORWARD_DIM, TOKEN_DIM))
        self.dropout = nn.Dropout(0.1)

    def forward(self, value, memory=None):
        query = self.norm(value)
        key = self.memory_norm(memory) if self.cross else query
        batch = value.shape[0]
        def split(projected):
            return projected.reshape(batch, -1, ATTENTION_HEADS, TOKEN_DIM // ATTENTION_HEADS).transpose(1, 2)
        with sdpa_kernel(SDPBackend.MATH):
            attended = nn.functional.scaled_dot_product_attention(
                split(self.query(query)), split(self.key(key)), split(self.value(key)),
                dropout_p=0.1 if self.training else 0.0)
        attended = self.attention_output(attended.transpose(1, 2).reshape(batch, -1, TOKEN_DIM))
        value = value + self.dropout(attended)
        return value + self.dropout(self.ff(self.ff_norm(value)))


class SpatialHead(nn.Module):
    def __init__(self, variant, dropout):
        super().__init__()
        self.variant = variant
        self.projection = nn.Sequential(nn.Linear(512, TOKEN_DIM), nn.LayerNorm(TOKEN_DIM))
        if variant == "spatial":
            self.output = nn.Sequential(nn.Linear(60 * TOKEN_DIM, 64), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(64, 10))
        else:
            self.register_buffer("position", fixed_2d_position())
            self.blocks = nn.ModuleList(AttentionBlock() for _ in range(ENCODER_LAYERS))
            self.queries = nn.Parameter(torch.empty(1, 5, TOKEN_DIM))
            nn.init.normal_(self.queries, std=0.02)
            self.query_attention = AttentionBlock()
            self.cross_attention = AttentionBlock(cross=True)
            self.output = nn.Sequential(nn.LayerNorm(TOKEN_DIM), nn.Linear(TOKEN_DIM, 64),
                                        nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features):
        if features.ndim != 4 or tuple(features.shape[1:]) != (512, *GRID):
            raise WaypointModelError("spatial-invalid-feature-shape")
        # Autocast is used by the CNN, but attention/normalization and waypoint
        # regression stay float32 on T4 to avoid half-precision overflow.
        with torch.autocast(device_type=features.device.type, enabled=False):
            tokens = self.projection(features.float().flatten(2).transpose(1, 2))
            if self.variant == "spatial":
                return self.output(tokens.flatten(1)).reshape(-1, 5, 2)
            memory = tokens + self.position
            for block in self.blocks:
                memory = block(memory)
            queries = self.query_attention(self.queries.expand(features.shape[0], -1, -1))
            return self.output(self.cross_attention(queries, memory))


class SpatialWaypointModel(WaypointModel):
    def __init__(self, *, variant="hybrid", pretrained=True, weights_path=None, config=None):
        if variant not in ("spatial", "hybrid"):
            raise WaypointModelError("spatial-unknown-experimental-variant")
        super().__init__(pretrained=pretrained, weights_path=weights_path, config=config)
        self.variant = variant
        self.encoder.avgpool = nn.Identity()
        self.head = SpatialHead(variant, self.config.head_dropout)
        self.train(self.training)

    def forward(self, image):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or tuple(image.shape[1:]) != (3, 180, 320):
            raise WaypointModelError("spatial-invalid-input-tensor-shape")
        if image.shape[0] < 1 or not image.is_floating_point() or not bool(torch.isfinite(image).all()):
            raise WaypointModelError("spatial-nonfinite-or-invalid-input")
        features = self.encoder(image).reshape(-1, 512, *GRID)
        return self.straight_prior[None] + self.head(features) * self.config.label_scale_m


def create_spatial_model(variant, *, config=None, pretrained=True, weights_path=None):
    if variant == "resnet":
        return WaypointModel(config=config, pretrained=pretrained, weights_path=weights_path)
    return SpatialWaypointModel(variant=variant, config=config, pretrained=pretrained, weights_path=weights_path)
