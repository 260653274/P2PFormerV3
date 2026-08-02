"""Contour evidence conditioning used by P2PFormerV3.

The module is a source-compatible reconstruction of the contour feature
enhancer described by P2PFormerV2.  It deliberately exposes the ablation
routes used by the V3 design (SFE, DFE, and their fusion) while keeping a
single, fixed output interface for the support-box generator.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import RoIAlign


def _meshgrid(y: torch.Tensor,
              x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use explicit ij indexing when supported without dropping torch 1.x."""
    try:
        return torch.meshgrid(y, x, indexing='ij')
    except TypeError:  # pragma: no cover - only exercised on old torch.
        return torch.meshgrid(y, x)


class ContourEvidenceConditioner(nn.Module):
    """Build contour-aware per-instance memories from P2/P3/P4.

    Args:
        in_channels: Channel count of every feature level.
        crop_size: Per-level aligned RoI crop side.
        sample_side: Sparse sampling lattice side.
        num_heads: Dense-to-sparse cross-attention heads.
        feature_strides: Input strides for P2/P3/P4.
        offset_scale: Maximum sparse offset in crop pixels.
        mask_floor: Residual floor for the differentiable feature mask.
        mask_mode: ``soft`` (V3), ``hard`` (V2 semantics), or ``none``.
        variant: ``full``, ``sfe``, or ``dfe`` ablation route.
    """

    _VALID_MASK_MODES = ('soft', 'hard', 'none')
    _VALID_VARIANTS = ('full', 'sfe', 'dfe')

    def __init__(self,
                 in_channels: int = 256,
                 crop_size: int = 32,
                 sample_side: int = 8,
                 num_heads: int = 8,
                 feature_strides: Sequence[int] = (4, 8, 16),
                 offset_scale: float = 4.0,
                 mask_floor: float = 0.1,
                 contour_half_width: float = 8.0,
                 dilation_temperature: float = 0.1,
                 hard_threshold: float = 0.5,
                 mask_mode: str = 'soft',
                 variant: str = 'full',
                 roi_chunk_size: int = 16) -> None:
        super().__init__()
        if crop_size != 32 or sample_side != 8:
            raise ValueError('V3 v0.1 fixes crop/sample sides to 32/8')
        if len(feature_strides) != 3:
            raise ValueError('Conditioner expects exactly P2/P3/P4')
        if in_channels % num_heads != 0:
            raise ValueError('in_channels must be divisible by num_heads')
        if mask_mode not in self._VALID_MASK_MODES:
            raise ValueError('Unknown mask mode: {}'.format(mask_mode))
        if variant not in self._VALID_VARIANTS:
            raise ValueError('Unknown conditioner variant: {}'.format(
                variant))

        self.in_channels = in_channels
        self.crop_size = crop_size
        self.sample_side = sample_side
        self.offset_scale = float(offset_scale)
        self.mask_floor = float(mask_floor)
        self.contour_half_width = float(contour_half_width)
        self.dilation_temperature = float(dilation_temperature)
        self.hard_threshold = float(hard_threshold)
        self.mask_mode = mask_mode
        self.variant = variant
        self.roi_chunk_size = int(roi_chunk_size)
        if self.roi_chunk_size < 1:
            raise ValueError('roi_chunk_size must be positive')

        self.roi_aligners = nn.ModuleList([
            RoIAlign(
                output_size=(crop_size, crop_size),
                spatial_scale=1.0 / stride,
                sampling_ratio=2,
                pool_mode='avg',
                aligned=True) for stride in feature_strides
        ])

        self.offset_net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=5,
                stride=4,
                padding=2,
                groups=in_channels,
                bias=True), nn.GELU(),
            nn.Conv2d(in_channels, 2, kernel_size=1, bias=True))

        self.query_norm = nn.LayerNorm(in_channels)
        self.memory_norm = nn.LayerNorm(in_channels)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=False)
        self.level_weights = nn.Parameter(torch.zeros(3))

        self.contour_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(64, 2, kernel_size=1))
        self.fusion_projection = nn.Linear(in_channels * 2, in_channels)

        base_axis = torch.arange(sample_side, dtype=torch.float32)
        base_axis = 4.0 * base_axis + 1.5
        base_y, base_x = _meshgrid(base_axis, base_axis)
        base_grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(0)
        self.register_buffer('base_grid_pixels', base_grid)

        dilation_axis = torch.arange(-16, 17, dtype=torch.float32)
        dilation_y, dilation_x = _meshgrid(dilation_axis, dilation_axis)
        dilation_offsets = torch.stack(
            (dilation_x.reshape(-1), dilation_y.reshape(-1)), dim=-1)
        self.register_buffer('dilation_offsets', dilation_offsets)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.offset_net[-1].weight)
        nn.init.zeros_(self.offset_net[-1].bias)
        for module in (self.contour_head[0], self.contour_head[-1],
                       self.fusion_projection):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def token_count(self) -> int:
        return self.sample_side * self.sample_side

    def _empty_output(self, reference: torch.Tensor) -> Dict[str, object]:
        c = self.in_channels
        s = self.crop_size
        token_count = self.token_count
        tokens = tuple(
            reference.new_empty((0, token_count, c)) for _ in range(3))
        positions = tuple(reference.new_empty((0, token_count, c))
                          for _ in range(3))
        grids = tuple(reference.new_empty((0, self.sample_side,
                                           self.sample_side, 2))
                      for _ in range(3))
        offsets = tuple(reference.new_empty((0, 2, self.sample_side,
                                             self.sample_side))
                        for _ in range(3))
        return dict(
            memories=tokens,
            position_embeddings=positions,
            dense_feature=reference.new_empty((0, c, s, s)),
            image_context=reference.new_empty((0, c)),
            contour_logits=reference.new_empty((0, 2, s, s)),
            contour_probability=reference.new_empty((0, 1, s, s)),
            feature_mask=reference.new_empty((0, 1, s, s)),
            sample_grids=grids,
            offsets=offsets,
            roi_features=tuple(
                reference.new_empty((0, c, s, s)) for _ in range(3)))

    def _sample_grid(self, crop: torch.Tensor,
                     learn_offsets: bool) -> Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        n = crop.shape[0]
        if learn_offsets:
            raw_offsets = self.offset_net(crop)
            offsets = self.offset_scale * torch.tanh(raw_offsets)
        else:
            raw_offsets = crop.new_zeros(
                (n, 2, self.sample_side, self.sample_side))
            offsets = raw_offsets

        offset_xy = offsets.permute(0, 2, 3, 1)
        pixel_grid = self.base_grid_pixels.to(crop).expand(n, -1, -1, -1)
        pixel_grid = (pixel_grid + offset_xy).clamp_(
            min=0.0, max=float(self.crop_size - 1))
        normalized_grid = 2.0 * (pixel_grid + 0.5) / self.crop_size - 1.0
        sampled = F.grid_sample(
            crop,
            normalized_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        return sampled, normalized_grid, offsets

    def _cross_attend(self, crop: torch.Tensor,
                      sparse_tokens: torch.Tensor) -> torch.Tensor:
        n, c, h, w = crop.shape
        dense_tokens = crop.flatten(2).transpose(1, 2)
        query = self.query_norm(dense_tokens).transpose(0, 1)
        memory = self.memory_norm(sparse_tokens).transpose(0, 1)
        attended, _ = self.cross_attention(
            query, memory, memory, need_weights=False)
        dense_tokens = dense_tokens + attended.transpose(0, 1)
        return dense_tokens.transpose(1, 2).reshape(n, c, h, w)

    def _position_embedding(
            self, normalized_grid: torch.Tensor) -> torch.Tensor:
        """Return DETR-style 2-D sine encoding with shape [N,64,256]."""
        positions = (normalized_grid.flatten(1, 2) + 1.0) * 0.5
        positions = positions * (2.0 * math.pi)
        num_feats = self.in_channels // 2
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=positions.device)
        dim_t = 10000**(2 * torch.div(
            dim_t, 2, rounding_mode='floor') / num_feats)

        def encode(values: torch.Tensor) -> torch.Tensor:
            values = values.float().unsqueeze(-1) / dim_t
            encoded = torch.stack(
                (values[..., 0::2].sin(), values[..., 1::2].cos()),
                dim=-1).flatten(-2)
            return encoded

        pos_x = encode(positions[..., 0])
        pos_y = encode(positions[..., 1])
        return torch.cat((pos_y, pos_x), dim=-1).to(normalized_grid.dtype)

    def _ellipse_mask(self, rois: torch.Tensor) -> torch.Tensor:
        widths = (rois[:, 3] - rois[:, 1]).clamp_min(1.0)
        heights = (rois[:, 4] - rois[:, 2]).clamp_min(1.0)
        radius_x = (self.contour_half_width * self.crop_size /
                    widths).clamp(1.0, 16.0)
        radius_y = (self.contour_half_width * self.crop_size /
                    heights).clamp(1.0, 16.0)
        offsets = self.dilation_offsets.to(rois)
        ellipse = ((offsets[None, :, 0] / radius_x[:, None])**2 +
                   (offsets[None, :, 1] / radius_y[:, None])**2 <= 1.0)
        return ellipse

    def _contour_band(self, contour_probability: torch.Tensor,
                      rois: torch.Tensor) -> torch.Tensor:
        if self.mask_mode == 'none':
            return torch.ones_like(contour_probability)

        n, _, h, w = contour_probability.shape
        patches = F.unfold(
            contour_probability, kernel_size=33, padding=16)
        valid_source = contour_probability.new_ones((n, 1, h, w))
        valid = F.unfold(valid_source, kernel_size=33, padding=16) > 0.5
        ellipse = self._ellipse_mask(rois).unsqueeze(-1)
        valid = valid & ellipse

        if self.mask_mode == 'hard':
            band = ((patches > self.hard_threshold) & valid).any(
                dim=1, keepdim=True).to(contour_probability.dtype)
        else:
            logits = patches.float() / self.dilation_temperature
            logits = logits.masked_fill(~valid, -1e4)
            weights = logits.softmax(dim=1)
            band = (weights * patches.float()).sum(dim=1, keepdim=True)
            band = band.to(contour_probability.dtype)
        return band.reshape(n, 1, h, w)

    def _apply_mask(self, feature: torch.Tensor,
                    band: torch.Tensor) -> torch.Tensor:
        if self.mask_mode == 'soft':
            gate = self.mask_floor + (1.0 - self.mask_floor) * band
        else:
            gate = band
        return feature * gate

    def forward(self, features: Sequence[torch.Tensor],
                rois: torch.Tensor) -> Dict[str, object]:
        """Condition P2/P3/P4 features for flattened building RoIs.

        Args:
            features: Three tensors `[B,256,H_l,W_l]` in P2/P3/P4 order.
            rois: `[N,5]` with `(batch_index,x1,y1,x2,y2)` in input pixels.

        Returns:
            A dictionary whose ``memories`` and ``position_embeddings`` are
            ordered coarse-to-fine `(E4,E3,E2)`.
        """
        if len(features) != 3:
            raise ValueError('Expected exactly P2/P3/P4 features')
        if rois.ndim != 2 or rois.shape[-1] != 5:
            raise ValueError('rois must have shape [N,5]')
        if rois.shape[0] == 0:
            return self._empty_output(features[0])
        if rois.shape[0] > self.roi_chunk_size:
            chunks = [
                self.forward(features, roi_chunk)
                for roi_chunk in rois.split(self.roi_chunk_size, dim=0)
            ]
            tuple_keys = ('memories', 'position_embeddings', 'sample_grids',
                          'offsets', 'roi_features')
            merged: Dict[str, object] = {}
            for key in tuple_keys:
                merged[key] = tuple(
                    torch.cat([chunk[key][level] for chunk in chunks], dim=0)
                    for level in range(3))
            for key in ('dense_feature', 'image_context', 'contour_logits',
                        'contour_probability', 'feature_mask'):
                merged[key] = torch.cat(
                    [chunk[key] for chunk in chunks], dim=0)
            return merged

        crops: List[torch.Tensor] = []
        sparse_tokens: List[torch.Tensor] = []
        grids: List[torch.Tensor] = []
        offsets: List[torch.Tensor] = []
        enhanced: List[torch.Tensor] = []
        learn_offsets = self.variant != 'dfe'

        for feature, aligner in zip(features, self.roi_aligners):
            crop = aligner(feature, rois)
            sampled, grid, level_offsets = self._sample_grid(
                crop, learn_offsets=learn_offsets)
            tokens = sampled.flatten(2).transpose(1, 2)
            if self.variant == 'sfe':
                dense = crop
            else:
                dense = self._cross_attend(crop, tokens)
            crops.append(crop)
            sparse_tokens.append(tokens)
            grids.append(grid)
            offsets.append(level_offsets)
            enhanced.append(dense)

        weights = self.level_weights.softmax(dim=0)
        fused = sum(weight * level
                    for weight, level in zip(weights, enhanced))
        contour_logits = self.contour_head(fused)
        contour_probability = contour_logits.softmax(dim=1)[:, 1:2]
        if self.variant == 'sfe':
            band = torch.ones_like(contour_probability)
        else:
            band = self._contour_band(contour_probability, rois)

        memories: List[torch.Tensor] = []
        positions: List[torch.Tensor] = []
        for crop, sparse, grid in zip(crops, sparse_tokens, grids):
            if self.variant == 'sfe':
                memory = sparse
            else:
                masked = self._apply_mask(crop, band)
                dense_sample = F.grid_sample(
                    masked,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False).flatten(2).transpose(1, 2)
                if self.variant == 'dfe':
                    memory = dense_sample
                else:
                    memory = self.fusion_projection(
                        torch.cat((sparse, dense_sample), dim=-1))
            memories.append(memory)
            positions.append(self._position_embedding(grid))

        dense_feature = (fused if self.variant == 'sfe' else
                         self._apply_mask(fused, band))
        # Denoising is coarse-to-fine, hence reverse P2/P3/P4 here.
        return dict(
            memories=tuple(memories[::-1]),
            position_embeddings=tuple(positions[::-1]),
            dense_feature=dense_feature,
            image_context=fused.mean(dim=(-2, -1)),
            contour_logits=contour_logits,
            contour_probability=contour_probability,
            feature_mask=band,
            sample_grids=tuple(grids[::-1]),
            offsets=tuple(offsets[::-1]),
            roi_features=tuple(crops[::-1]))
