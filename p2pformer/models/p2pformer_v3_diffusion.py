"""Geometry and diffusion primitives for P2PFormerV3.

This module is deliberately self-contained. It does not register an
MMDetection head and can be tested without the legacy MMCV model stack.
All public tensors use batch-first layout: ``[num_rois, num_slots, channels]``.
"""

from __future__ import division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _smooth_l1_beta(pred, target, beta=0.01):
    """Version-stable Smooth L1 with an explicit transition width."""
    diff = (pred - target).abs()
    if beta < 1e-12:
        return diff
    return torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)


def cxcywh_to_xyxy(boxes):
    """Convert boxes from center-size to corner form."""
    half_size = 0.5 * boxes[..., 2:4]
    return torch.cat((boxes[..., :2] - half_size, boxes[..., :2] + half_size),
                     dim=-1)


def _aligned_giou_xyxy(boxes1, boxes2, eps=1e-7):
    """Aligned generalized IoU.  Inputs have identical ``[..., 4]`` shape."""
    boxes1 = boxes1.float()
    boxes2 = boxes2.float()
    inter_lt = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    wh1 = (boxes1[..., 2:] - boxes1[..., :2]).clamp(min=0)
    wh2 = (boxes2[..., 2:] - boxes2[..., :2]).clamp(min=0)
    area1 = wh1[..., 0] * wh1[..., 1]
    area2 = wh2[..., 0] * wh2[..., 1]
    union = (area1 + area2 - inter).clamp(min=eps)
    iou = inter / union

    enclosing_lt = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclosing_rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = (enclosing_wh[..., 0] *
                      enclosing_wh[..., 1]).clamp(min=eps)
    penalty = (enclosing_area - union).clamp(min=0) / enclosing_area
    return (iou - penalty).clamp(min=-1.0, max=1.0)


def pairwise_giou_cxcywh(boxes1, boxes2):
    """Pairwise generalized IoU for two center-size box sets."""
    if boxes1.dim() != 2 or boxes2.dim() != 2:
        raise ValueError('pairwise GIoU expects [M, 4] and [K, 4] tensors')
    if boxes1.size(-1) != 4 or boxes2.size(-1) != 4:
        raise ValueError('box tensors must end in four coordinates')
    xyxy1 = cxcywh_to_xyxy(boxes1.float())[:, None, :]
    xyxy2 = cxcywh_to_xyxy(boxes2.float())[None, :, :]
    return _aligned_giou_xyxy(xyxy1, xyxy2)


def make_anchor_lattice(rows=5,
                        cols=8,
                        base_size=(0.25, 0.25),
                        device=None,
                        dtype=torch.float32):
    """Create the canonical row-major spatial anchor lattice.

    The default is the approved 8-by-5 (40-slot) layout.  Anchors are raw
    normalized ``(cx, cy, w, h)`` boxes, not standardized diffusion states.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError('rows and cols must be positive')
    if isinstance(base_size, (int, float)):
        base_size = (float(base_size), float(base_size))
    ys = (torch.arange(rows, device=device, dtype=dtype) + 0.5) / rows
    xs = (torch.arange(cols, device=device, dtype=dtype) + 0.5) / cols
    yy = ys[:, None].expand(rows, cols)
    xx = xs[None, :].expand(rows, cols)
    ww = xx.new_full(xx.shape, float(base_size[0]))
    hh = xx.new_full(xx.shape, float(base_size[1]))
    return torch.stack((xx, yy, ww, hh), dim=-1).reshape(rows * cols, 4)


class SupportBoxCodec(nn.Module):
    """Encode/decode support boxes in the standardized 4-D diffusion space."""

    def __init__(self, mean, std, eps=1e-6, max_log_size=20.0):
        super(SupportBoxCodec, self).__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        if mean.numel() != 4 or std.numel() != 4:
            raise ValueError('mean and std must each contain four values')
        if bool((std <= 0).any()):
            raise ValueError('all standard deviations must be positive')
        self.register_buffer('mean', mean.reshape(4))
        self.register_buffer('std', std.reshape(4))
        self.eps = float(eps)
        self.max_log_size = float(max_log_size)

    def encode(self, boxes):
        if boxes.size(-1) != 4:
            raise ValueError('boxes must end in (cx, cy, w, h)')
        boxes_f = boxes.float()
        log_size = torch.log(boxes_f[..., 2:4].clamp(min=self.eps) + self.eps)
        geometry = torch.cat((boxes_f[..., :2], log_size), dim=-1)
        mean = self.mean.to(device=boxes.device, dtype=torch.float32)
        std = self.std.to(device=boxes.device, dtype=torch.float32)
        return (geometry - mean) / std

    def decode(self, states):
        if states.size(-1) != 4:
            raise ValueError('states must end in four coordinates')
        mean = self.mean.to(device=states.device, dtype=torch.float32)
        std = self.std.to(device=states.device, dtype=torch.float32)
        geometry = states.float() * std + mean
        min_log = math.log(self.eps)
        log_size = geometry[..., 2:4].clamp(min=min_log, max=self.max_log_size)
        size = (torch.exp(log_size) - self.eps).clamp(min=self.eps)
        return torch.cat((geometry[..., :2], size), dim=-1)

    def decode_xyxy(self, states):
        return cxcywh_to_xyxy(self.decode(states))


def build_corner_support_boxes(primitives,
                               rho=6.0 / 32.0,
                               margin=2.0 / 32.0,
                               min_size=4.0 / 32.0,
                               guard=(-0.25, 1.25),
                               eps=1e-6,
                               validate=True):
    """Build local AABB supports from ``[previous, current, next]`` triples."""
    if primitives.size(-1) != 6:
        raise ValueError('corner primitives must end in six coordinates')
    points = primitives.reshape(primitives.shape[:-1] + (3, 2))
    if validate and points.numel() > 0:
        outside = (points < guard[0]) | (points > guard[1])
        if bool(outside.any()):
            raise ValueError(
                'primitive coordinate lies outside the guard range')

    center = points[..., 1, :]
    neighbors = torch.stack((points[..., 0, :], points[..., 2, :]), dim=-2)
    rays = neighbors - center.unsqueeze(-2)
    ray_lengths = torch.sqrt((rays * rays).sum(dim=-1, keepdim=True) +
                             eps * eps)
    ray_scale = (float(rho) / ray_lengths).clamp(max=1.0)
    clipped = center.unsqueeze(-2) + ray_scale * rays
    local_points = torch.cat(
        (clipped[..., :1, :], center.unsqueeze(-2), clipped[..., 1:, :]),
        dim=-2)

    lower = local_points.min(dim=-2)[0] - float(margin)
    upper = local_points.max(dim=-2)[0] + float(margin)
    box_center = 0.5 * (lower + upper)
    box_size = (upper - lower).clamp(min=float(min_size))
    return torch.cat((box_center, box_size), dim=-1)


class SupportTargetBuilder(object):
    """Construct support targets and one fixed clean anchor coupling."""

    def __init__(self,
                 codec,
                 anchors=None,
                 rho=6.0 / 32.0,
                 margin=2.0 / 32.0,
                 min_size=4.0 / 32.0,
                 guard=(-0.25, 1.25),
                 null_weight=0.1):
        if not isinstance(codec, SupportBoxCodec):
            raise TypeError('codec must be a SupportBoxCodec')
        self.codec = codec
        self.anchors = anchors
        self.rho = float(rho)
        self.margin = float(margin)
        self.min_size = float(min_size)
        self.guard = guard
        self.null_weight = float(null_weight)

    def build_support_boxes(self, primitives, validate=True):
        return build_corner_support_boxes(
            primitives,
            rho=self.rho,
            margin=self.margin,
            min_size=self.min_size,
            guard=self.guard,
            validate=validate)

    def build_coupling(self, primitives, anchors, support_boxes=None):
        """Return fixed targets for one polygon whose primitives are cyclic."""
        if primitives.dim() != 2 or primitives.size(-1) != 6:
            raise ValueError('primitives must have shape [K, 6]')
        if anchors.dim() != 2 or anchors.size(-1) != 4:
            raise ValueError('anchors must have shape [M, 4]')
        if support_boxes is None:
            support_boxes = self.build_support_boxes(primitives)
        if support_boxes.shape != (primitives.size(0), 4):
            raise ValueError('support_boxes must have shape [K, 4]')
        if primitives.size(0) > anchors.size(0):
            raise ValueError('the number of primitives exceeds slot capacity')

        device = primitives.device
        num_slots = anchors.size(0)
        num_gt = primitives.size(0)
        target_boxes = anchors.clone().to(
            device=device, dtype=primitives.dtype)
        target_primitives = primitives.new_zeros((num_slots, 6))
        validity = primitives.new_zeros((num_slots, ))
        matched_anchor = torch.empty((0, ), dtype=torch.long, device=device)
        matched_gt = torch.empty((0, ), dtype=torch.long, device=device)

        if num_gt > 0:
            anchor_f = anchors.detach().to(device=device, dtype=torch.float32)
            support_f = support_boxes.detach().float()
            center_cost = torch.cdist(anchor_f[:, :2], support_f[:, :2], p=1)
            size_cost = torch.cdist(
                torch.log(anchor_f[:, 2:4].clamp(min=1e-6)),
                torch.log(support_f[:, 2:4].clamp(min=1e-6)),
                p=1)
            giou_cost = 1.0 - pairwise_giou_cxcywh(anchor_f, support_f)
            cost = 4.0 * center_cost + 2.0 * giou_cost + size_cost
            row, col = linear_sum_assignment(
                cost.detach().cpu().double().numpy())
            matched_anchor = torch.as_tensor(
                row, dtype=torch.long, device=device)
            matched_gt = torch.as_tensor(col, dtype=torch.long, device=device)
            target_boxes[matched_anchor] = support_boxes[matched_gt]
            target_primitives[matched_anchor] = primitives[matched_gt]
            validity[matched_anchor] = 1.0

        target_state = self.codec.encode(target_boxes)
        successors_cw = torch.full((num_slots, ),
                                   -1,
                                   dtype=torch.long,
                                   device=device)
        successors_ccw = successors_cw.clone()
        slot_for_gt = torch.full((num_gt, ),
                                 -1,
                                 dtype=torch.long,
                                 device=device)
        if num_gt > 0:
            slot_for_gt[matched_gt] = matched_anchor
            for gt_index in range(num_gt):
                slot = slot_for_gt[gt_index]
                successors_cw[slot] = slot_for_gt[(gt_index + 1) % num_gt]
                successors_ccw[slot] = slot_for_gt[(gt_index - 1) % num_gt]

        geometry_weights = (validity + self.null_weight * (1.0 - validity))
        return {
            'target_boxes': target_boxes,
            'target_state': target_state,
            'target_primitives': target_primitives,
            'validity': validity,
            'geometry_weights': geometry_weights,
            'successors_cw': successors_cw,
            'successors_ccw': successors_ccw,
            'matched_anchor_indices': matched_anchor,
            'matched_gt_indices': matched_gt,
            'slot_for_gt': slot_for_gt,
        }

    def build(self, primitives, anchors=None):
        """Build a batched clean coupling from variable-length primitives.

        Args:
            primitives (Tensor or sequence[Tensor]): One ``[K, 6]`` tensor or
                a sequence containing one tensor per RoI.
            anchors (Tensor, optional): Raw ``[M, 4]`` cxcywh anchors.  If
                omitted, the constructor anchors or the default 8x5 lattice
                are used.

        Returns:
            dict: Batched ``z0``, active mask, primitive and successor targets,
            target boxes, geometry weights, and the shared raw anchors.
        """
        if torch.is_tensor(primitives):
            primitive_list = [primitives]
        else:
            primitive_list = list(primitives)
        if len(primitive_list) == 0:
            raise ValueError('at least one RoI primitive tensor is required')
        reference = primitive_list[0]
        if anchors is None:
            anchors = self.anchors
        if anchors is None:
            anchors = make_anchor_lattice(
                device=reference.device, dtype=reference.dtype)
        anchors = anchors.to(device=reference.device, dtype=reference.dtype)
        coupled = [
            self.build_coupling(item, anchors) for item in primitive_list
        ]

        def stack(key):
            return torch.stack([item[key] for item in coupled], dim=0)

        return {
            'z0': stack('target_state'),
            'active_mask': stack('validity').bool(),
            'primitive_targets': stack('target_primitives'),
            'successors_cw': stack('successors_cw'),
            'successors_ccw': stack('successors_ccw'),
            'target_boxes': stack('target_boxes'),
            'geometry_weights': stack('geometry_weights'),
            'anchors': anchors,
            'per_instance': coupled,
        }


def permute_coupled_targets(targets, permutation):
    """Jointly permute slots and correctly remap successor indices."""
    permutation = permutation.long()
    num_slots = permutation.numel()
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        num_slots, dtype=torch.long, device=permutation.device)
    output = {}
    slot_keys = ('target_boxes', 'target_state', 'target_primitives',
                 'validity', 'geometry_weights')
    for key in slot_keys:
        if key in targets:
            output[key] = targets[key][permutation]
    for key in ('successors_cw', 'successors_ccw'):
        if key in targets:
            successor = targets[key][permutation]
            valid = successor >= 0
            remapped = successor.clone()
            remapped[valid] = inverse[successor[valid]]
            output[key] = remapped
    for key, value in targets.items():
        if key not in output and key not in slot_keys and key not in (
                'successors_cw', 'successors_ccw'):
            output[key] = value
    return output


def _cosine_betas(num_timesteps=1000, offset=0.008, max_beta=0.999):
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
    phase = ((steps / float(num_timesteps) + offset) /
             (1.0 + offset)) * math.pi / 2.0
    cumulative = torch.cos(phase).pow(2)
    cumulative = cumulative / cumulative[0].clone()
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(min=1e-8, max=max_beta).float()


class CosineDiffusionSchedule(nn.Module):
    """Cosine VP schedule plus deterministic DDIM utilities."""

    def __init__(self, num_timesteps=1000, offset=0.008, max_beta=0.999):
        super(CosineDiffusionSchedule, self).__init__()
        if num_timesteps < 2:
            raise ValueError('num_timesteps must be at least two')
        betas = _cosine_betas(num_timesteps, offset, max_beta)
        alpha_bars = torch.cumprod(1.0 - betas.double(), dim=0).float()
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.num_timesteps = int(num_timesteps)

    def _normalize_timesteps(self, timesteps, batch_size, device):
        if not torch.is_tensor(timesteps):
            timesteps = torch.full((batch_size, ),
                                   int(timesteps),
                                   dtype=torch.long,
                                   device=device)
        else:
            timesteps = timesteps.to(device=device, dtype=torch.long)
            if timesteps.dim() == 0:
                timesteps = timesteps.expand(batch_size)
        if timesteps.shape != (batch_size, ):
            raise ValueError('timesteps must be scalar or have shape [N]')
        if bool(((timesteps < 0) | (timesteps >= self.num_timesteps)).any()):
            raise ValueError('diffusion timestep is outside the schedule')
        return timesteps

    def alpha_bar(self, timesteps, reference):
        timesteps = self._normalize_timesteps(timesteps, reference.size(0),
                                              reference.device)
        values = self.alpha_bars.float().index_select(0, timesteps)
        shape = (reference.size(0), ) + (1, ) * (reference.dim() - 1)
        return values.reshape(shape)

    def log_snr(self, timesteps, batch_size=None, device=None):
        if torch.is_tensor(timesteps):
            if timesteps.dim() == 0:
                if batch_size is None:
                    batch_size = 1
            else:
                batch_size = timesteps.numel()
            if device is None:
                device = timesteps.device
        if batch_size is None:
            raise ValueError('batch_size is required for a scalar timestep')
        if device is None:
            device = self.alpha_bars.device
        dummy = torch.empty((batch_size, 1), device=device)
        alpha_bar = self.alpha_bar(timesteps, dummy).reshape(batch_size)
        alpha_bar = alpha_bar.clamp(min=1e-12, max=1.0 - 1e-7)
        return torch.log(alpha_bar) - torch.log1p(-alpha_bar)

    def q_sample(self, clean_state, timesteps, noise=None):
        if noise is None:
            noise = torch.randn(
                clean_state.shape,
                dtype=clean_state.dtype,
                device=clean_state.device)
        if noise.shape != clean_state.shape:
            raise ValueError('noise and clean_state must have identical shape')
        clean_f = clean_state.float()
        noise_f = noise.float()
        alpha_bar = self.alpha_bar(timesteps, clean_f)
        return (torch.sqrt(alpha_bar) * clean_f + torch.sqrt(
            (1.0 - alpha_bar).clamp(min=0)) * noise_f)

    def ddim_step(self, state, pred_clean, timesteps, next_timestep=None):
        """One deterministic DDIM step; ``None`` denotes virtual clean."""
        state_f = state.float()
        pred_f = pred_clean.float()
        if state_f.shape != pred_f.shape:
            raise ValueError('state and pred_clean must have identical shape')
        alpha_t = self.alpha_bar(timesteps, state_f)
        pred_noise = ((state_f - torch.sqrt(alpha_t) * pred_f) / torch.sqrt(
            (1.0 - alpha_t).clamp(min=1e-12)))
        if next_timestep is None or int(next_timestep) < 0:
            return pred_f
        alpha_s = self.alpha_bar(int(next_timestep), state_f)
        return (torch.sqrt(alpha_s) * pred_f + torch.sqrt(
            (1.0 - alpha_s).clamp(min=0)) * pred_noise)

    def inference_timesteps(self, num_steps):
        if num_steps <= 0 or num_steps > self.num_timesteps:
            raise ValueError('num_steps is outside the valid range')
        return tuple(
            int((self.num_timesteps - 1) -
                i * self.num_timesteps / float(num_steps))
            for i in range(num_steps))


def _sinusoidal_embedding(values, dim, max_period=10000.0):
    if dim <= 0:
        raise ValueError('embedding dimension must be positive')
    values = values.float().reshape(-1)
    half = dim // 2
    if half == 0:
        return values[:, None]
    frequencies = torch.exp(
        -math.log(max_period) *
        torch.arange(half, device=values.device, dtype=torch.float32) /
        float(half))
    arguments = values[:, None] * frequencies[None, :]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=1)
    if dim % 2:
        embedding = torch.cat(
            (embedding, embedding.new_zeros((embedding.size(0), 1))), dim=1)
    return embedding


class _SelfAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(_SelfAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads,
                              self.head_dim).permute(0, 2, 1, 3)

    def forward(self, tensor):
        batch, length, _ = tensor.shape
        qkv = self.qkv(tensor).reshape(batch, length, 3, self.num_heads,
                                       self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * self.scale
        attention = self.dropout(torch.softmax(scores, dim=-1)).to(value.dtype)
        output = torch.matmul(attention, value)
        output = output.permute(0, 2, 1, 3).reshape(batch, length,
                                                    self.hidden_dim)
        return self.output(output)


class _CrossAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super(_CrossAttention, self).__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        for projection in (self.query, self.key, self.value, self.output):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def _heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.num_heads,
                              self.head_dim).permute(0, 2, 1, 3)

    def project_condition(self, memory, position):
        if memory.shape != position.shape:
            raise ValueError('memory and position must have identical shape')
        return (self._heads(self.key(memory + position)),
                self._heads(self.value(memory)))

    def forward(self, tensor, condition_cache):
        key, value = condition_cache
        query = self._heads(self.query(tensor))
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * self.scale
        attention = self.dropout(torch.softmax(scores, dim=-1)).to(value.dtype)
        output = torch.matmul(attention, value)
        batch, _, length, _ = output.shape
        output = output.permute(0, 2, 1, 3).reshape(batch, length,
                                                    self.hidden_dim)
        return self.output(output)


class _AdaLNZeroBlock(nn.Module):

    def __init__(self, hidden_dim=256, num_heads=8, ffn_dim=1024, dropout=0.0):
        super(_AdaLNZeroBlock, self).__init__()
        self.norm_self = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attention = _SelfAttention(hidden_dim, num_heads, dropout)
        self.cross_attention = _CrossAttention(hidden_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim))
        self.modulation = nn.Sequential(nn.SiLU(),
                                        nn.Linear(hidden_dim, 9 * hidden_dim))
        nn.init.constant_(self.modulation[-1].weight, 0.0)
        nn.init.constant_(self.modulation[-1].bias, 0.0)

    @staticmethod
    def _modulate(tensor, shift, scale):
        return tensor * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def project_condition(self, memory, position):
        return self.cross_attention.project_condition(memory, position)

    def forward(self, hidden, condition, condition_cache):
        modulation = self.modulation(condition)
        chunks = modulation.chunk(9, dim=-1)
        shift_sa, scale_sa, gate_sa = chunks[0:3]
        shift_ca, scale_ca, gate_ca = chunks[3:6]
        shift_ffn, scale_ffn, gate_ffn = chunks[6:9]

        branch = self._modulate(self.norm_self(hidden), shift_sa, scale_sa)
        hidden = hidden + gate_sa[:, None, :] * self.self_attention(branch)
        branch = self._modulate(self.norm_cross(hidden), shift_ca, scale_ca)
        hidden = hidden + gate_ca[:, None, :] * self.cross_attention(
            branch, condition_cache)
        branch = self._modulate(self.norm_ffn(hidden), shift_ffn, scale_ffn)
        hidden = hidden + gate_ffn[:, None, :] * self.ffn(branch)
        return hidden


class CoarseToFineSetDenoiser(nn.Module):
    """Three-block P4-to-P2 set denoiser with cached condition K/V."""

    def __init__(self,
                 num_slots=40,
                 state_dim=4,
                 hidden_dim=256,
                 num_heads=8,
                 ffn_dim=1024,
                 time_embed_dim=128,
                 num_timesteps=1000,
                 dropout=0.0):
        super(CoarseToFineSetDenoiser, self).__init__()
        self.num_slots = int(num_slots)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, 128), nn.SiLU(), nn.Linear(128, hidden_dim))
        self.anchor_embed = nn.Sequential(
            nn.Linear(4, 128), nn.SiLU(), nn.Linear(128, hidden_dim))
        self.time_embed_dim = int(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.image_projection = nn.Linear(hidden_dim, hidden_dim)
        self.schedule = CosineDiffusionSchedule(num_timesteps=num_timesteps)
        self.blocks = nn.ModuleList([
            _AdaLNZeroBlock(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(3)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, state_dim)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.output_head.bias, 0.0)

    def prepare_condition(self, memories, pos):
        if len(memories) != 3 or len(pos) != 3:
            raise ValueError(
                'exactly three coarse-to-fine memories are required')
        caches = []
        for block, memory, position in zip(self.blocks, memories, pos):
            if memory.dim() != 3 or memory.size(-1) != self.hidden_dim:
                raise ValueError('each memory must have shape [N, L, D]')
            caches.append(block.project_condition(memory, position))
        return tuple(caches)

    def _time_condition(self, timesteps, batch_size, device):
        if torch.is_tensor(timesteps) and timesteps.dtype.is_floating_point:
            log_snr = timesteps.to(device=device, dtype=torch.float32)
            if log_snr.dim() == 0:
                log_snr = log_snr.expand(batch_size)
            if log_snr.shape != (batch_size, ):
                raise ValueError('floating t must be log-SNR with shape [N]')
        else:
            log_snr = self.schedule.log_snr(
                timesteps, batch_size=batch_size, device=device)
        return self.time_mlp(
            _sinusoidal_embedding(log_snr, self.time_embed_dim))

    def forward(self,
                z_t,
                t,
                anchors,
                memories,
                pos,
                c_img,
                condition_cache=None):
        """Return ``(z0_hat, h3)``.

        Integer ``t`` denotes schedule indices. Floating ``t`` is interpreted
        as precomputed log-SNR. ``anchors`` are raw normalized cxcywh boxes.
        """
        if z_t.dim() != 3 or z_t.shape[1:] != (self.num_slots, self.state_dim):
            raise ValueError('z_t must have shape [N, num_slots, state_dim]')
        batch_size = z_t.size(0)
        if anchors.dim() == 2:
            anchors = anchors.unsqueeze(0)
        if anchors.shape[1:] != (self.num_slots, 4):
            raise ValueError('anchors must have shape [M,4] or [N,M,4]')
        if anchors.size(0) not in (1, batch_size):
            raise ValueError('anchor batch dimension must be one or N')
        if c_img.shape != (batch_size, self.hidden_dim):
            raise ValueError('c_img must have shape [N, D]')
        if condition_cache is None:
            condition_cache = self.prepare_condition(memories, pos)
        if len(condition_cache) != 3:
            raise ValueError('condition cache must contain three levels')

        hidden = self.state_embed(z_t) + self.anchor_embed(
            anchors.to(device=z_t.device, dtype=z_t.dtype))
        condition = (
            self._time_condition(t, batch_size, z_t.device) +
            self.image_projection(c_img))
        for block, cache in zip(self.blocks, condition_cache):
            hidden = block(hidden, condition, cache)
        pred_clean = self.output_head(self.output_norm(hidden))
        return pred_clean, hidden


def ddim_sample(denoiser,
                schedule,
                initial_noise,
                anchors,
                memories,
                pos,
                c_img,
                num_steps=2):
    """Run the approved deterministic 1/2/4-NFE sampler."""
    timesteps = schedule.inference_timesteps(num_steps)
    cache = denoiser.prepare_condition(memories, pos)
    state = initial_noise.float()
    final_hidden = None
    for index, timestep in enumerate(timesteps):
        t_batch = torch.full((state.size(0), ),
                             timestep,
                             dtype=torch.long,
                             device=state.device)
        pred_clean, final_hidden = denoiser(
            state,
            t_batch,
            anchors,
            memories,
            pos,
            c_img,
            condition_cache=cache)
        next_timestep = (
            timesteps[index + 1] if index + 1 < len(timesteps) else None)
        state = schedule.ddim_step(state, pred_clean, t_batch, next_timestep)
    return state, final_hidden


class FinalPrimitiveHead(nn.Module):
    """Final-only differentiable local sampling and corner recovery."""

    def __init__(self, hidden_dim=256, grid_size=4):
        super(FinalPrimitiveHead, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)
        self.local_projection = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.primitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 6))
        self.validity_head = nn.Linear(hidden_dim, 1)
        self.last_oob_rate = torch.empty(0)

    def _sample_local(self, dense_map, boxes):
        batch, channels, _, _ = dense_map.shape
        if boxes.dim() != 3 or boxes.size(0) != batch or boxes.size(-1) != 4:
            raise ValueError('boxes must have shape [N, M, 4]')
        num_slots = boxes.size(1)
        offsets = (
            (torch.arange(
                self.grid_size, dtype=boxes.dtype, device=boxes.device) + 0.5)
            / float(self.grid_size) - 0.5)
        offset_x = offsets.view(1, 1, 1, self.grid_size)
        offset_y = offsets.view(1, 1, self.grid_size, 1)
        grid_x = (
            boxes[..., 0, None, None] + boxes[..., 2, None, None] * offset_x)
        grid_y = (
            boxes[..., 1, None, None] + boxes[..., 3, None, None] * offset_y)
        grid_x = grid_x.expand(batch, num_slots, self.grid_size,
                               self.grid_size)
        grid_y = grid_y.expand(batch, num_slots, self.grid_size,
                               self.grid_size)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        self.last_oob_rate = ((grid < 0.0) | (grid > 1.0)).any(
            dim=-1).float().mean(dim=(1, 2, 3)).detach()
        grid = (2.0 * grid - 1.0).reshape(batch, num_slots * self.grid_size,
                                          self.grid_size, 2)
        sampled = F.grid_sample(
            dense_map,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampled = sampled.reshape(batch, channels, num_slots, self.grid_size,
                                  self.grid_size)
        return sampled.permute(0, 2, 1, 3, 4).contiguous()

    def forward(self, h3, boxes, c32):
        if h3.dim() != 3 or h3.size(-1) != self.hidden_dim:
            raise ValueError('h3 must have shape [N, M, D]')
        if c32.dim() != 4 or c32.size(1) != self.hidden_dim:
            raise ValueError('c32 must have shape [N, D, H, W]')
        sampled = self._sample_local(c32, boxes)
        local = sampled.mean(dim=(-1, -2))
        fused = self.fusion_norm(h3 + self.local_projection(local))
        primitives = 1.5 * torch.sigmoid(self.primitive_head(fused)) - 0.25
        validity_logits = self.validity_head(fused).squeeze(-1)
        return primitives, validity_logits, fused


class SuccessorTopologyHead(nn.Module):
    """Pairwise neural successor scorer."""

    def __init__(self, hidden_dim=256, topology_dim=64):
        super(SuccessorTopologyHead, self).__init__()
        self.query = nn.Linear(hidden_dim, topology_dim)
        self.key = nn.Linear(hidden_dim, topology_dim)
        self.scale = topology_dim**-0.5
        self.geometry_mlp = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def _normalize(vector, eps=1e-6):
        norm = torch.sqrt((vector * vector).sum(dim=-1, keepdim=True) +
                          eps * eps)
        return vector / norm

    def forward(self, features, primitives):
        if features.dim() != 3 or primitives.shape[:2] != features.shape[:2]:
            raise ValueError('features and primitives must share [N, M]')
        if primitives.size(-1) != 6:
            raise ValueError('primitives must end in six coordinates')
        query = self.query(features)
        key = self.key(features)
        content = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        previous = primitives[..., 0:2]
        centers = primitives[..., 2:4]
        following = primitives[..., 4:6]
        outgoing = self._normalize(following - centers)
        incoming = self._normalize(centers - previous)
        delta = centers.unsqueeze(1) - centers.unsqueeze(2)
        out_pair = outgoing.unsqueeze(2).expand(-1, -1, centers.size(1), -1)
        in_pair = incoming.unsqueeze(1).expand(-1, centers.size(1), -1, -1)
        distance = torch.sqrt((delta * delta).sum(dim=-1, keepdim=True) +
                              1e-12)
        geometry = torch.cat((delta, out_pair, in_pair, distance), dim=-1)
        logits = content + self.geometry_mlp(geometry).squeeze(-1)
        diagonal = torch.eye(
            logits.size(-1), dtype=torch.bool,
            device=logits.device).unsqueeze(0)
        return logits.masked_fill(diagonal, -1e4)


def masked_log_sinkhorn(logits, active, num_iters=20):
    """Stable masked log-Sinkhorn with isolated dummy edges for empty rows."""
    if logits.dim() != 3 or logits.size(-1) != logits.size(-2):
        raise ValueError('logits must have shape [N, M, M]')
    if active.shape != logits.shape[:2]:
        raise ValueError('active mask must have shape [N, M]')
    values = logits.float()
    active = active.bool()
    num_slots = logits.size(-1)
    diagonal = torch.eye(
        num_slots, dtype=torch.bool, device=logits.device).unsqueeze(0)
    edge_mask = (active.unsqueeze(2) & active.unsqueeze(1) & ~diagonal)
    empty_rows = ~edge_mask.any(dim=-1)
    safe_mask = edge_mask | (empty_rows.unsqueeze(-1) & diagonal)
    values = values.masked_fill(~safe_mask, float('-inf'))
    for _ in range(int(num_iters)):
        values = values - torch.logsumexp(values, dim=-1, keepdim=True)
        values = values - torch.logsumexp(values, dim=-2, keepdim=True)
    return values.masked_fill(~edge_mask, float('-inf'))


def _validate_successor_cycle(successor, active):
    indices = torch.nonzero(active, as_tuple=False).reshape(-1)
    if indices.numel() < 3:
        return
    targets = successor[indices]
    if bool((targets < 0).any()) or bool((targets >= active.numel()).any()):
        raise ValueError('active successor index is out of range')
    if not bool(active[targets].all()):
        raise ValueError('an active successor points to an inactive slot')
    if bool((targets == indices).any()):
        raise ValueError('self successors are not valid topology targets')
    if torch.unique(targets).numel() != indices.numel():
        raise ValueError('successor targets must be a permutation')
    visited = set()
    current = int(indices[0].item())
    for _ in range(indices.numel()):
        if current in visited:
            break
        visited.add(current)
        current = int(successor[current].item())
    if len(visited) != indices.numel() or current != int(indices[0].item()):
        raise ValueError('successor targets must form one cycle')


def topology_loss(logits,
                  active,
                  successors_cw,
                  successors_ccw,
                  num_iters=20,
                  validate=True):
    """Bidirectional per-instance successor NLL after masked Sinkhorn."""
    active = active.bool()
    if (successors_cw.shape != active.shape or
            successors_ccw.shape != active.shape):
        raise ValueError('successor tensors must have shape [N, M]')
    eligible = active.sum(dim=-1) >= 3
    if validate:
        for batch_index in range(active.size(0)):
            if bool(eligible[batch_index]):
                _validate_successor_cycle(successors_cw[batch_index],
                                          active[batch_index])
                _validate_successor_cycle(successors_ccw[batch_index],
                                          active[batch_index])
    if not bool(eligible.any()):
        return logits.sum() * 0.0
    log_prob = masked_log_sinkhorn(logits, active, num_iters=num_iters)
    num_slots = logits.size(-1)
    counts = active.float().sum(dim=-1).clamp(min=1.0)

    def orientation_nll(successors):
        indices = successors.clamp(min=0, max=num_slots - 1).long()
        chosen = log_prob.gather(2, indices.unsqueeze(-1)).squeeze(-1)
        chosen = chosen.masked_fill(~active, 0.0)
        return -chosen.sum(dim=-1) / counts

    clockwise = orientation_nll(successors_cw)
    counterclockwise = orientation_nll(successors_ccw)
    pair = torch.stack((clockwise, counterclockwise), dim=-1)
    selected = pair.min(dim=-1)[0]
    ties = clockwise == counterclockwise
    selected = torch.where(ties, pair.mean(dim=-1), selected)
    return selected[eligible].mean()


def _segments_intersect(a, b, c, d, eps=1e-9):

    def orient(p, q, r):
        return float((q[0] - p[0]) * (r[1] - p[1]) -
                     (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p, q, r):
        return (min(float(p[0]), float(q[0])) - eps <= float(r[0]) <=
                max(float(p[0]), float(q[0])) + eps and
                min(float(p[1]), float(q[1])) - eps <= float(r[1]) <=
                max(float(p[1]), float(q[1])) + eps)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True
    return ((abs(o1) <= eps and on_segment(a, b, c)) or
            (abs(o2) <= eps and on_segment(a, b, d)) or
            (abs(o3) <= eps and on_segment(c, d, a)) or
            (abs(o4) <= eps and on_segment(c, d, b)))


def _extract_cycles(successor, active_indices):
    remaining = set(int(index) for index in active_indices)
    cycles = []
    while remaining:
        start = min(remaining)
        cycle = []
        current = start
        while current not in cycle:
            cycle.append(current)
            remaining.discard(current)
            current = successor[current]
        cycles.append(cycle)
    return cycles


def solve_cycles(logits,
                 primitives,
                 validity_logits,
                 validity_threshold=0.5,
                 duplicate_distance=1.0 / 64.0,
                 max_2opt_iterations=100):
    """Hard one-cycle reconciliation for inference.

    Returns one diagnostics dictionary per RoI.  Hungarian, subtour merging,
    and 2-opt are intentionally outside autograd.
    """
    if logits.dim() != 3 or primitives.shape[:2] != logits.shape[:2]:
        raise ValueError('topology tensors must share [N, M] dimensions')
    results = []
    with torch.no_grad():
        probabilities = torch.sigmoid(validity_logits.float())
        centers_all = primitives[..., 2:4].float()
        for batch_index in range(logits.size(0)):
            if (not bool(torch.isfinite(logits[batch_index]).all()) or
                    not bool(
                        torch.isfinite(validity_logits[batch_index]).all()) or
                    not bool(torch.isfinite(primitives[batch_index]).all())):
                results.append({
                    'valid': False,
                    'indices': torch.empty(
                        (0, ), dtype=torch.long, device=logits.device),
                    'subtour_count': 0,
                    'merge_count': 0,
                    'two_opt_count': 0,
                    'failure': 'nonfinite_geometry',
                })
                continue
            candidates = torch.nonzero(
                probabilities[batch_index] > validity_threshold,
                as_tuple=False).reshape(-1)
            ranked = sorted(
                [int(x) for x in candidates],
                key=lambda x: -float(probabilities[batch_index, x]))
            active = []
            for index in ranked:
                if all(
                        float(
                            torch.norm(centers_all[batch_index, index] -
                                       centers_all[batch_index, kept])) >=
                        duplicate_distance for kept in active):
                    active.append(index)
            if len(active) < 3:
                results.append({
                    'valid':
                    False,
                    'indices':
                    torch.empty((0, ), dtype=torch.long, device=logits.device),
                    'subtour_count':
                    0,
                    'merge_count':
                    0,
                    'two_opt_count':
                    0,
                    'failure':
                    'fewer_than_three_vertices',
                })
                continue

            active_tensor = torch.as_tensor(
                active, dtype=torch.long, device=logits.device)
            scores = logits[batch_index][active_tensor][:, active_tensor]
            cost = -scores.detach().cpu().double().numpy()
            for diagonal_index in range(len(active)):
                cost[diagonal_index, diagonal_index] = 1e12
            rows, cols = linear_sum_assignment(cost)
            successor = {
                active[int(row)]: active[int(col)]
                for row, col in zip(rows, cols)
            }
            cycles = _extract_cycles(successor, active)
            initial_subtours = len(cycles)
            merge_count = 0
            while len(cycles) > 1:
                first, second = cycles[0], cycles[1]
                best = None
                best_delta = -float('inf')
                for a in first:
                    b = successor[a]
                    for c in second:
                        d = successor[c]
                        delta = (
                            float(logits[batch_index, a, d]) +
                            float(logits[batch_index, c, b]) -
                            float(logits[batch_index, a, b]) -
                            float(logits[batch_index, c, d]))
                        if delta > best_delta:
                            best_delta = delta
                            best = (a, b, c, d)
                a, b, c, d = best
                successor[a] = d
                successor[c] = b
                merge_count += 1
                cycles = _extract_cycles(successor, active)

            start = max(
                active, key=lambda x: float(probabilities[batch_index, x]))
            order = [start]
            while len(order) < len(active):
                order.append(successor[order[-1]])
            points = centers_all[batch_index]
            two_opt_count = 0
            changed = True
            while changed and two_opt_count < max_2opt_iterations:
                changed = False
                num_points = len(order)
                for i in range(num_points):
                    i_next = (i + 1) % num_points
                    for j in range(i + 2, num_points):
                        j_next = (j + 1) % num_points
                        if i == 0 and j_next == 0:
                            continue
                        if _segments_intersect(points[order[i]],
                                               points[order[i_next]],
                                               points[order[j]],
                                               points[order[j_next]]):
                            order[i_next:j + 1] = reversed(order[i_next:j + 1])
                            two_opt_count += 1
                            changed = True
                            break
                    if changed:
                        break

            polygon = points[torch.as_tensor(
                order, dtype=torch.long, device=points.device)]
            shifted = torch.roll(polygon, shifts=-1, dims=0)
            signed_area = 0.5 * torch.sum(polygon[:, 0] * shifted[:, 1] -
                                          shifted[:, 0] * polygon[:, 1])
            if abs(float(signed_area)) <= 1e-8:
                results.append({
                    'valid': False,
                    'indices': torch.as_tensor(
                        order, dtype=torch.long, device=logits.device),
                    'subtour_count': initial_subtours,
                    'merge_count': merge_count,
                    'two_opt_count': two_opt_count,
                    'failure': 'degenerate_polygon',
                })
                continue
            if float(signed_area) > 0:
                order = list(reversed(order))
            remaining_crossing = False
            for i in range(len(order)):
                for j in range(i + 2, len(order)):
                    if i == 0 and (j + 1) % len(order) == 0:
                        continue
                    if _segments_intersect(points[order[i]],
                                           points[order[(i + 1) % len(order)]],
                                           points[order[j]],
                                           points[order[(j + 1) %
                                                        len(order)]]):
                        remaining_crossing = True
            results.append({
                'valid':
                not remaining_crossing,
                'indices':
                torch.as_tensor(order, dtype=torch.long, device=logits.device),
                'subtour_count':
                initial_subtours,
                'merge_count':
                merge_count,
                'two_opt_count':
                two_opt_count,
                'failure':
                ('intersection_cap' if remaining_crossing else None),
            })
    return results


def _masked_mean(values, mask):
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def primitive_reversal_loss(pred,
                            target,
                            active,
                            beta=0.01,
                            edge_threshold=1e-4,
                            eps=1e-6):
    """V3 coordinate, center, and incident-angle primitive objective."""
    pred = pred.float()
    target = target.float()
    active = active.bool()
    target_reverse = torch.cat(
        (target[..., 4:6], target[..., 2:4], target[..., 0:2]), dim=-1)

    def branch(branch_target):
        coordinate = _smooth_l1_beta(pred, branch_target, beta).mean(dim=-1)
        center = _smooth_l1_beta(pred[..., 2:4], branch_target[..., 2:4],
                                 beta).mean(dim=-1)
        pred_minus = pred[..., 2:4] - pred[..., 0:2]
        pred_plus = pred[..., 4:6] - pred[..., 2:4]
        gt_minus = branch_target[..., 2:4] - branch_target[..., 0:2]
        gt_plus = branch_target[..., 4:6] - branch_target[..., 2:4]

        def angle_term(pred_direction, gt_direction):
            pred_sq = (pred_direction * pred_direction).sum(dim=-1)
            gt_sq = (gt_direction * gt_direction).sum(dim=-1)
            valid = gt_sq > edge_threshold * edge_threshold
            denominator = (
                pred_sq.clamp(min=eps * eps).sqrt() *
                gt_sq.clamp(min=eps * eps).sqrt())
            cosine = ((pred_direction * gt_direction).sum(dim=-1) /
                      denominator).clamp(
                          min=-1.0, max=1.0)
            return 1.0 - cosine, valid

        minus_loss, minus_valid = angle_term(pred_minus, gt_minus)
        plus_loss, plus_valid = angle_term(pred_plus, gt_plus)
        angle_count = (minus_valid.float() + plus_valid.float()).clamp(min=1.0)
        angle = (minus_loss * minus_valid.float() +
                 plus_loss * plus_valid.float()) / angle_count
        return coordinate + 0.5 * center + 0.5 * angle

    forward = branch(target)
    reverse = branch(target_reverse)
    pair = torch.stack((forward, reverse), dim=-1)
    best = pair.min(dim=-1)[0]
    ties = forward == reverse
    best = torch.where(ties, pair.mean(dim=-1), best)
    return _masked_mean(best, active)


def _sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    targets = targets.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none')
    probability_t = probability * targets + (1.0 - probability) * (1.0 -
                                                                   targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * (1.0 - probability_t).pow(gamma) * cross_entropy


class P2PFormerV3PrimitiveLoss(nn.Module):
    """Standalone explicit V3 reversal/center/angle primitive loss."""

    def __init__(self, beta=0.01, edge_threshold=1e-4, eps=1e-6):
        super(P2PFormerV3PrimitiveLoss, self).__init__()
        self.beta = float(beta)
        self.edge_threshold = float(edge_threshold)
        self.eps = float(eps)

    def forward(self, pred, target, active):
        return primitive_reversal_loss(
            pred,
            target,
            active,
            beta=self.beta,
            edge_threshold=self.edge_threshold,
            eps=self.eps)


class P2PFormerV3Loss(nn.Module):
    """Composite geometry/validity/topology loss from the Micro Design."""

    def __init__(self,
                 x0_weight=1.0,
                 support_weight=2.0,
                 validity_weight=1.0,
                 primitive_weight=5.0,
                 topology_weight=1.0,
                 null_geometry_weight=0.1,
                 sinkhorn_iters=20):
        super(P2PFormerV3Loss, self).__init__()
        self.x0_weight = float(x0_weight)
        self.support_weight = float(support_weight)
        self.validity_weight = float(validity_weight)
        self.primitive_weight = float(primitive_weight)
        self.topology_weight = float(topology_weight)
        self.null_geometry_weight = float(null_geometry_weight)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.primitive_loss = P2PFormerV3PrimitiveLoss()

    def forward(self, pred_state, target_state, pred_boxes, target_boxes,
                validity_logits, validity_targets, pred_primitives,
                target_primitives, successor_logits, successors_cw,
                successors_ccw):
        active = validity_targets > 0.5
        state_cost = _smooth_l1_beta(
            pred_state.float(), target_state.float(), beta=0.01).mean(-1)
        state_weights = (
            active.float() + self.null_geometry_weight * (~active).float())
        loss_x0 = (state_cost *
                   state_weights).sum() / state_weights.sum().clamp(min=1.0)

        box_l1 = (pred_boxes.float() - target_boxes.float()).abs().mean(dim=-1)
        box_giou = 1.0 - _aligned_giou_xyxy(
            cxcywh_to_xyxy(pred_boxes), cxcywh_to_xyxy(target_boxes))
        loss_support = _masked_mean(box_l1 + box_giou, active)
        loss_validity = _sigmoid_focal_loss(validity_logits.float(),
                                            validity_targets.float()).mean()
        loss_primitive = self.primitive_loss(pred_primitives,
                                             target_primitives, active)
        loss_topology = topology_loss(
            successor_logits,
            active,
            successors_cw,
            successors_ccw,
            num_iters=self.sinkhorn_iters,
            validate=True)
        total = (
            self.x0_weight * loss_x0 + self.support_weight * loss_support +
            self.validity_weight * loss_validity +
            self.primitive_weight * loss_primitive +
            self.topology_weight * loss_topology)
        return {
            'loss_x0': loss_x0,
            'loss_sbox': loss_support,
            'loss_valid': loss_validity,
            'loss_primitive': loss_primitive,
            'loss_topology': loss_topology,
            'loss_total': total,
        }


__all__ = [
    'SupportBoxCodec', 'SupportTargetBuilder', 'CosineDiffusionSchedule',
    'CoarseToFineSetDenoiser', 'FinalPrimitiveHead', 'SuccessorTopologyHead',
    'P2PFormerV3PrimitiveLoss', 'P2PFormerV3Loss',
    'build_corner_support_boxes', 'cxcywh_to_xyxy', 'ddim_sample',
    'make_anchor_lattice', 'masked_log_sinkhorn', 'pairwise_giou_cxcywh',
    'permute_coupled_targets', 'primitive_reversal_loss', 'solve_cycles',
    'topology_loss'
]
