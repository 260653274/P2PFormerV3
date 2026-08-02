"""MMDetection integration head for P2PFormerV3 V1.

The head keeps the public contract of :class:`P2PFormerHead` while replacing
its one-shot primitive decoder with contour-conditioned support-box diffusion.
All generator geometry is normalized in one shared, un-clipped outer RoI.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models.builder import HEADS

from .p2pformer_v3_contour import ContourEvidenceConditioner
from .p2pformer_v3_diffusion import (
    CoarseToFineSetDenoiser, CosineDiffusionSchedule, FinalPrimitiveHead,
    P2PFormerV3Loss, SuccessorTopologyHead, SupportBoxCodec,
    SupportTargetBuilder, ddim_sample, make_anchor_lattice,
    masked_log_sinkhorn, solve_cycles)


def _signed_area(vertices: np.ndarray) -> float:
    following = np.roll(vertices, -1, axis=0)
    return float(0.5 * np.sum(vertices[:, 0] * following[:, 1] -
                              following[:, 0] * vertices[:, 1]))


def _orientation(a: np.ndarray,
                 b: np.ndarray,
                 c: np.ndarray,
                 eps: float = 1e-8) -> int:
    value = float(np.cross(b - a, c - a))
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a: np.ndarray,
                b: np.ndarray,
                point: np.ndarray,
                eps: float = 1e-8) -> bool:
    return (min(a[0], b[0]) - eps <= point[0] <=
            max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= point[1] <=
            max(a[1], b[1]) + eps)


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                        d: np.ndarray) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(a, b, c)) or
            (o2 == 0 and _on_segment(a, b, d)) or
            (o3 == 0 and _on_segment(c, d, a)) or
            (o4 == 0 and _on_segment(c, d, b)))


def _can_remove_vertex(vertices: np.ndarray, index: int,
                       winding: float) -> bool:
    count = len(vertices)
    previous = (index - 1) % count
    following = (index + 1) % count
    replacement_a = vertices[previous]
    replacement_b = vertices[following]
    for edge in range(count):
        edge_next = (edge + 1) % count
        if edge in (previous, index, following):
            continue
        if edge_next in (previous, index, following):
            continue
        if _segments_intersect(replacement_a, replacement_b, vertices[edge],
                               vertices[edge_next]):
            return False
    reduced = np.delete(vertices, index, axis=0)
    area = _signed_area(reduced)
    return abs(area) > 1e-8 and math.copysign(1.0, area) == winding


def simplify_closed_vertices(vertices: np.ndarray,
                             max_vertices: int = 40) -> np.ndarray:
    """Topology-preserving closed-curve Visvalingam simplification."""
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 2).copy()
    if len(vertices) > 1 and np.linalg.norm(vertices[0] - vertices[-1]) < 1e-6:
        vertices = vertices[:-1]
    if len(vertices) > 0:
        previous = np.roll(vertices, 1, axis=0)
        vertices = vertices[np.linalg.norm(vertices - previous, axis=1) >=
                            1e-4]
    if len(vertices) < 3:
        raise ValueError('a polygon needs at least three unique vertices')
    area = _signed_area(vertices)
    if abs(area) <= 1e-8:
        raise ValueError('cannot simplify a zero-area polygon')
    winding = math.copysign(1.0, area)
    while len(vertices) > max_vertices:
        previous = np.roll(vertices, 1, axis=0)
        following = np.roll(vertices, -1, axis=0)
        areas = np.abs(
            np.cross(vertices - previous, following - vertices)) * 0.5
        removed = False
        for index in np.argsort(areas):
            if _can_remove_vertex(vertices, int(index), winding):
                vertices = np.delete(vertices, int(index), axis=0)
                removed = True
                break
        if not removed:
            raise ValueError('simplification would change polygon topology')
    return vertices


def _vertices_to_primitives(vertices: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.roll(vertices, 1, dims=0), vertices,
                      torch.roll(vertices, -1, dims=0)), dim=-1)


def _clip_segment(start: np.ndarray,
                  end: np.ndarray,
                  low: float,
                  high: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Liang-Barsky clip of one original polygon edge."""
    delta = end - start
    p = (-delta[0], delta[0], -delta[1], delta[1])
    q = (start[0] - low, high - start[0], start[1] - low,
         high - start[1])
    enter, leave = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(float(pi)) < 1e-12:
            if qi < 0:
                return None
            continue
        ratio = float(qi / pi)
        if pi < 0:
            enter = max(enter, ratio)
        else:
            leave = min(leave, ratio)
        if enter > leave:
            return None
    return start + enter * delta, start + leave * delta


def _draw_8_connected_line(canvas: np.ndarray, start: np.ndarray,
                           end: np.ndarray) -> None:
    """Integer Bresenham rasterization with diagonal (8-connected) steps."""
    x0, y0 = np.rint(start).astype(np.int64)
    x1, y1 = np.rint(end).astype(np.int64)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx - dy
    while True:
        canvas[y0, x0] = 1
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled > -dy:
            error -= dy
            x0 += step_x
        if doubled < dx:
            error += dx
            y0 += step_y


def rasterize_contour_target(primitives: torch.Tensor,
                             crop_size: int = 32) -> torch.Tensor:
    """Rasterize only clipped original boundary edges, never crop closure."""
    canvas = np.zeros((crop_size, crop_size), dtype=np.int64)
    if primitives.numel() == 0:
        return torch.from_numpy(canvas).to(device=primitives.device)
    centers = primitives.detach().float().cpu().numpy()[:, 2:4]
    pixel_centers = centers * float(crop_size) - 0.5
    for index in range(len(pixel_centers)):
        next_index = (index + 1) % len(pixel_centers)
        clipped = _clip_segment(pixel_centers[index],
                                pixel_centers[next_index], 0.0,
                                float(crop_size - 1))
        if clipped is not None:
            _draw_8_connected_line(canvas, clipped[0], clipped[1])
    return torch.from_numpy(canvas).to(device=primitives.device)


def _pairwise_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:4], boxes2[None, :, 2:4])
    intersection_wh = (right_bottom - left_top).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    return intersection / (area1[:, None] + area2[None, :] - intersection
                           ).clamp(min=1e-6)


@HEADS.register_module(force=True)
class P2PFormerV3Head(BaseModule):
    """Contour-conditioned support-box diffusion head."""

    def __init__(self,
                 in_channels: int = 256,
                 num_slots: int = 40,
                 hidden_dim: int = 256,
                 expand_scale: float = 1.1,
                 state_mean: Sequence[float] = (0.5, 0.5, -1.25, -1.25),
                 state_std: Sequence[float] = (0.3, 0.3, 0.24, 0.24),
                 conditioner: Optional[Dict] = None,
                 diffusion_steps: int = 1000,
                 inference_nfe: int = 2,
                 inference_seed: int = 3407,
                 validity_threshold: float = 0.5,
                 proposal_probability: float = 0.1,
                 proposal_warmup_steps: int = 1000,
                 proposal_iou_threshold: float = 0.5,
                 loss_weights: Optional[Dict[str, float]] = None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs) -> None:
        super(P2PFormerV3Head, self).__init__(init_cfg)
        if num_slots != 40:
            raise ValueError(
                'P2PFormerV3 V1 fixes the support set to 40 slots')
        if hidden_dim != in_channels:
            raise ValueError('V1 requires hidden_dim == in_channels')
        if inference_nfe not in (1, 2, 4):
            raise ValueError('inference_nfe must be one of 1, 2, or 4')
        self.in_channels = int(in_channels)
        self.num_slots = int(num_slots)
        self.hidden_dim = int(hidden_dim)
        self.expand_scale = float(expand_scale)
        self.inference_nfe = int(inference_nfe)
        self.inference_seed = int(inference_seed)
        self.validity_threshold = float(validity_threshold)
        self.proposal_probability = float(proposal_probability)
        self.proposal_warmup_steps = int(proposal_warmup_steps)
        self.proposal_iou_threshold = float(proposal_iou_threshold)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        conditioner_cfg = dict(conditioner or {})
        conditioner_cfg.setdefault('in_channels', in_channels)
        self.contour_conditioner = ContourEvidenceConditioner(
            **conditioner_cfg)
        self.support_codec = SupportBoxCodec(state_mean, state_std)
        self.register_buffer('support_anchors', make_anchor_lattice())
        self.target_builder = SupportTargetBuilder(self.support_codec)
        self.diffusion_schedule = CosineDiffusionSchedule(diffusion_steps)
        self.support_denoiser = CoarseToFineSetDenoiser(
            num_slots=num_slots,
            hidden_dim=hidden_dim,
            num_timesteps=diffusion_steps)
        self.primitive_recovery = FinalPrimitiveHead(hidden_dim=hidden_dim)
        self.successor_head = SuccessorTopologyHead(hidden_dim=hidden_dim)
        self.geometry_criterion = P2PFormerV3Loss()
        default_weights = dict(
            contour=1.0,
            x0=1.0,
            support_box=2.0,
            validity=1.0,
            primitive=5.0,
            topology=1.0)
        default_weights.update(loss_weights or {})
        self.loss_weights = default_weights
        self.register_buffer(
            '_training_steps', torch.zeros((), dtype=torch.long))
        self.last_inference_diagnostics: List[Dict] = []

    def requires_detector_proposals(self) -> bool:
        return (self.training and self.proposal_probability > 0 and
                int(self._training_steps.item()) >=
                self.proposal_warmup_steps)

    def _validate_features(
            self, features: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        if torch.is_tensor(features) or len(features) < 3:
            raise ValueError(
                'V3 requires line_fpn=True and P2/P3/P4 feature tensors')
        selected = tuple(features[:3])
        for level, feature in enumerate(selected):
            if feature.dim() != 4 or feature.size(1) != self.in_channels:
                raise ValueError('invalid contour feature at level {}'.format(
                    level))
        for finer, coarser in zip(selected, selected[1:]):
            if (finer.shape[-2] < coarser.shape[-2] or
                    finer.shape[-1] < coarser.shape[-1]):
                raise ValueError('features must be ordered P2, P3, P4')
        return selected

    def _flatten_training_instances(self, gt_bboxes, gt_lines):
        records = []
        simplified_count = 0
        skipped_count = 0
        for image_index, (boxes, components) in enumerate(
                zip(gt_bboxes, gt_lines)):
            if len(boxes) != len(components):
                raise ValueError(
                    'each building box must match one gt_lines component')
            for local_index, (box, component) in enumerate(
                    zip(boxes, components)):
                try:
                    component_np = np.asarray(
                        component, dtype=np.float32).reshape(-1, 6)
                    if (len(component_np) < 3 or
                            not np.isfinite(component_np).all()):
                        raise ValueError('invalid primitive component')
                    vertices = component_np[:, 2:4]
                    was_simplified = len(vertices) > self.num_slots
                    vertices = simplify_closed_vertices(
                        vertices, max_vertices=self.num_slots)
                except (TypeError, ValueError):
                    skipped_count += 1
                    continue
                if was_simplified:
                    simplified_count += 1
                vertices_tensor = torch.as_tensor(
                    vertices, device=box.device, dtype=box.dtype)
                primitive = _vertices_to_primitives(vertices_tensor)
                records.append((image_index, local_index, box, primitive))
        return records, simplified_count, skipped_count

    def _match_detector_proposals(self, gt_bboxes, detector_proposals):
        matches = []
        for image_index, boxes in enumerate(gt_bboxes):
            if detector_proposals is None:
                matches.append([None] * len(boxes))
                continue
            proposals = detector_proposals[image_index]
            if proposals.numel() == 0 or boxes.numel() == 0:
                matches.append([None] * len(boxes))
                continue
            proposals = proposals[..., :4].to(
                device=boxes.device, dtype=boxes.dtype).detach()
            overlaps = _pairwise_iou(boxes[..., :4], proposals)
            best_iou, best_index = overlaps.max(dim=1)
            image_matches = []
            for local_index in range(len(boxes)):
                if float(best_iou[local_index]) >= self.proposal_iou_threshold:
                    image_matches.append(proposals[best_index[local_index]])
                else:
                    image_matches.append(None)
            matches.append(image_matches)
        return matches

    def _outer_roi(self,
                   box: torch.Tensor,
                   center_fraction: float = 0.0,
                   log_fraction: float = 0.0) -> torch.Tensor:
        size = (box[2:4] - box[0:2]).clamp(min=1.0)
        center = 0.5 * (box[0:2] + box[2:4])
        if center_fraction > 0:
            center_noise = 2.0 * torch.rand_like(center) - 1.0
            center = center + center_fraction * size * center_noise
        if log_fraction > 0:
            scale_noise = 2.0 * torch.rand_like(size) - 1.0
            size = size * torch.exp(log_fraction * scale_noise)
        size = size * self.expand_scale
        return torch.cat((center - 0.5 * size, center + 0.5 * size))

    @staticmethod
    def _normalize_primitive(primitive: torch.Tensor,
                             roi: torch.Tensor) -> torch.Tensor:
        size = (roi[2:4] - roi[0:2]).clamp(min=1e-6)
        points = primitive.reshape(-1, 3, 2)
        points = (points - roi[0:2]) / size
        return points.reshape(-1, 6)

    def _select_training_rois(self, records, gt_bboxes,
                              detector_proposals):
        proposal_matches = self._match_detector_proposals(
            gt_bboxes, detector_proposals)
        rois = []
        normalized_primitives = []
        rejection_count = 0
        proposal_fallback_count = 0
        mode_counts = [0, 0, 0, 0]
        hard_probability = max(0.0, 0.20 - self.proposal_probability)
        mode_thresholds = (
            0.45,
            0.80,
            0.80 + hard_probability,
            0.80 + hard_probability + self.proposal_probability)

        for image_index, local_index, gt_box, primitive in records:
            draw = float(torch.rand(()))
            if draw < mode_thresholds[0]:
                mode = 0  # clean
            elif draw < mode_thresholds[1]:
                mode = 1  # moderate synthetic jitter
            elif draw < mode_thresholds[2]:
                mode = 2  # hard synthetic jitter
            else:
                mode = 3  # detached FCOS proposal
            mode_counts[mode] += 1

            matched_proposal = proposal_matches[image_index][local_index]
            if mode == 3 and matched_proposal is None:
                mode = 2
                proposal_fallback_count += 1

            accepted_roi = None
            accepted_primitive = None
            for _ in range(10):
                if mode == 0:
                    candidate = self._outer_roi(gt_box)
                elif mode == 1:
                    candidate = self._outer_roi(gt_box, 0.10, 0.15)
                elif mode == 2:
                    candidate = self._outer_roi(gt_box, 0.20, 0.20)
                else:
                    candidate = self._outer_roi(matched_proposal)
                normalized = self._normalize_primitive(primitive, candidate)
                if bool(((normalized >= -0.25) &
                         (normalized <= 1.25)).all()):
                    accepted_roi = candidate
                    accepted_primitive = normalized
                    break
                rejection_count += 1
                if mode in (0, 3):
                    break

            if accepted_roi is None:
                # The expanded clean GT crop is guaranteed to cover every GT
                # vertex and is the safe fallback for an unlucky perturbation.
                accepted_roi = self._outer_roi(gt_box)
                accepted_primitive = self._normalize_primitive(
                    primitive, accepted_roi)
                if not bool(((accepted_primitive >= -0.25) &
                             (accepted_primitive <= 1.25)).all()):
                    continue
            roi = torch.cat((accepted_roi.new_tensor([image_index]),
                             accepted_roi))
            rois.append(roi)
            normalized_primitives.append(accepted_primitive)

        if not rois:
            reference = gt_bboxes[0]
            empty_rois = reference.new_empty((0, 5))
        else:
            empty_rois = torch.stack(rois, dim=0)
        diagnostics = dict(
            rejection_count=rejection_count,
            proposal_fallback_count=proposal_fallback_count,
            mode_counts=mode_counts)
        return empty_rois, normalized_primitives, diagnostics

    def _permute_targets(self, targets):
        batch_size, num_slots = targets['z0'].shape[:2]
        device = targets['z0'].device
        anchors = self.support_anchors.to(device=device,
                                          dtype=targets['z0'].dtype)
        anchor_batches = []
        slot_keys = ('z0', 'active_mask', 'primitive_targets',
                     'target_boxes', 'geometry_weights')
        output = {key: [] for key in slot_keys}
        output.update(successors_cw=[], successors_ccw=[])
        for batch_index in range(batch_size):
            permutation = torch.randperm(num_slots, device=device)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(num_slots, device=device)
            anchor_batches.append(anchors[permutation])
            for key in slot_keys:
                output[key].append(targets[key][batch_index, permutation])
            for key in ('successors_cw', 'successors_ccw'):
                successor = targets[key][batch_index, permutation].clone()
                valid = successor >= 0
                successor[valid] = inverse[successor[valid]]
                output[key].append(successor)
        output = {
            key: torch.stack(value, dim=0)
            for key, value in output.items()
        }
        return output, torch.stack(anchor_batches, dim=0)

    def _zero_losses(self, features, diagnostics=None):
        zero = sum(feature.sum() for feature in features) * 0.0
        for parameter in self.parameters():
            if parameter.numel():
                zero = zero + parameter.reshape(-1)[0] * 0.0
        losses = dict(
            loss_v3_contour=zero,
            loss_v3_x0=zero,
            loss_v3_sbox=zero,
            loss_v3_valid=zero,
            loss_v3_primitive=zero,
            loss_v3_topology=zero)
        if diagnostics:
            losses.update({
                key: zero.detach().new_tensor(float(value))
                for key, value in diagnostics.items()
            })
        return losses

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_lines,
                      matched_idxs=None,
                      detector_proposals=None,
                      **kwargs):
        features = self._validate_features(x)
        with torch.no_grad():
            self._training_steps.add_(1)
        records, simplified_count, skipped_count = \
            self._flatten_training_instances(gt_bboxes, gt_lines)
        if not records:
            return self._zero_losses(
                features,
                dict(v3_simplified_count=simplified_count,
                     v3_skipped_count=skipped_count))

        rois, normalized_primitives, exposure = self._select_training_rois(
            records, gt_bboxes, detector_proposals)
        if not normalized_primitives:
            return self._zero_losses(
                features,
                dict(v3_simplified_count=simplified_count,
                     v3_skipped_count=skipped_count + len(records)))

        condition = self.contour_conditioner(features, rois)
        contour_targets = torch.stack([
            rasterize_contour_target(primitive)
            for primitive in normalized_primitives
        ], dim=0).long()
        loss_contour = F.cross_entropy(
            condition['contour_logits'].float(),
            contour_targets,
            reduction='mean')

        targets = self.target_builder.build(
            normalized_primitives, anchors=self.support_anchors)
        targets, anchors = self._permute_targets(targets)
        batch_size = targets['z0'].size(0)
        timesteps = torch.randint(
            1,
            self.diffusion_schedule.num_timesteps,
            (batch_size, ),
            dtype=torch.long,
            device=targets['z0'].device)
        noisy_state = self.diffusion_schedule.q_sample(
            targets['z0'], timesteps, torch.randn_like(targets['z0']))
        pred_state, hidden = self.support_denoiser(
            noisy_state, timesteps, anchors, condition['memories'],
            condition['position_embeddings'], condition['image_context'])
        pred_boxes = self.support_codec.decode(pred_state)
        primitives, validity_logits, fused = self.primitive_recovery(
            hidden, pred_boxes, condition['dense_feature'])
        successor_logits = self.successor_head(fused, primitives)
        raw_losses = self.geometry_criterion(
            pred_state, targets['z0'], pred_boxes, targets['target_boxes'],
            validity_logits, targets['active_mask'].float(), primitives,
            targets['primitive_targets'], successor_logits,
            targets['successors_cw'], targets['successors_ccw'])

        # Do not return criterion.loss_total: MMDetection sums every key that
        # contains "loss", which would otherwise double-count all components.
        losses = dict(
            loss_v3_contour=loss_contour * self.loss_weights['contour'],
            loss_v3_x0=raw_losses['loss_x0'] * self.loss_weights['x0'],
            loss_v3_sbox=(raw_losses['loss_sbox'] *
                          self.loss_weights['support_box']),
            loss_v3_valid=(raw_losses['loss_valid'] *
                           self.loss_weights['validity']),
            loss_v3_primitive=(raw_losses['loss_primitive'] *
                               self.loss_weights['primitive']),
            loss_v3_topology=(raw_losses['loss_topology'] *
                              self.loss_weights['topology']))
        reference = loss_contour.detach()
        total_instances = max(len(normalized_primitives), 1)
        losses.update(
            v3_simplified_count=reference.new_tensor(float(simplified_count)),
            v3_skipped_count=reference.new_tensor(float(skipped_count)),
            v3_jitter_rejection_rate=reference.new_tensor(
                exposure['rejection_count'] / float(total_instances)),
            v3_proposal_fallback_rate=reference.new_tensor(
                exposure['proposal_fallback_count'] / float(total_instances)),
            v3_active_supports=targets['active_mask'].float().sum(
                dim=1).mean().detach(),
            v3_local_grid_oob_rate=(
                self.primitive_recovery.last_oob_rate.mean().detach()))
        return losses

    @staticmethod
    def _map_boxes_to_augmented(boxes, img_metas, boxes_are_rescaled):
        mapped = []
        for per_image, meta in zip(boxes, img_metas):
            coordinates = per_image[..., :4].clone()
            if boxes_are_rescaled and coordinates.numel():
                scale_factor = coordinates.new_tensor(
                    meta.get('scale_factor', [1.0, 1.0, 1.0, 1.0]))
                if scale_factor.numel() == 1:
                    scale_factor = scale_factor.repeat(4)
                coordinates = coordinates * scale_factor[:4]
                # FCOS rescale divides only by scale_factor. It does not undo
                # a test-time flip, so this RoI remains in the flipped frame
                # used by the feature maps. The detector unflips the contour.
            mapped.append(coordinates)
        return mapped

    def simple_test(self,
                    x,
                    img_metas,
                    bboxes,
                    boxes_are_rescaled=True,
                    **kwargs):
        features = self._validate_features(x)
        boxes_augmented = self._map_boxes_to_augmented(
            bboxes, img_metas, boxes_are_rescaled)
        owners = []
        flat_boxes = []
        for image_index, per_image in enumerate(boxes_augmented):
            if len(per_image):
                flat_boxes.append(per_image)
                owners.append(
                    per_image.new_full((len(per_image), ),
                                       image_index,
                                       dtype=torch.long))
        if not flat_boxes:
            reference = features[0]
            return (reference.new_empty((0, self.num_slots, 6)),
                    reference.new_empty((0, self.num_slots)),
                    torch.empty(
                        (0, self.num_slots),
                        dtype=torch.long,
                        device=reference.device))
        flat_boxes = torch.cat(flat_boxes, dim=0)
        owners = torch.cat(owners, dim=0)
        center = 0.5 * (flat_boxes[:, :2] + flat_boxes[:, 2:4])
        size = (flat_boxes[:, 2:4] - flat_boxes[:, :2]).clamp(
            min=1.0) * self.expand_scale
        outer = torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1)
        rois = torch.cat((owners[:, None].to(dtype=outer.dtype), outer),
                         dim=-1)
        condition = self.contour_conditioner(features, rois)

        generator_device = outer.device if outer.is_cuda else 'cpu'
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.inference_seed)
        initial_noise = torch.randn(
            (len(outer), self.num_slots, 4),
            dtype=torch.float32,
            device=outer.device,
            generator=generator)
        clean_state, hidden = ddim_sample(
            self.support_denoiser,
            self.diffusion_schedule,
            initial_noise,
            self.support_anchors,
            condition['memories'],
            condition['position_embeddings'],
            condition['image_context'],
            num_steps=self.inference_nfe)
        support_boxes = self.support_codec.decode(clean_state)
        primitives, validity_logits, fused = self.primitive_recovery(
            hidden, support_boxes, condition['dense_feature'])
        successor_logits = self.successor_head(fused, primitives)
        cycles = solve_cycles(
            successor_logits,
            primitives,
            validity_logits,
            validity_threshold=self.validity_threshold)

        points = primitives.reshape(len(outer), self.num_slots, 3, 2)
        outer_size = outer[:, 2:4] - outer[:, 0:2]
        points = (outer[:, None, None, 0:2] +
                  outer_size[:, None, None, :] * points)
        lines = points.reshape(len(outer), self.num_slots, 6)
        scores = lines.new_zeros((len(outer), self.num_slots))
        order = torch.zeros(
            (len(outer), self.num_slots),
            dtype=torch.long,
            device=lines.device)
        diagnostics = []
        for batch_index, cycle in enumerate(cycles):
            diagnostic = dict(cycle)
            diagnostic['q_topology'] = 0.0
            diagnostic['local_grid_oob_rate'] = float(
                self.primitive_recovery.last_oob_rate[batch_index].cpu())
            if cycle['valid']:
                indices = cycle['indices']
                scores[batch_index, indices] = 1.0
                order[batch_index, indices] = torch.arange(
                    len(indices), device=order.device)
                active = torch.zeros(
                    (1, self.num_slots), dtype=torch.bool, device=lines.device)
                active[0, indices] = True
                log_probability = masked_log_sinkhorn(
                    successor_logits[batch_index:batch_index + 1], active)
                following = torch.roll(indices, -1)
                mean_log_probability = log_probability[
                    0, indices, following].mean()
                diagnostic['q_topology'] = float(
                    mean_log_probability.exp().cpu())
            diagnostics.append(diagnostic)
        self.last_inference_diagnostics = diagnostics
        return lines, scores, order


__all__ = [
    'P2PFormerV3Head', 'rasterize_contour_target',
    'simplify_closed_vertices'
]
