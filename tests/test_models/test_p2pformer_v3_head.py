import math

import numpy as np
import torch

import mmdet.models  # noqa: F401
from mmdet.models.detectors.base import BaseDetector
from p2pformer.models.p2pformer_v3_head import (
    P2PFormerV3Head, rasterize_contour_target, simplify_closed_vertices)


def _primitives(vertices):
    vertices = np.asarray(vertices, dtype=np.float32)
    return np.concatenate(
        (np.roll(vertices, 1, axis=0), vertices,
         np.roll(vertices, -1, axis=0)), axis=-1)


def _tiny_head(inference_nfe=1):
    return P2PFormerV3Head(
        in_channels=32,
        hidden_dim=32,
        state_mean=[0.5, 0.5, -1.25, -1.25],
        state_std=[0.3, 0.3, 0.24, 0.24],
        conditioner=dict(
            crop_size=32,
            sample_side=8,
            num_heads=8,
            feature_strides=(4, 8, 16),
            roi_chunk_size=4),
        inference_nfe=inference_nfe,
        proposal_probability=0.0)


def _features(requires_grad=False):
    return [
        torch.randn(1, 32, 16, 16, requires_grad=requires_grad),
        torch.randn(1, 32, 8, 8, requires_grad=requires_grad),
        torch.randn(1, 32, 4, 4, requires_grad=requires_grad),
        torch.randn(1, 32, 2, 2, requires_grad=requires_grad),
    ]


def test_topology_preserving_simplification_caps_complex_polygon():
    angles = np.linspace(0.0, 2.0 * math.pi, 80, endpoint=False)
    radii = 1.0 + 0.05 * np.sin(7.0 * angles)
    vertices = np.stack((radii * np.cos(angles),
                         radii * np.sin(angles)), axis=-1).astype(np.float32)
    simplified = simplify_closed_vertices(vertices, max_vertices=40)

    def signed_area(points):
        shifted = np.roll(points, -1, axis=0)
        return 0.5 * np.sum(points[:, 0] * shifted[:, 1] -
                            shifted[:, 0] * points[:, 1])

    assert simplified.shape == (40, 2)
    assert np.sign(signed_area(simplified)) == np.sign(signed_area(vertices))
    assert len(np.unique(simplified, axis=0)) == 40


def test_cropped_contour_target_does_not_close_along_crop_border():
    vertices = torch.tensor([
        [-0.2, 0.2],
        [0.5, 0.2],
        [0.5, 0.8],
        [-0.2, 0.8],
    ])
    primitives = torch.cat((torch.roll(vertices, 1, 0), vertices,
                            torch.roll(vertices, -1, 0)), dim=-1)
    target = rasterize_contour_target(primitives)
    assert target.shape == (32, 32)
    assert target.dtype == torch.long
    assert target[6, 0] == 1
    assert target[25, 0] == 1
    assert target[16, 0] == 0


def test_head_forward_train_has_independent_weighted_losses_and_gradients():
    torch.manual_seed(4)
    head = _tiny_head()
    head.train()
    features = _features(requires_grad=True)
    vertices = np.array(
        [[8.0, 8.0], [56.0, 8.0], [56.0, 56.0], [8.0, 56.0]],
        dtype=np.float32)
    losses = head.forward_train(
        features,
        [dict(img_shape=(64, 64, 3))],
        [torch.tensor([[8.0, 8.0, 56.0, 56.0]])],
        [[_primitives(vertices)]])
    expected = {
        'loss_v3_contour', 'loss_v3_x0', 'loss_v3_sbox',
        'loss_v3_valid', 'loss_v3_primitive', 'loss_v3_topology'
    }
    assert {key for key in losses if key.startswith('loss_')} == expected
    assert 'loss_total' not in losses
    assert all(torch.isfinite(value) for value in losses.values())
    total = sum(losses[key] for key in expected)
    total.backward()
    assert features[0].grad is not None
    assert head.contour_conditioner.offset_net[-1].weight.grad is not None
    assert head.support_denoiser.output_head.weight.grad is not None
    assert head.primitive_recovery.primitive_head[-1].weight.grad is not None
    assert head.successor_head.query.weight.grad is not None


def test_head_all_empty_gt_returns_graph_connected_finite_losses():
    head = _tiny_head()
    features = _features(requires_grad=True)
    losses = head.forward_train(
        features,
        [dict(img_shape=(64, 64, 3))],
        [torch.empty(0, 4)],
        [[]])
    loss_values = [value for key, value in losses.items()
                   if key.startswith('loss_')]
    assert len(loss_values) == 6
    assert all(value.item() == 0.0 for value in loss_values)
    sum(loss_values).backward()
    assert features[0].grad is not None


def test_simple_test_is_seeded_and_preserves_fixed_instance_shapes():
    torch.manual_seed(9)
    head = _tiny_head(inference_nfe=1).eval()
    features = _features()
    boxes = [torch.tensor([[4.0, 8.0, 28.0, 24.0, 0.9]])]
    meta = [
        dict(
            img_shape=(64, 64, 3),
            ori_shape=(32, 16, 3),
            scale_factor=np.array([4.0, 2.0, 4.0, 2.0], dtype=np.float32),
            flip=False)
    ]
    with torch.no_grad():
        first = head.simple_test(
            features, meta, boxes, boxes_are_rescaled=True)
        second = head.simple_test(
            features, meta, boxes, boxes_are_rescaled=True)
    assert first[0].shape == (1, 40, 6)
    assert first[1].shape == (1, 40)
    assert first[2].shape == (1, 40)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert set(first[1].unique().tolist()).issubset({0.0, 1.0})


def test_joint_slot_permutation_remaps_anchors_and_successors():
    head = _tiny_head()
    vertices = torch.tensor([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8],
                             [0.2, 0.8]])
    primitives = torch.cat((torch.roll(vertices, 1, 0), vertices,
                            torch.roll(vertices, -1, 0)), dim=-1)
    targets = head.target_builder.build(
        [primitives], anchors=head.support_anchors)
    torch.manual_seed(17)
    permuted, anchors = head._permute_targets(targets)
    active = permuted['active_mask'][0]
    successors = permuted['successors_cw'][0, active]
    active_indices = torch.nonzero(active, as_tuple=False).flatten()
    assert bool(active[successors].all())
    assert torch.unique(successors).numel() == 4
    for slot in active_indices:
        successor = permuted['successors_cw'][0, slot]
        assert torch.allclose(permuted['primitive_targets'][0, slot, 4:6],
                              permuted['primitive_targets'][0, successor, 2:4])
    distances = torch.cdist(anchors[0], head.support_anchors)
    assert torch.allclose(distances.min(dim=1)[0], torch.zeros(40))


def test_proposal_warmup_and_detached_matched_roi(monkeypatch):
    head = _tiny_head()
    head.proposal_probability = 0.1
    head.proposal_warmup_steps = 2
    head.train()
    assert not head.requires_detector_proposals()
    head._training_steps.fill_(1)
    assert not head.requires_detector_proposals()
    head._training_steps.fill_(2)
    assert head.requires_detector_proposals()
    head.eval()
    assert not head.requires_detector_proposals()
    head.train()

    monkeypatch.setattr(
        torch, 'rand', lambda *args, **kwargs: torch.tensor(0.95))
    box = torch.tensor([0.0, 0.0, 10.0, 10.0])
    vertices = torch.tensor([[1.0, 1.0], [9.0, 1.0], [9.0, 9.0],
                             [1.0, 9.0]])
    primitive = torch.cat((torch.roll(vertices, 1, 0), vertices,
                           torch.roll(vertices, -1, 0)), dim=-1)
    proposal = torch.tensor(
        [[1.0, 1.0, 9.0, 9.0]], requires_grad=True)
    rois, _, diagnostics = head._select_training_rois(
        [(0, 0, box, primitive)], [box.unsqueeze(0)], [proposal])
    assert torch.allclose(rois[0, 1:],
                          torch.tensor([0.6, 0.6, 9.4, 9.4]),
                          atol=1e-6)
    assert not rois.requires_grad
    assert diagnostics['proposal_fallback_count'] == 0

    _, _, diagnostics = head._select_training_rois(
        [(0, 0, box, primitive)], [box.unsqueeze(0)], None)
    assert diagnostics['proposal_fallback_count'] == 1


def test_loss_parser_does_not_double_count_v3_components():
    losses = {
        'loss_v3_contour': torch.tensor(1.0),
        'loss_v3_x0': torch.tensor(2.0),
        'loss_v3_sbox': torch.tensor(3.0),
        'loss_v3_valid': torch.tensor(4.0),
        'loss_v3_primitive': torch.tensor(5.0),
        'loss_v3_topology': torch.tensor(6.0),
        'v3_active_supports': torch.tensor(7.0),
    }
    parsed, logs = BaseDetector._parse_losses(None, losses)
    assert parsed.item() == 21.0
    assert logs['loss'] == 21.0


def test_box_mapping_keeps_fcos_flip_frame_and_unscaled_identity():
    box = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9]])
    base_meta = dict(
        img_shape=(20, 20, 3),
        scale_factor=np.array([2.0, 3.0, 2.0, 3.0], dtype=np.float32),
        flip=True)
    expected = torch.tensor([[2.0, 6.0, 6.0, 12.0]])
    for direction in ('horizontal', 'vertical', 'diagonal'):
        meta = dict(base_meta, flip_direction=direction)
        mapped = P2PFormerV3Head._map_boxes_to_augmented(
            [box], [meta], True)[0]
        assert torch.equal(mapped, expected)
    identity = P2PFormerV3Head._map_boxes_to_augmented(
        [box], [base_meta], False)[0]
    assert torch.equal(identity, box[:, :4])
