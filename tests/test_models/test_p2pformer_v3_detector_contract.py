import numpy as np
import torch
import torch.nn as nn

# Importing the registry first mirrors normal MMDetection startup and avoids
# the legacy repository's detector-registration import cycle.
import mmdet.models  # noqa: F401
from p2pformer.models.p2pformer import P2PFormerSegmentor


class _OneClassHead(nn.Module):
    num_classes = 1


def _bare_detector():
    detector = P2PFormerSegmentor.__new__(P2PFormerSegmentor)
    nn.Module.__init__(detector)
    detector.bbox_head = _OneClassHead()
    return detector


def test_contour_projection_uses_independent_xy_scale_factors():
    detector = _bare_detector()
    contour = np.array(
        [[20.0, 40.0], [60.0, 40.0], [60.0, 120.0], [20.0, 120.0]],
        dtype=np.float32)
    labels = torch.zeros(1, dtype=torch.long)
    bboxes = torch.tensor([[20.0, 40.0, 60.0, 120.0, 0.9]])
    meta = dict(
        img_shape=(200, 100, 3),
        ori_shape=(50, 50, 3),
        scale_factor=np.array([2.0, 4.0, 2.0, 4.0], dtype=np.float32),
        flip=False)

    masks, kept_boxes, kept_labels = detector.single_convert_contour2mask(
        [contour], labels, bboxes, meta)

    assert kept_boxes.shape == (1, 5)
    assert kept_labels.tolist() == [0]
    assert len(masks[0]) == 1
    mask = masks[0][0]
    assert mask[15, 20]
    assert not mask[5, 5]


def test_invalid_polygon_filters_bbox_and_label_in_lockstep():
    detector = _bare_detector()
    valid = np.array(
        [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]],
        dtype=np.float32)
    invalid = np.empty((0, 2), dtype=np.float32)
    labels = torch.zeros(2, dtype=torch.long)
    bboxes = torch.tensor([
        [10.0, 10.0, 30.0, 30.0, 0.9],
        [40.0, 40.0, 50.0, 50.0, 0.8],
    ])
    meta = dict(
        img_shape=(64, 64, 3),
        ori_shape=(64, 64, 3),
        scale_factor=np.ones(4, dtype=np.float32),
        flip=False)

    masks, kept_boxes, kept_labels = detector.single_convert_contour2mask(
        [valid, invalid], labels, bboxes, meta)

    assert len(masks[0]) == 1
    assert torch.equal(kept_boxes, bboxes[:1])
    assert torch.equal(kept_labels, labels[:1])


def test_all_flip_modes_are_undone_before_rasterization():
    detector = _bare_detector()
    labels = torch.zeros(1, dtype=torch.long)
    bboxes = torch.tensor([[80.0, 10.0, 90.0, 20.0, 0.9]])
    contours = {
        'horizontal': [[90.0, 10.0], [80.0, 10.0], [80.0, 20.0],
                       [90.0, 20.0]],
        'vertical': [[10.0, 40.0], [20.0, 40.0], [20.0, 30.0],
                     [10.0, 30.0]],
        'diagonal': [[90.0, 40.0], [80.0, 40.0], [80.0, 30.0],
                     [90.0, 30.0]],
    }
    for direction, contour in contours.items():
        meta = dict(
            img_shape=(50, 100, 3),
            ori_shape=(50, 100, 3),
            scale_factor=np.ones(4, dtype=np.float32),
            flip=True,
            flip_direction=direction)
        contour = np.asarray(contour, dtype=np.float32)
        masks, _, _ = detector.single_convert_contour2mask(
            [contour], labels, bboxes, meta)
        assert masks[0][0][15, 15]
