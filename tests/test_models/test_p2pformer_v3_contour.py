import pytest
import torch

from p2pformer.models.p2pformer_v3_contour import (
    ContourEvidenceConditioner)


def _features(requires_grad=False):
    return [
        torch.randn(1, 256, 32, 32, requires_grad=requires_grad),
        torch.randn(1, 256, 16, 16, requires_grad=requires_grad),
        torch.randn(1, 256, 8, 8, requires_grad=requires_grad),
    ]


@pytest.mark.parametrize('variant', ['full', 'sfe', 'dfe'])
@pytest.mark.parametrize('mask_mode', ['soft', 'hard', 'none'])
def test_contour_conditioner_shapes(variant, mask_mode):
    module = ContourEvidenceConditioner(
        variant=variant, mask_mode=mask_mode)
    rois = torch.tensor([[0., 8., 8., 120., 120.]])
    output = module(_features(), rois)

    assert [item.shape for item in output['memories']] == [
        (1, 64, 256), (1, 64, 256), (1, 64, 256)
    ]
    assert [item.shape for item in output['position_embeddings']] == [
        (1, 64, 256), (1, 64, 256), (1, 64, 256)
    ]
    assert output['dense_feature'].shape == (1, 256, 32, 32)
    assert output['image_context'].shape == (1, 256)
    assert output['contour_logits'].shape == (1, 2, 32, 32)
    assert output['feature_mask'].shape == (1, 1, 32, 32)
    assert torch.isfinite(output['dense_feature']).all()


def test_contour_conditioner_gradient_reaches_offsets_and_input():
    module = ContourEvidenceConditioner()
    features = _features(requires_grad=True)
    rois = torch.tensor([[0., 8., 8., 120., 120.]])
    output = module(features, rois)
    loss = output['dense_feature'].square().mean()
    loss = loss + sum(item.square().mean()
                      for item in output['memories'])
    loss.backward()

    offset_grad = module.offset_net[-1].weight.grad
    assert offset_grad is not None
    assert offset_grad.abs().sum() > 0
    assert features[0].grad is not None
    assert features[0].grad.abs().sum() > 0


def test_contour_conditioner_empty_rois():
    module = ContourEvidenceConditioner()
    output = module(_features(), torch.empty(0, 5))
    assert output['dense_feature'].shape == (0, 256, 32, 32)
    assert all(item.shape == (0, 64, 256)
               for item in output['memories'])


def test_contour_conditioner_chunks_rois():
    module = ContourEvidenceConditioner(roi_chunk_size=1)
    rois = torch.tensor([[0., 8., 8., 120., 120.],
                         [0., 16., 12., 112., 124.]])
    output = module(_features(), rois)
    assert output['dense_feature'].shape == (2, 256, 32, 32)
    assert all(item.shape == (2, 64, 256)
               for item in output['memories'])


def test_sfe_ablation_bypasses_feature_mask_for_downstream_dense_map():
    module = ContourEvidenceConditioner(variant='sfe', mask_mode='soft')
    rois = torch.tensor([[0., 8., 8., 120., 120.]])
    output = module(_features(), rois)
    expected = sum(output['roi_features']) / 3.0
    assert torch.equal(output['feature_mask'],
                       torch.ones_like(output['feature_mask']))
    assert torch.allclose(output['dense_feature'], expected, atol=1e-6)
