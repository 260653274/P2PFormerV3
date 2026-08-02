import math

import pytest
import torch
import torch.nn as nn

from p2pformer.models.p2pformer_v3_diffusion import (
    CoarseToFineSetDenoiser, CosineDiffusionSchedule, FinalPrimitiveHead,
    P2PFormerV3Loss, P2PFormerV3PrimitiveLoss, SuccessorTopologyHead,
    SupportBoxCodec, SupportTargetBuilder, ddim_sample, make_anchor_lattice,
    masked_log_sinkhorn, solve_cycles, topology_loss)


def _square_primitives(dtype=torch.float32):
    vertices = torch.tensor([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
                            dtype=dtype)
    triples = []
    for index in range(len(vertices)):
        triples.append(
            torch.cat((vertices[(index - 1) % 4], vertices[index],
                       vertices[(index + 1) % 4])))
    return torch.stack(triples)


def _successors(batch_size, num_slots, active_count):
    clockwise = torch.full((batch_size, num_slots), -1, dtype=torch.long)
    counterclockwise = clockwise.clone()
    for batch_index in range(batch_size):
        for index in range(active_count):
            clockwise[batch_index, index] = (index + 1) % active_count
            counterclockwise[batch_index, index] = (index - 1) % active_count
    return clockwise, counterclockwise


def test_cosine_schedule_and_oracle_ddim():
    schedule = CosineDiffusionSchedule(num_timesteps=1000)
    assert schedule.betas.shape == (1000, )
    assert bool(torch.isfinite(schedule.alpha_bars).all())
    assert bool((schedule.betas > 0).all())
    assert bool((schedule.betas < 1).all())
    assert bool((schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]).all())
    expected = [
        0.999958715775, 0.847012161327, 0.493843590440, 0.144272102386,
        2.428766907e-9
    ]
    actual = [schedule.alpha_bars[i].item() for i in (0, 249, 499, 749, 999)]
    assert torch.allclose(
        torch.tensor(actual), torch.tensor(expected), rtol=2e-5, atol=1e-10)
    assert schedule.inference_timesteps(1) == (999, )
    assert schedule.inference_timesteps(2) == (999, 499)
    assert schedule.inference_timesteps(4) == (999, 749, 499, 249)

    clean = torch.randn(2, 5, 4)
    noise = torch.randn_like(clean)
    t = torch.tensor([999, 999])
    noisy = schedule.q_sample(clean, t, noise)
    stepped = schedule.ddim_step(noisy, clean, t, 499)
    alpha_s = schedule.alpha_bar(499, clean)
    expected_step = (
        torch.sqrt(alpha_s) * clean + torch.sqrt(1.0 - alpha_s) * noise)
    assert torch.allclose(stepped, expected_step, atol=2e-5, rtol=2e-5)
    recovered = schedule.ddim_step(stepped, clean, torch.tensor([499, 499]),
                                   None)
    assert torch.equal(recovered, clean.float())


@pytest.mark.parametrize('num_steps', [1, 2, 4])
def test_ddim_sampler_uses_exact_nfe(num_steps):
    schedule = CosineDiffusionSchedule()
    clean = torch.randn(2, 3, 4)

    class OracleDenoiser(nn.Module):

        def __init__(self, target):
            super(OracleDenoiser, self).__init__()
            self.target = target
            self.calls = []

        def prepare_condition(self, memories, pos):
            return ()

        def forward(self,
                    z_t,
                    t,
                    anchors,
                    memories,
                    pos,
                    c_img,
                    condition_cache=None):
            self.calls.append(int(t[0]))
            hidden = z_t.new_zeros((z_t.size(0), z_t.size(1), 8))
            return self.target, hidden

    model = OracleDenoiser(clean)
    sampled, hidden = ddim_sample(
        model,
        schedule,
        torch.randn_like(clean),
        torch.zeros(3, 4), [], [],
        torch.zeros(2, 8),
        num_steps=num_steps)
    assert model.calls == list(schedule.inference_timesteps(num_steps))
    assert torch.equal(sampled, clean.float())
    assert hidden.shape == (2, 3, 8)


def test_support_target_builder_batches_variable_polygons():
    mean = [0.5, 0.5, math.log(0.25), math.log(0.25)]
    std = [0.25, 0.25, 0.5, 0.5]
    codec = SupportBoxCodec(mean, std)
    anchors = make_anchor_lattice()
    builder = SupportTargetBuilder(codec, anchors=anchors)
    primitives = _square_primitives()
    support_boxes = builder.build_support_boxes(primitives)
    assert support_boxes.shape == (4, 4)
    assert bool(torch.isfinite(support_boxes).all())
    assert bool((support_boxes[:, 2:] >= 4.0 / 32.0).all())

    targets = builder.build([primitives, primitives[:3]])
    assert targets['z0'].shape == (2, 40, 4)
    assert targets['active_mask'].shape == (2, 40)
    assert targets['primitive_targets'].shape == (2, 40, 6)
    assert targets['successors_cw'].shape == (2, 40)
    assert targets['anchors'].shape == (40, 4)
    assert targets['active_mask'].sum(dim=1).tolist() == [4, 3]
    assert bool(torch.isfinite(targets['z0']).all())
    assert torch.allclose(
        codec.decode(targets['z0']),
        targets['target_boxes'],
        atol=2e-6,
        rtol=2e-6)
    for batch_index, active_count in enumerate((4, 3)):
        active = targets['active_mask'][batch_index]
        successor = targets['successors_cw'][batch_index, active]
        assert successor.numel() == active_count
        assert bool(active[successor].all())


def test_denoiser_shapes_cache_equivariance_and_gradient():
    torch.manual_seed(3)
    batch_size, num_slots, hidden_dim = 1, 40, 256
    model = CoarseToFineSetDenoiser(
        num_slots=num_slots, hidden_dim=hidden_dim, dropout=0.0)
    model.eval()
    state = torch.randn(batch_size, num_slots, 4)
    anchors = make_anchor_lattice()
    memories = [torch.randn(batch_size, 64, hidden_dim) for _ in range(3)]
    positions = [torch.randn(batch_size, 64, hidden_dim) for _ in range(3)]
    image_context = torch.randn(batch_size, hidden_dim)
    timesteps = torch.tensor([749])

    pred, hidden = model(state, timesteps, anchors, memories, positions,
                         image_context)
    cache = model.prepare_condition(memories, positions)
    pred_cached, hidden_cached = model(
        state,
        timesteps,
        anchors,
        memories,
        positions,
        image_context,
        condition_cache=cache)
    assert pred.shape == (batch_size, num_slots, 4)
    assert hidden.shape == (batch_size, num_slots, hidden_dim)
    assert bool(torch.isfinite(pred).all())
    assert bool(torch.isfinite(hidden).all())
    assert torch.allclose(pred, pred_cached, atol=1e-7, rtol=1e-7)
    assert torch.allclose(hidden, hidden_cached, atol=1e-7, rtol=1e-7)

    permutation = torch.randperm(num_slots)
    permuted_pred, _ = model(state[:, permutation], timesteps,
                             anchors[permutation], memories, positions,
                             image_context)
    assert torch.allclose(
        permuted_pred, pred[:, permutation], atol=2e-6, rtol=2e-6)

    loss = pred.square().mean() + hidden.square().mean()
    loss.backward()
    assert model.output_head.weight.grad is not None
    assert bool(torch.isfinite(model.output_head.weight.grad).all())
    assert model.blocks[0].modulation[-1].weight.grad is not None
    assert bool(
        torch.isfinite(model.blocks[0].modulation[-1].weight.grad).all())


def test_denoiser_accepts_empty_roi_batch():
    model = CoarseToFineSetDenoiser(
        num_slots=4, hidden_dim=32, num_heads=4, ffn_dim=64, time_embed_dim=16)
    state = torch.empty(0, 4, 4)
    anchors = make_anchor_lattice(rows=2, cols=2)
    memories = [torch.empty(0, 5, 32) for _ in range(3)]
    positions = [torch.empty(0, 5, 32) for _ in range(3)]
    pred, hidden = model(state, torch.empty(0, dtype=torch.long), anchors,
                         memories, positions, torch.empty(0, 32))
    assert pred.shape == (0, 4, 4)
    assert hidden.shape == (0, 4, 32)


def test_sinkhorn_topology_and_primitive_loss_are_stable():
    logits = torch.zeros(3, 4, 4, requires_grad=True)
    active = torch.tensor([[False, False, False, False],
                           [True, False, False, False],
                           [True, True, True, False]])
    log_prob = masked_log_sinkhorn(logits, active, num_iters=20)
    assert not bool(torch.isnan(log_prob).any())
    probability = log_prob[2, :3, :3].exp()
    assert torch.allclose(probability.sum(dim=-1), torch.ones(3), atol=1e-4)
    assert torch.allclose(probability.sum(dim=-2), torch.ones(3), atol=1e-4)
    assert torch.equal(torch.diagonal(probability), torch.zeros(3))

    uniform_logits = torch.zeros(1, 4, 4, requires_grad=True)
    active_all = torch.ones(1, 4, dtype=torch.bool)
    clockwise, counterclockwise = _successors(1, 4, 4)
    loss_topology = topology_loss(uniform_logits, active_all, clockwise,
                                  counterclockwise)
    assert torch.allclose(
        loss_topology, torch.tensor(math.log(3.0)), atol=1e-5)

    target = _square_primitives().unsqueeze(0)
    primitive_loss = P2PFormerV3PrimitiveLoss()
    loss_forward = primitive_loss(target, target,
                                  torch.ones(1, 4, dtype=torch.bool))
    reversed_target = torch.cat(
        (target[..., 4:6], target[..., 2:4], target[..., 0:2]), dim=-1)
    loss_reverse = primitive_loss(target, reversed_target,
                                  torch.ones(1, 4, dtype=torch.bool))
    assert loss_forward.item() < 1e-7
    assert loss_reverse.item() < 1e-7

    zero_pred = torch.zeros_like(target, requires_grad=True)
    finite_loss = primitive_loss(zero_pred, target,
                                 torch.ones(1, 4, dtype=torch.bool))
    finite_loss.backward()
    assert torch.isfinite(finite_loss)
    assert bool(torch.isfinite(zero_pred.grad).all())


def test_final_heads_composite_loss_and_gradients():
    torch.manual_seed(7)
    batch_size, num_slots, hidden_dim = 2, 6, 32
    primitive_head = FinalPrimitiveHead(hidden_dim=hidden_dim)
    topology_head = SuccessorTopologyHead(
        hidden_dim=hidden_dim, topology_dim=16)
    hidden = torch.randn(batch_size, num_slots, hidden_dim, requires_grad=True)
    boxes = torch.tensor([0.5, 0.5, 0.3,
                          0.3]).reshape(1, 1,
                                        4).repeat(batch_size, num_slots,
                                                  1).requires_grad_()
    dense = torch.randn(batch_size, hidden_dim, 32, 32, requires_grad=True)
    primitives, validity_logits, fused = primitive_head(hidden, boxes, dense)
    successor_logits = topology_head(fused, primitives)
    assert primitives.shape == (batch_size, num_slots, 6)
    assert validity_logits.shape == (batch_size, num_slots)
    assert fused.shape == (batch_size, num_slots, hidden_dim)
    assert successor_logits.shape == (batch_size, num_slots, num_slots)
    assert bool(torch.isfinite(primitives).all())
    assert bool(torch.isfinite(validity_logits).all())
    assert bool(torch.isfinite(successor_logits).all())
    assert primitive_head.last_oob_rate.shape == (batch_size, )
    assert bool((primitive_head.last_oob_rate == 0).all())

    validity_targets = torch.zeros(batch_size, num_slots)
    validity_targets[:, :4] = 1.0
    clockwise, counterclockwise = _successors(batch_size, num_slots, 4)
    pred_state = torch.randn(batch_size, num_slots, 4, requires_grad=True)
    target_state = torch.randn_like(pred_state)
    target_boxes = boxes.detach().clone()
    target_primitives = torch.rand(batch_size, num_slots, 6)
    criterion = P2PFormerV3Loss(sinkhorn_iters=20)
    losses = criterion(pred_state, target_state, boxes, target_boxes,
                       validity_logits, validity_targets, primitives,
                       target_primitives, successor_logits, clockwise,
                       counterclockwise)
    assert set(losses) == {
        'loss_x0', 'loss_sbox', 'loss_valid', 'loss_primitive',
        'loss_topology', 'loss_total'
    }
    assert all(bool(torch.isfinite(value)) for value in losses.values())
    losses['loss_total'].backward()
    for gradient in (hidden.grad, boxes.grad, dense.grad, pred_state.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
    assert topology_head.query.weight.grad is not None
    assert bool(torch.isfinite(topology_head.query.weight.grad).all())


def test_hard_cycle_solver_merges_subtours():
    primitives = _square_primitives().unsqueeze(0)
    logits = torch.zeros(1, 4, 4)
    logits[:, torch.arange(4), torch.arange(4)] = -1e4
    logits[0, 0, 1] = 5.0
    logits[0, 1, 0] = 5.0
    logits[0, 2, 3] = 5.0
    logits[0, 3, 2] = 5.0
    validity = torch.full((1, 4), 10.0)
    result = solve_cycles(logits, primitives, validity)[0]
    assert result['valid']
    assert result['indices'].shape == (4, )
    assert torch.unique(result['indices']).numel() == 4
    assert result['subtour_count'] == 2
    assert result['merge_count'] == 1


def test_cycle_solver_rejects_nonfinite_and_degenerate_geometry():
    logits = torch.zeros(1, 4, 4)
    logits[:, torch.arange(4), torch.arange(4)] = -1e4
    validity = torch.full((1, 4), 10.0)

    collinear = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
                               [0.0, 0.0, 0.2, 0.0, 0.3, 0.0],
                               [0.0, 0.0, 0.4, 0.0, 0.5, 0.0],
                               [0.0, 0.0, 0.6, 0.0, 0.7, 0.0]]])
    result = solve_cycles(logits, collinear, validity)[0]
    assert not result['valid']
    assert result['failure'] == 'degenerate_polygon'

    nonfinite = collinear.clone()
    nonfinite[0, 0, 2] = float('nan')
    result = solve_cycles(logits, nonfinite, validity)[0]
    assert not result['valid']
    assert result['failure'] == 'nonfinite_geometry'

    nonfinite_logits = logits.clone()
    nonfinite_logits[0, 0, 1] = float('nan')
    result = solve_cycles(nonfinite_logits, collinear, validity)[0]
    assert not result['valid']
    assert result['failure'] == 'nonfinite_geometry'


def test_cycle_solver_detects_collinear_nonadjacent_overlap():
    centers = torch.tensor([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0],
                            [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    primitives = torch.cat((torch.roll(centers, 1, 0), centers,
                            torch.roll(centers, -1, 0)), dim=-1).unsqueeze(0)
    logits = torch.full((1, 6, 6), -10.0)
    for index in range(6):
        logits[0, index, (index + 1) % 6] = 10.0
    validity = torch.full((1, 6), 10.0)
    result = solve_cycles(
        logits, primitives, validity, max_2opt_iterations=0)[0]
    assert not result['valid']
    assert result['failure'] == 'intersection_cap'
