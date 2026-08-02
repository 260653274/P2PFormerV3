# P2PFormerV3 V1 Implementation

> Status: implemented and locally smoke-tested on 2026-08-02.

This milestone implements the approved Micro Design as a new
`P2PFormerV3Head` while retaining the released ResNet-50, deformable
multiscale neck, FCOS detector, dataset, and external evaluation contract.
The implementation is intended to establish a reproducible architecture and
training path. The 20-iteration checkpoint described below is a systems smoke
test, not a quality result.

## Implemented path

1. P2/P3/P4 are RoI-aligned to `32×32` and processed by the reconstructed
   SFE/DFE contour conditioner. The predicted soft residual Feature Mask has a
   `0.1` floor and is supervised by a one-cell boundary target.
2. Each variable-length corner cycle is converted to local support AABBs,
   clean-coupled once to an `8×5` anchor lattice, and jointly slot-permuted.
   Components above 40 vertices use topology-preserving closed-curve
   Visvalingam simplification rather than truncation.
3. A cosine VP process corrupts standardized 4-D support states. A shared
   three-block AdaLN-Zero set denoiser predicts clean supports from cached
   coarse-to-fine contour memories.
4. Final support boxes sample `4×4` regions from the masked dense crop. The
   resulting features predict 6-D `[previous,current,next]` primitives and
   slot validity.
5. A pairwise successor head is trained with bidirectional log-Sinkhorn NLL.
   Inference uses Hungarian cycle cover, subtour merging, robust intersection
   checks, 2-opt, and clockwise winding normalization.
6. The selected cycle is encoded as binary slot scores and cycle ranks, so the
   existing detector can construct polygons without changing its public
   result type. Invalid, non-finite, degenerate, or unresolved cycles filter
   their bbox/label/mask entries in lockstep.
7. Training logs the final local-grid out-of-bound rate, alongside synthetic
   jitter rejection, proposal fallback, simplification, and active-slot
   diagnostics.

The training objective is emitted as six independent MMDetection losses:

`L_contour + L_x0 + 2 L_sbox + L_valid + 5 L_primitive + L_topology`.

No aggregate `loss_total` key is returned, because MMDetection automatically
sums every key containing `loss`.

## Configuration and data statistics

The executable configuration is
`p2pformer/configs/configs/p2pformer_v3_r50_whu-mix.py`. It initializes the
unchanged scene subsystem from the released V1 checkpoint and uses a reduced
learning rate for the backbone, neck, and FCOS head.

Support normalization was computed from the full WHU-Mix training JSON:

- 507,760 polygon components;
- 3,775,586 active supports;
- 2,139 components simplified to 40 vertices;
- state mean `[0.498762, 0.493438, -1.252593, -1.252201]`;
- state standard deviation `[0.303683, 0.304021, 0.235536, 0.237316]`.

The exact artifact is stored at
`p2pformer/configs/_base_/support_stats/whumix_p2pformerv3.json` and can be
regenerated with `tools/compute_p2pformerv3_stats.py`.

## Local verification record

- V3 subsystem parameters: 5,883,283.
- Full model parameters: 38,964,945 (`38.965 M`), versus V1 `41.275 M`.
- V1 checkpoint migration: all 590 shared keys load; every missing or
  unexpected key is confined to the intentionally replaced `line_head.*`.
- Directed unit tests: 37 passed across contour conditioning, diffusion,
  support coupling, primitive loss, topology, detector projection, and V3
  head integration at the time of this record.
- Real WHU-Mix single-batch forward/backward: all detector and V3 losses are
  finite; conditioner, denoiser, primitive, and topology branches all receive
  finite non-zero gradients. A `416×416`, seven-instance batch peaked near
  `906 MiB` of allocated CUDA memory in the directed check.
- 20-iteration training smoke test: completed without NaN or OOM and saved
  `work_dirs/p2pformerv3_v1_smoke/iter_20.pth`. Runner-reported peak memory was
  `4,033 MiB`; total loss was finite in every iteration. Contour CE changed
  from `0.6255` at iteration 1 to `0.3106` at iteration 20.
- Real validation-image inference: deterministic repeated output at the fixed
  seed. The 20-iteration model correctly rejected all cycles at the formal
  `0.5` validity threshold; this indicates an untrained validity branch, not
  convergence. A diagnostic-only `0.1` threshold exercised complete 1- and
  2-NFE polygon/raster paths for all 58 FCOS detections. It is not a reported
  accuracy setting.

## Reproduction commands

```bash
conda activate p2pformer
cd P2PFormer

python -m pytest -q \
  tests/test_models/test_p2pformer_v3_contour.py \
  tests/test_models/test_p2pformer_v3_diffusion.py \
  tests/test_models/test_p2pformer_v3_detector_contract.py \
  tests/test_models/test_p2pformer_v3_head.py

python tools/train.py \
  p2pformer/configs/configs/p2pformer_v3_r50_whu-mix.py \
  --work-dir work_dirs/p2pformerv3_v1_smoke \
  --no-validate --seed 3407 --deterministic \
  --cfg-options data.samples_per_gpu=1 data.workers_per_gpu=0 \
    runner.type=IterBasedRunner runner.max_epochs=None runner.max_iters=20 \
    lr_config.by_epoch=False lr_config.step='[15]' \
    checkpoint_config.by_epoch=False checkpoint_config.interval=20 \
    log_config.interval=1
```

## Known scientific boundary

The smoke run verifies implementation integrity only. It does not establish
that diffusion improves over a same-capacity deterministic refiner or over a
faithful P2PFormerV2 reconstruction. Full training, V2 reverse reproduction,
the deterministic 1/2-call controls, 1/2/4-NFE comparison, boundary metrics,
and the approved ablation matrix remain required for the paper claim.

The topology Sinkhorn uses 20 normalization iterations rather than the five
shown in the initial diagram. Five iterations were not sufficiently doubly
stochastic in directed numerical tests; the operation is still parameter-free
and negligible at 40 slots.
