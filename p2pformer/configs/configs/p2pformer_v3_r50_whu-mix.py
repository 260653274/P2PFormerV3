"""P2PFormerV3 V1 configuration for WHU-Mix.

The detector, backbone, deformable neck, and FCOS branch are inherited from
the released P2PFormer R50 configuration.  Only the contour/generation head
and its data contract are replaced.
"""

_base_ = './p2pformer_corner_r50_whu-mix.py'

model = dict(
    backbone=dict(init_cfg=None),
    line_fpn=True,
    line_fpn_start_level=0,
    line_head=dict(
        _delete_=True,
        type='P2PFormerV3Head',
        in_channels=256,
        num_slots=40,
        hidden_dim=256,
        expand_scale=1.1,
        state_mean=[
            0.49876153339294055,
            0.49343782920772017,
            -1.2525931130012171,
            -1.2522010412292803,
        ],
        state_std=[
            0.3036825069114701,
            0.3040210909059293,
            0.2355359596146996,
            0.23731648461641258,
        ],
        conditioner=dict(
            crop_size=32,
            sample_side=8,
            num_heads=8,
            feature_strides=(4, 8, 16),
            offset_scale=4.0,
            mask_floor=0.1,
            contour_half_width=8.0,
            dilation_temperature=0.1,
            mask_mode='soft',
            variant='full',
            roi_chunk_size=16),
        diffusion_steps=1000,
        inference_nfe=2,
        inference_seed=3407,
        validity_threshold=0.5,
        proposal_probability=0.1,
        proposal_warmup_steps=1000,
        loss_weights=dict(
            contour=1.0,
            x0=1.0,
            support_box=2.0,
            validity=1.0,
            primitive=5.0,
            topology=1.0)))

# The V3 target builder consumes the native variable-length corner triples.
# V1's 36 reference points and matched order bins are intentionally omitted.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,
         poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(640, 320), (640, 640)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.530, 116.280, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(
        type='LineSampleWithAlignReference',
        point_nums=36,
        reset_bbox=True,
        with_reference_points=False),
    dict(type='LineDefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_lines']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline))

# Initialize the unchanged scene branch from the released V1 checkpoint.  The
# replaced line head is loaded non-strictly by MMCV.
load_from = 'checkpoints/p2pformer_corner_r50_whu_mix.pth'

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'neck': dict(lr_mult=0.1, decay_mult=1.0),
            'bbox_head': dict(lr_mult=0.1, decay_mult=1.0),
        }))

find_unused_parameters = True
