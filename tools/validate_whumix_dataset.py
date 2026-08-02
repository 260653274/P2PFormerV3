#!/usr/bin/env python
"""Build WHU-Mix through MMDetection and execute its real data pipeline."""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
from mmdet.datasets import build_dataset
from mmcv import Config
from mmcv.parallel import DataContainer


EXPECTED_LENGTHS = {
    'train': 43614,
    'val': 2911,
    'test1': 11675,
    'test2': 6011,
}


def value(value):
    """Unwrap an MMCV DataContainer without assuming a tensor type."""
    return value.data if isinstance(value, DataContainer) else value


def shape_of(item):
    item = value(item)
    shape = getattr(item, 'shape', None)
    if shape is not None:
        return tuple(shape)
    try:
        return (len(item), )
    except TypeError:
        return None


def validate_split(name, dataset):
    if len(dataset) == 0:
        raise RuntimeError(f'{name} dataset is empty')
    expected_length = EXPECTED_LENGTHS[name]
    if len(dataset) != expected_length:
        raise RuntimeError(
            f'{name} length mismatch: {len(dataset)} != {expected_length}')

    sample = dataset[0]
    required = {'img'}
    if name == 'train':
        required.update({'gt_bboxes', 'gt_labels'})
        # The base dataset config uses the legacy contour pipeline, while the
        # released P2PFormer configs replace it with the line/corner pipeline.
        # Validate the schema actually selected after config inheritance.
        if 'gt_lines' in sample:
            required.update({
                'gt_lines', 'reference_points', 'matched_idxs'
            })
        elif 'gt_masks' in sample:
            required.update({
                'gt_masks', 'gt_polys', 'key_points_masks', 'key_points'
            })
        else:
            raise RuntimeError(
                'train sample matches neither the P2PFormer primitive schema '
                'nor the legacy contour schema')
    missing = sorted(required.difference(sample))
    if missing:
        raise RuntimeError(f'{name} sample misses keys: {missing}')

    print(f'{name}: length={len(dataset)}')
    print(f'{name}: keys={sorted(sample)}')
    for key in sorted(required):
        print(f'{name}: {key}.shape={shape_of(sample[key])}')

    if name == 'train':
        bboxes = value(sample['gt_bboxes'])
        labels = value(sample['gt_labels'])
        if len(bboxes) != len(labels):
            raise RuntimeError(
                f'train bbox/label mismatch: {len(bboxes)} != {len(labels)}')
        if not np.isfinite(bboxes.cpu().numpy()).all():
            raise RuntimeError('train bboxes contain non-finite values')
        print(f'train: instances={len(labels)}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='p2pformer/configs/_base_/datasets/whu-mix_line.py')
    parser.add_argument(
        '--dataset-root',
        help='Override the WHU-Mix root encoded in the config.')
    parser.add_argument(
        '--all-splits',
        action='store_true',
        help='Also build the full test1 and test2 evaluation pipelines.')
    return parser.parse_args()


def set_dataset_paths(cfg, dataset_root):
    dataset_root = Path(dataset_root).resolve()
    paths = {
        'train': ('train.json', 'image'),
        'val': ('val/val.json', 'val/image'),
        'test': ('val/val.json', 'val/image'),
    }
    for split, (annotation, images) in paths.items():
        split_cfg = cfg.data[split]
        split_cfg.ann_file = str(dataset_root / annotation)
        split_cfg.img_prefix = f'{dataset_root / images}/'


def build_test_split(cfg, dataset_root, split):
    paths = {
        'test1': ('test1/test-1.json', 'test1/image'),
        'test2': ('test2/test-2.json', 'test2/image'),
    }
    annotation, images = paths[split]
    split_cfg = copy.deepcopy(cfg.data.test)
    split_cfg.ann_file = str(dataset_root / annotation)
    split_cfg.img_prefix = f'{dataset_root / images}/'
    split_cfg.test_mode = True
    return build_dataset(split_cfg)


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)

    cfg = Config.fromfile(args.config)
    if args.dataset_root:
        set_dataset_paths(cfg, args.dataset_root)
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)

    validate_split('train', train_dataset)
    validate_split('val', val_dataset)
    if args.all_splits:
        dataset_root = Path(args.dataset_root).resolve() if args.dataset_root \
            else (repo / '../datasets/WHU-Mix').resolve()
        validate_split(
            'test1', build_test_split(cfg, dataset_root, 'test1'))
        validate_split(
            'test2', build_test_split(cfg, dataset_root, 'test2'))
    print('WHU-Mix runtime validation: PASS')


if __name__ == '__main__':
    main()
