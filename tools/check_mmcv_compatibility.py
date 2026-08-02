#!/usr/bin/env python
from importlib.metadata import PackageNotFoundError, version

import torch


def distribution_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    distributions = {
        name: distribution_version(name)
        for name in ('mmcv', 'mmcv-lite', 'mmcv-full')
    }
    if distributions['mmcv'] is not None:
        raise RuntimeError(
            'The MMCV 2.x distribution "mmcv" conflicts with this project.')
    if distributions['mmcv-lite'] is not None:
        raise RuntimeError(
            'The MMCV 2.x distribution "mmcv-lite" conflicts with this '
            'project and does not provide CUDA operators.')
    if distributions['mmcv-full'] != '1.7.2':
        raise RuntimeError(
            'Expected mmcv-full==1.7.2, found '
            f'{distributions["mmcv-full"]!r}.')

    import mmdet
    import mmcv
    from mmcv import Config
    from mmcv.parallel import DataContainer
    from mmcv.parallel._functions import Scatter
    from mmcv.runner import BaseModule
    from mmcv.utils import Registry

    legacy_apis = (Config, Registry, DataContainer, BaseModule)
    if mmcv.__version__ != distributions['mmcv-full']:
        raise RuntimeError(
            f'Imported MMCV {mmcv.__version__}, but the installed mmcv-full '
            f'distribution is {distributions["mmcv-full"]}.')

    print(f'MMCV: {mmcv.__version__}')
    print(f'MMDetection: {mmdet.__version__}')
    print(
        'Legacy APIs: ' +
        ', '.join(api.__name__ for api in legacy_apis) + ': PASS')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable to PyTorch.')

    scattered = Scatter.forward([0], torch.tensor([1.0]))
    if not scattered[0].is_cuda:
        raise RuntimeError('MMCV DataParallel scatter did not reach CUDA.')

    from mmcv.ops import nms

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]],
        device='cuda')
    scores = torch.tensor([0.9, 0.8], device='cuda')
    _, keep = nms(boxes, scores, 0.5)
    if keep.tolist() != [0]:
        raise RuntimeError(f'Unexpected CUDA NMS result: {keep.tolist()}')

    print(
        f'CUDA: {torch.version.cuda}, '
        f'GPU: {torch.cuda.get_device_name(0)}, '
        f'capability: {torch.cuda.get_device_capability(0)}')
    print('MMCV DataParallel scatter: PASS')
    print('MMCV CUDA NMS: PASS')
    print('P2PFormer MMCV compatibility: PASS')


if __name__ == '__main__':
    main()
