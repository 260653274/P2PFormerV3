from packaging.version import InvalidVersion, Version

MMCV_REQUIRED_VERSION = '1.7.2'
TORCH_DEVICE_STREAM_MIN_VERSION = '2.8'


def check_mmcv_version(version):
    try:
        parsed_version = Version(version)
    except InvalidVersion as exc:
        raise RuntimeError(f'Invalid MMCV version: {version!r}') from exc

    required_version = Version(MMCV_REQUIRED_VERSION)
    if parsed_version != required_version:
        raise RuntimeError(
            'P2PFormer embeds MMDetection 2.25.1 and requires '
            f'mmcv-full=={MMCV_REQUIRED_VERSION}, but MMCV {version} is '
            'installed. MMCV 2.x is not a drop-in replacement: it removes '
            'mmcv.runner, mmcv.parallel, and the legacy Config/Registry APIs. '
            'Use the pinned MMCV 1.x runtime or migrate the complete project '
            'to MMDetection 3.x and MMEngine.')

    return parsed_version


def patch_mmcv_parallel_for_torch():
    """Adapt MMCV 1.x scatter to the PyTorch 2.8 device API."""
    import torch

    try:
        torch_version = Version(torch.__version__)
    except InvalidVersion as exc:
        raise RuntimeError(
            f'Invalid PyTorch version: {torch.__version__!r}') from exc

    if torch_version < Version(TORCH_DEVICE_STREAM_MIN_VERSION):
        return False

    from mmcv.parallel import _functions as mmcv_functions

    original_get_stream = mmcv_functions._get_stream
    if getattr(original_get_stream, '_p2pformer_device_adapter', False):
        return True

    def get_stream(device):
        if isinstance(device, int):
            device = (
                torch.device('cpu') if device < 0
                else torch.device('cuda', device))
        return original_get_stream(device)

    get_stream._p2pformer_device_adapter = True
    mmcv_functions._get_stream = get_stream
    return True
