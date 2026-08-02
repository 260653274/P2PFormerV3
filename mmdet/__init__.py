# Copyright (c) OpenMMLab. All rights reserved.
import warnings

warnings.filterwarnings(
    'ignore',
    message=r'On January 1, 2023, MMCV will release v2\.0\.0',
    category=UserWarning,
    module=r'mmcv')

import mmcv  # noqa: E402

from .mmcv_compat import (  # noqa: E402
    MMCV_REQUIRED_VERSION, check_mmcv_version,
    patch_mmcv_parallel_for_torch)
from .version import __version__, short_version  # noqa: E402

check_mmcv_version(mmcv.__version__)
patch_mmcv_parallel_for_torch()

__all__ = [
    '__version__', 'short_version', 'MMCV_REQUIRED_VERSION',
    'check_mmcv_version', 'patch_mmcv_parallel_for_torch'
]
