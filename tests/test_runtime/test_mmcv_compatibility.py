import pytest

from mmdet.mmcv_compat import (MMCV_REQUIRED_VERSION, check_mmcv_version,
                               patch_mmcv_parallel_for_torch)


def test_required_mmcv_version():
    assert str(check_mmcv_version(MMCV_REQUIRED_VERSION)) == '1.7.2'


@pytest.mark.parametrize(
    'version', ['1.7.1', '1.7.2rc1', '2.0.0', '2.2.0'])
def test_incompatible_mmcv_versions(version):
    with pytest.raises(RuntimeError, match='not a drop-in replacement'):
        check_mmcv_version(version)


def test_invalid_mmcv_version():
    with pytest.raises(RuntimeError, match='Invalid MMCV version'):
        check_mmcv_version('not-a-version')


def test_mmcv_parallel_patch_is_idempotent():
    assert patch_mmcv_parallel_for_torch()
    assert patch_mmcv_parallel_for_torch()
