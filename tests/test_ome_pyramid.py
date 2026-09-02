"""Building a multiscale pyramid: downsample one array, or a whole pyramid.

Needs zarr-python (to create the arrays) and dask (the downsampling engine),
so it runs on the coverage CI leg where both are installed.
"""

import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("dask.array")

import abczarr  # noqa: E402
from abczarr.ome import (  # noqa: E402
    create_pyramid,
    default_levels,
    downsample_array,
)


def _group_with_base(
    tmp_path: pathlib.Path,
    shape: tuple = (16, 16),
    chunks: tuple = (4, 4),
) -> object:
    group = abczarr.create_group(str(tmp_path / "img.zarr"), overwrite=True)
    base = group.create_array("0", shape=shape, dtype="float32", chunks=chunks)
    base[:] = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
    return group


def test_downsample_array_halves_and_averages(
    tmp_path: pathlib.Path,
) -> None:
    group = _group_with_base(tmp_path)
    made = downsample_array(group, "0", "1")
    assert made.shape == (8, 8)
    # the top-left coarse voxel is the mean of the 2x2 window {0, 1, 16, 17}
    assert float(np.asarray(made[0, 0])) == pytest.approx(8.5)
    assert "1" in list(group.keys())


def test_create_pyramid_builds_all_levels(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path)
    levels = create_pyramid(group, "0", levels=2)
    assert [tuple(a.shape) for a in levels] == [(16, 16), (8, 8), (4, 4)]
    assert sorted(group.keys()) == ["0", "1", "2"]


def test_create_pyramid_default_depth_fills_to_a_chunk(
    tmp_path: pathlib.Path,
) -> None:
    group = _group_with_base(tmp_path, shape=(16, 16), chunks=(4, 4))
    levels = create_pyramid(group, "0")
    # 16 -> 8 -> 4 reaches the chunk size, so two extra levels
    assert len(levels) == 3
    assert tuple(levels[-1].shape) == (4, 4)


def test_no_pyramid_axis_is_left_at_full_resolution(
    tmp_path: pathlib.Path,
) -> None:
    group = _group_with_base(tmp_path, shape=(3, 16, 16), chunks=(1, 4, 4))
    made = downsample_array(group, "0", "1", no_pyramid_axis=0)
    assert made.shape == (3, 8, 8)


def test_median_reduction(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path)
    made = downsample_array(group, "0", "1", reduction="median")
    assert made.shape == (8, 8)


def test_unknown_reduction_is_rejected(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path)
    with pytest.raises(ValueError, match="unknown reduction"):
        downsample_array(group, "0", "1", reduction="bogus")


def test_default_levels_counts_halvings_to_a_chunk() -> None:
    assert default_levels((16, 16), (4, 4)) == 2
    assert default_levels((64,), (4,)) == 4
    # the excluded axis does not drive the count
    assert default_levels((100, 16, 16), (100, 4, 4), no_pyramid_axis=0) == 2
