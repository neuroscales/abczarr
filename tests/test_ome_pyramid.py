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


def test_factor_1_on_an_axis_keeps_it_at_full_resolution(
    tmp_path: pathlib.Path,
) -> None:
    group = _group_with_base(tmp_path, shape=(3, 16, 16), chunks=(1, 4, 4))
    # a mapping halves the axes it does not mention, keeping axis 0
    made = downsample_array(group, "0", "1", factor={0: 1})
    assert made.shape == (3, 8, 8)


def test_per_axis_factor_sequence(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path, shape=(3, 16, 16), chunks=(1, 4, 4))
    made = downsample_array(group, "0", "1", factor=(1, 4, 2))
    assert made.shape == (3, 4, 8)


def test_factor_by_dimension_name(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "named.zarr")
    group = abczarr.create_group(root, overwrite=True)
    base = group.create_array(
        "0", shape=(3, 16, 16), dtype="float32", chunks=(1, 4, 4),
        dimension_names=["c", "y", "x"],
    )
    base[:] = 0.0
    made = downsample_array(group, "0", "1", factor={"c": 1})
    assert made.shape == (3, 8, 8)


def test_median_reduction(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path)
    made = downsample_array(group, "0", "1", reduction="median")
    assert made.shape == (8, 8)


def test_unknown_reduction_is_rejected(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path)
    with pytest.raises(ValueError, match="unknown reduction"):
        downsample_array(group, "0", "1", reduction="bogus")


def test_name_by_scale_factor(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path)
    levels = create_pyramid(group, "0", levels=2, name="{scale}")
    # a halving pyramid: levels named by cumulative factor, 2 then 4
    assert sorted(group.keys()) == ["0", "2", "4"]
    assert [tuple(a.shape) for a in levels] == [(16, 16), (8, 8), (4, 4)]


def test_name_callable(tmp_path: pathlib.Path) -> None:
    group = _group_with_base(tmp_path)
    create_pyramid(group, "0", levels=2, name=lambda level: f"s{level}")
    assert sorted(group.keys()) == ["0", "s1", "s2"]


def test_default_levels_counts_divisions_to_a_chunk() -> None:
    assert default_levels((16, 16), (4, 4)) == 2
    assert default_levels((64,), (4,)) == 4
    # an axis kept at factor 1 does not drive the count
    assert default_levels((100, 16, 16), (100, 4, 4), factor={0: 1}) == 2
    # a larger factor reaches the chunk in fewer levels
    assert default_levels((64,), (4,), factor=4) == 2


def test_factor_mapping_none_key_sets_the_default(
    tmp_path: pathlib.Path,
) -> None:
    from abczarr.ome.pyramid import _resolve_factors

    # a None key sets the default for axes not otherwise mentioned
    assert _resolve_factors({None: 4, 0: 1}, 3) == (1, 4, 4)
    # without a None key, the default is a halving
    assert _resolve_factors({0: 1}, 3) == (1, 2, 2)


def test_create_pyramid_writes_multiscales_metadata(
    tmp_path: pathlib.Path,
) -> None:
    from abczarr.ome.metadata.v0_5.ome import OMEImage

    root = str(tmp_path / "named.zarr")
    group = abczarr.create_group(root, overwrite=True)
    base = group.create_array(
        "0", shape=(3, 16, 16), dtype="float32", chunks=(1, 4, 4),
        dimension_names=["c", "y", "x"],
    )
    base[:] = 0.0
    # a channel axis kept at full resolution, the two space axes halved
    create_pyramid(group, "0", levels=2, factor={"c": 1})

    ome = group.attrs["ome"]
    assert ome["version"] == "0.5"
    (multiscale,) = ome["multiscales"]
    # one axis per dimension, typed from its name (c -> channel, x/y -> space)
    assert multiscale["axes"] == [
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    # one dataset per level, each with the scale back onto the base resolution
    assert [d["path"] for d in multiscale["datasets"]] == ["0", "1", "2"]
    assert [
        d["coordinateTransformations"][0]["scale"]
        for d in multiscale["datasets"]
    ] == [[1.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 4.0, 4.0]]
    # the written metadata is valid OME v0.5, and survives a re-open
    OMEImage.from_dict(ome)
    assert "ome" in dict(abczarr.open_group(root).attrs)


def test_create_pyramid_names_unnamed_axes_generically(
    tmp_path: pathlib.Path,
) -> None:
    group = _group_with_base(tmp_path, shape=(16, 16), chunks=(4, 4))
    create_pyramid(group, "0", levels=1)
    axes = group.attrs["ome"]["multiscales"][0]["axes"]
    # an array without dimension names gets generic, space-typed axes
    assert axes == [
        {"name": "dim_0", "type": "space"},
        {"name": "dim_1", "type": "space"},
    ]


def test_write_metadata_false_skips_the_multiscales(
    tmp_path: pathlib.Path,
) -> None:
    group = _group_with_base(tmp_path)
    create_pyramid(group, "0", levels=1, write_metadata=False)
    assert "ome" not in dict(group.attrs)
