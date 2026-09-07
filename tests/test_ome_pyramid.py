"""The multiscale pyramid builder.

[downsample_array][abczarr.ome.pyramid.downsample_array] coarsens one array,
and [create_pyramid][abczarr.ome.pyramid.create_pyramid] builds a whole
pyramid and records it as OME metadata through an
[ImageConfig][abczarr.ome.config.ImageConfig]. These tests pin the windowed
reduction, the per-axis factor resolution, the level count, the recorded
coordinate transforms, and the propagation of dimension names.
"""

import pathlib

import numpy as np
import pytest

import abczarr
from abczarr.ome import create_pyramid, default_levels, downsample_array

# a real backend is needed to create and read arrays
pytest.importorskip("zarr")


def _group(tmp_path: pathlib.Path, name: str = "img.zarr") -> object:
    return abczarr.create_group(str(pathlib.Path(tmp_path) / name))


def _base(
    group: object,
    data: "np.ndarray",
    *,
    dimension_names: object = ("y", "x"),
    chunks: object = (4, 4),
) -> object:
    arr = group.create_array(
        "0",
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        dimension_names=dimension_names,
    )
    arr.store(data)
    return arr


def _dataset_scale(dataset: object) -> list:
    for t in dataset.coordinateTransformations:
        if t.type == "scale":
            return t.scale
    return None


def _dataset_translation(dataset: object) -> object:
    for t in dataset.coordinateTransformations:
        if t.type == "translation":
            return t.translation
    return None


# --- downsample_array ------------------------------------------------------


def test_downsample_array_halves_with_the_window_mean(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(group, np.arange(16, dtype="float64").reshape(4, 4))
    coarse = downsample_array(group, "0", "1")
    assert coarse.shape == (2, 2)
    # each coarse voxel is the mean of its 2x2 window
    assert np.array_equal(
        np.asarray(coarse[:, :]), [[2.5, 4.5], [10.5, 12.5]]
    )


def test_a_factor_of_one_leaves_an_axis_at_full_resolution(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(
        group,
        np.arange(24, dtype="float64").reshape(2, 3, 4),
        dimension_names=("c", "y", "x"),
        chunks=(2, 3, 4),
    )
    # halve y and x, keep the channel axis whole
    coarse = downsample_array(group, "0", "1", factor={"c": 1})
    assert coarse.shape == (2, 1, 2)


def test_downsample_array_rejects_an_unknown_reduction(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(group, np.zeros((4, 4), dtype="float64"))
    with pytest.raises(ValueError, match="unknown reduction"):
        downsample_array(group, "0", "1", reduction="mode")


# --- default_levels --------------------------------------------------------


def test_default_levels_counts_down_to_one_chunk() -> None:
    # 64 -> 32 -> 16 -> 8 (the chunk), three halvings
    assert default_levels((64, 64), (8, 8), factor=2) == 3
    assert default_levels((8, 8), (8, 8), factor=2) == 0


# --- create_pyramid --------------------------------------------------------


def test_create_pyramid_builds_levels_and_records_metadata(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(group, np.arange(64, dtype="float64").reshape(8, 8))
    levels = create_pyramid(group, "0", levels=2, scale=[0.5, 0.5])
    assert [a.shape for a in levels] == [(8, 8), (4, 4), (2, 2)]

    ome = group.ome
    multiscale = ome.multiscales[0]
    assert [d.path for d in multiscale.datasets] == ["0", "1", "2"]
    # window strategy: level l has scale s0*2**l and offset s0*(2**l - 1)/2
    assert _dataset_scale(multiscale.datasets[0]) == [0.5, 0.5]
    assert _dataset_scale(multiscale.datasets[1]) == [1.0, 1.0]
    assert _dataset_translation(multiscale.datasets[1]) == [0.25, 0.25]
    assert _dataset_scale(multiscale.datasets[2]) == [2.0, 2.0]


def test_create_pyramid_propagates_dimension_names(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(
        group,
        np.arange(16, dtype="float64").reshape(4, 4),
        dimension_names=("y", "x"),
    )
    create_pyramid(group, "0", levels=1)
    assert tuple(group["1"].metadata.dimension_names) == ("y", "x")
    axes = [a.name for a in group.ome.multiscales[0].axes]
    assert axes == ["y", "x"]


def test_create_pyramid_stops_when_no_axis_can_shrink(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(group, np.arange(4, dtype="float64").reshape(2, 2), chunks=(2, 2))
    # asked for three levels, but a 2x2 halves once to 1x1 and then stops
    levels = create_pyramid(group, "0", levels=3)
    assert [a.shape for a in levels] == [(2, 2), (1, 1)]


def test_create_pyramid_names_levels_by_scale(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(group, np.arange(64, dtype="float64").reshape(8, 8))
    create_pyramid(group, "0", levels=2, name="s{scale}")
    assert "s2" in group.keys()
    assert "s4" in group.keys()


def test_create_pyramid_lowers_to_a_requested_version(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(group, np.arange(16, dtype="float64").reshape(4, 4))
    create_pyramid(group, "0", levels=1, version="0.4")
    assert group.ome.version == "0.4"


def test_create_pyramid_needs_axes_without_dimension_names(
    tmp_path: pathlib.Path,
) -> None:
    group = _group(tmp_path)
    _base(
        group,
        np.arange(16, dtype="float64").reshape(4, 4),
        dimension_names=None,
    )
    with pytest.raises(ValueError, match="dimension names"):
        create_pyramid(group, "0", levels=1)
    # passing axes explicitly works
    levels = create_pyramid(group, "0", levels=1, axes=["y", "x"])
    assert len(levels) == 2
