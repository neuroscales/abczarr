"""The zarr-python node adapters: open, read, write, and navigate a real
zarr v3 store through the uniform surface.

Runs only where zarr-python 3.x is installed (the coverage CI leg).
"""

import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from abczarr.abc.array import ZarrArray  # noqa: E402
from abczarr.abc.group import ZarrGroup  # noqa: E402
from abczarr.drivers.zarr_python import (  # noqa: E402
    ZarrPythonArray,
    ZarrPythonDriver,
    ZarrPythonGroup,
    ZarrPythonNode,
)
from abczarr.metadata.base import ArrayMetadata  # noqa: E402


def _open(path: str, mode: str = "r") -> object:
    """Open through the zarr-python driver, as abczarr.open would."""
    return ZarrPythonDriver().open(path, mode)


def _store(tmp_path: pathlib.Path) -> str:
    root = str(tmp_path / "data.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32"
    )
    array[:] = np.arange(64).reshape(8, 8)
    group.create_group("sub")
    return root


def test_both_node_kinds_share_a_common_base(tmp_path: pathlib.Path) -> None:
    # the array and group adapters share ZarrPythonNode, so open returns one
    # node type and the metadata/attrs/version accessors are written once
    root = _store(tmp_path)
    array = _open(root + "/img")
    group = _open(root)
    assert isinstance(array, ZarrPythonNode)
    assert isinstance(group, ZarrPythonNode)


# --------------------------------------------------------------------------
# open and navigate
# --------------------------------------------------------------------------


def test_open_returns_a_wrapped_group(tmp_path: pathlib.Path) -> None:
    node = _open(_store(tmp_path))
    assert isinstance(node, ZarrPythonGroup)
    assert isinstance(node, ZarrGroup)
    assert sorted(node.keys()) == ["img", "sub"]


def test_indexing_wraps_children(tmp_path: pathlib.Path) -> None:
    node = _open(_store(tmp_path))
    assert isinstance(node["img"], ZarrPythonArray)
    assert isinstance(node["img"], ZarrArray)
    assert isinstance(node["sub"], ZarrPythonGroup)


def test_native_is_the_backing_zarr_object(tmp_path: pathlib.Path) -> None:
    node = _open(_store(tmp_path))
    assert isinstance(node.native, zarr.Group)
    assert isinstance(node["img"].native, zarr.Array)


# --------------------------------------------------------------------------
# array surface
# --------------------------------------------------------------------------


def test_array_shape_dtype_chunks(tmp_path: pathlib.Path) -> None:
    arr = _open(_store(tmp_path))["img"]
    assert arr.shape == (8, 8)
    assert arr.ndim == 2
    assert arr.dtype == np.dtype("float32")
    assert arr.chunks == (4, 4)
    assert arr.shards is None
    assert arr.zarr_version == 3


def test_metadata_is_abczarr_metadata(tmp_path: pathlib.Path) -> None:
    arr = _open(_store(tmp_path))["img"]
    meta = arr.metadata
    assert isinstance(meta, ArrayMetadata)
    assert meta.shape == (8, 8)
    assert "v3:data_type:float32" in meta.required_features()


def test_read_returns_the_data(tmp_path: pathlib.Path) -> None:
    arr = _open(_store(tmp_path))["img"]
    assert np.asarray(arr[:2, :2]).tolist() == [[0.0, 1.0], [8.0, 9.0]]


def test_array_converts_to_numpy_and_dask(tmp_path: pathlib.Path) -> None:
    arr = _open(_store(tmp_path))["img"]
    assert np.asarray(arr).shape == (8, 8)
    # dask blocks align to the chunk (no shards here)
    assert arr.to_dask().chunksize == (4, 4)


# --------------------------------------------------------------------------
# write and create
# --------------------------------------------------------------------------


def test_write_through_the_surface(tmp_path: pathlib.Path) -> None:
    node = _open(_store(tmp_path), mode="a")
    arr = node["img"]
    arr[0, 0] = 999.0
    assert float(np.asarray(arr[0, 0])) == 999.0


def test_store_writes_a_dask_array_block_by_block(
    tmp_path: pathlib.Path,
) -> None:
    da = pytest.importorskip("dask.array")
    node = _open(_store(tmp_path), mode="a")
    arr = node["img"]
    source = da.zeros((8, 8), dtype="float32", chunks=(4, 4)) + 7.0
    arr.store(source)
    assert np.asarray(arr).tolist() == np.full((8, 8), 7.0).tolist()


def test_create_array_and_group(tmp_path: pathlib.Path) -> None:
    node = _open(_store(tmp_path), mode="a")
    made = node.create_array("new", shape=(3,), dtype="int16", chunks=(3,))
    assert isinstance(made, ZarrPythonArray)
    assert made.shape == (3,)
    node.create_group("grp2")
    assert sorted(node.keys()) == ["grp2", "img", "new", "sub"]


def test_delete_a_member(tmp_path: pathlib.Path) -> None:
    node = _open(_store(tmp_path), mode="a")
    del node["sub"]
    assert "sub" not in list(node.keys())


def test_attrs_write_through(tmp_path: pathlib.Path) -> None:
    root = _store(tmp_path)
    node = _open(root, "a")["img"]
    node.attrs["scale"] = 0.5
    # a freshly opened array reads the attribute back
    assert _open(root, "r")["img"].attrs["scale"] == 0.5
    del node.attrs["scale"]
    assert "scale" not in _open(root, "r")["img"].attrs
