"""The zarrista driver: feature declaration and reading/writing v3 arrays.

Needs zarr-python (to write the fixtures) and zarrista; runs on the coverage
CI leg where both are installed.
"""

import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("zarrista")

import abczarr  # noqa: E402
from abczarr.abc.capabilities import Support  # noqa: E402
from abczarr.api.registry import available_drivers  # noqa: E402
from abczarr.drivers.zarrista import (  # noqa: E402
    ZarristaArray,
    ZarristaDriver,
    ZarristaGroup,
    ZarristaNode,
)
from abczarr.metadata.base import ArrayMetadata  # noqa: E402


def _array_path(tmp_path: pathlib.Path, **create: object) -> str:
    root = str(tmp_path / "d.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32", **create
    )
    array[:] = np.arange(64).reshape(8, 8)
    return root + "/img"


# --------------------------------------------------------------------------
# registration and features
# --------------------------------------------------------------------------


def test_zarrista_is_registered() -> None:
    assert "zarrista" in [d.name for d in available_drivers()]


def test_capabilities() -> None:
    d = ZarristaDriver()
    assert d.available is True
    assert d.capability("sharding") is Support.NATIVE
    assert d.capability("v3:codec:zstd") is Support.NATIVE
    assert d.capability("v3:codec:blosc") is Support.NATIVE
    # a codec zarrista does not implement
    assert d.capability("v3:codec:packbits") is Support.NONE
    assert d.capability("v2:codec:zstd") is Support.NONE


# --------------------------------------------------------------------------
# reading and writing an array
# --------------------------------------------------------------------------


def test_open_read_a_v3_array(tmp_path: pathlib.Path) -> None:
    path = _array_path(
        tmp_path, compressors=[{"name": "zstd", "configuration": {"level": 5}}]
    )
    arr = abczarr.open_array(path, mode="r", driver="zarrista")
    assert isinstance(arr, ZarristaArray)
    assert arr.shape == (8, 8)
    assert arr.ndim == 2
    assert arr.dtype == np.dtype("float32")
    assert arr.chunks == (4, 4)
    assert arr.shards is None
    assert arr.zarr_version == 3
    assert np.asarray(arr[0:2, 0:2]).tolist() == [[0.0, 1.0], [8.0, 9.0]]


def test_metadata_is_abczarr_metadata(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="zarrista"
    )
    meta = arr.metadata
    assert isinstance(meta, ArrayMetadata)
    assert meta.shape == (8, 8)
    assert "v3:data_type:float32" in meta.required_features()


def test_write_through_the_surface(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="a", driver="zarrista"
    )
    arr[0:1, 0:1] = np.array([[99.0]], dtype="float32")
    assert np.asarray(arr[0:1, 0:1]).tolist() == [[99.0]]


def test_native_is_the_zarrista_array(tmp_path: pathlib.Path) -> None:
    import zarrista

    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="zarrista"
    )
    assert isinstance(arr.native, zarrista.Array)


def test_sharded_array_reports_shards(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "s.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(2, 2), shards=(4, 4), dtype="uint8"
    )
    array[:] = 0
    arr = abczarr.open_array(root + "/img", mode="r", driver="zarrista")
    assert arr.chunks == (2, 2)
    assert arr.shards == (4, 4)


# --------------------------------------------------------------------------
# groups: read from the store, arrays opened through zarrista
# --------------------------------------------------------------------------


def test_opening_a_group_navigates_to_its_arrays(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "g.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array("a", shape=(4,), chunks=(4,), dtype="int32")
    array[:] = np.arange(4)
    group.create_group("sub")

    node = abczarr.open(root, mode="r", driver="zarrista")
    assert isinstance(node, ZarristaGroup)
    assert isinstance(node, ZarristaNode)
    assert sorted(node.keys()) == ["a", "sub"]

    child = node["a"]
    assert isinstance(child, ZarristaArray)
    assert np.asarray(child[0:4]).tolist() == [0, 1, 2, 3]


def test_a_subgroup_is_another_zarrista_group(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g.zarr")
    group = zarr.open_group(root, mode="w")
    group.create_group("sub").create_array(
        "inner", shape=(2,), chunks=(2,), dtype="u1"
    )
    node = abczarr.open(root, mode="r", driver="zarrista")
    sub = node["sub"]
    assert isinstance(sub, ZarristaGroup)
    assert list(sub.keys()) == ["inner"]
    assert isinstance(sub["inner"], ZarristaArray)


def test_attrs_write_through(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g.zarr")
    zarr.open_group(root, mode="w").create_array(
        "a", shape=(2,), chunks=(2,), dtype="u1"
    )
    node = abczarr.open(root + "/a", mode="a", driver="zarrista")
    node.attrs["unit"] = "micrometer"
    reopened = abczarr.open(root + "/a", mode="r", driver="zarrista")
    assert reopened.attrs["unit"] == "micrometer"
