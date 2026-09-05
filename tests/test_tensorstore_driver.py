"""The TensorStore driver: feature declaration and reading/writing v3 arrays.

Needs both zarr-python (to write the fixtures) and tensorstore; runs on the
coverage CI leg where both are installed.
"""

import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
pytest.importorskip("tensorstore")

import abczarr  # noqa: E402
from abczarr.abc.capabilities import Support  # noqa: E402
from abczarr.api.config import ArrayConfig  # noqa: E402
from abczarr.api.registry import available_drivers  # noqa: E402
from abczarr.drivers.tensorstore import (  # noqa: E402
    TensorStoreArray,
    TensorStoreDriver,
    TensorStoreGroup,
    TensorStoreNode,
)
from abczarr.errors import UnsupportedZarrOperation  # noqa: E402
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


def test_tensorstore_is_registered() -> None:
    names = [d.name for d in available_drivers()]
    assert "tensorstore" in names


def test_coarse_capabilities() -> None:
    d = TensorStoreDriver()
    assert d.available is True
    assert d.capability("sharding") is Support.NATIVE
    assert d.capability("async") is Support.NATIVE
    assert d.supports("partial_read") is True


def test_codec_support() -> None:
    d = TensorStoreDriver()
    assert d.capability("v3:codec:zstd") is Support.NATIVE
    assert d.capability("v3:codec:sharding_indexed") is Support.NATIVE
    assert d.capability("v3:codec:transpose") is Support.NATIVE
    # a codec tensorstore does not implement
    assert d.capability("v3:codec:packbits") is Support.NONE
    assert d.capability("v3:chunk_grid:rectilinear") is Support.NONE


# --------------------------------------------------------------------------
# reading and writing an array
# --------------------------------------------------------------------------


def test_open_read_a_v3_array(tmp_path: pathlib.Path) -> None:
    path = _array_path(
        tmp_path,
        compressors=[{"name": "zstd", "configuration": {"level": 5}}],
    )
    arr = abczarr.open_array(path, mode="r", driver="tensorstore")
    assert isinstance(arr, TensorStoreArray)
    assert arr.shape == (8, 8)
    assert arr.ndim == 2
    assert arr.dtype == np.dtype("float32")
    assert arr.chunks == (4, 4)
    assert arr.zarr_version == 3
    assert np.asarray(arr[:2, :2]).tolist() == [[0.0, 1.0], [8.0, 9.0]]


def test_metadata_is_abczarr_metadata(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="tensorstore"
    )
    meta = arr.metadata
    assert isinstance(meta, ArrayMetadata)
    assert meta.shape == (8, 8)
    assert "v3:data_type:float32" in meta.required_features()


def test_write_through_the_surface(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="a", driver="tensorstore"
    )
    arr[0, 0] = 99.0
    assert float(np.asarray(arr[0, 0])) == 99.0


def test_native_is_the_tensorstore(tmp_path: pathlib.Path) -> None:
    import tensorstore as ts

    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="tensorstore"
    )
    assert isinstance(arr.native, ts.TensorStore)


def test_sharded_array_reports_shards(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "s.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(2, 2), shards=(4, 4), dtype="uint8"
    )
    array[:] = 0
    arr = abczarr.open_array(root + "/img", mode="r", driver="tensorstore")
    assert arr.chunks == (2, 2)
    assert arr.shards == (4, 4)


def test_can_open_a_supported_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        _array_path(tmp_path), mode="r", driver="tensorstore"
    )
    assert bool(TensorStoreDriver().can_open(arr.metadata)) is True


# --------------------------------------------------------------------------
# groups: read from the store, arrays opened through TensorStore
# --------------------------------------------------------------------------


def test_opening_a_group_navigates_to_its_arrays(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "g.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "a", shape=(4,), chunks=(4,), dtype="int32"
    )
    array[:] = np.arange(4)
    group.create_group("sub")

    node = abczarr.open(root, mode="r", driver="tensorstore")
    assert isinstance(node, TensorStoreGroup)
    assert sorted(node.keys()) == ["a", "sub"]

    child = node["a"]
    assert isinstance(child, TensorStoreArray)
    assert np.asarray(child[:]).tolist() == [0, 1, 2, 3]


def test_a_subgroup_is_another_tensorstore_group(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "g.zarr")
    group = zarr.open_group(root, mode="w")
    group.create_group("sub").create_array(
        "inner", shape=(2,), chunks=(2,), dtype="u1"
    )

    node = abczarr.open(root, mode="r", driver="tensorstore")
    sub = node["sub"]
    assert isinstance(sub, TensorStoreGroup)
    assert list(sub.keys()) == ["inner"]
    assert isinstance(sub["inner"], TensorStoreArray)


def test_open_group_requires_a_group(tmp_path: pathlib.Path) -> None:
    # open_array on a group raises through the api's own check
    root = str(tmp_path / "g.zarr")
    zarr.open_group(root, mode="w").create_array(
        "a", shape=(2,), chunks=(2,), dtype="u1"
    )
    with pytest.raises(UnsupportedZarrOperation):
        abczarr.open_array(root, mode="r", driver="tensorstore")


# --------------------------------------------------------------------------
# creating arrays through TensorStore
# --------------------------------------------------------------------------


def test_create_a_top_level_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.create(
        str(tmp_path / "a.zarr"),
        ArrayConfig(
            shape=(8, 8), dtype="float32", chunks=(4, 4), driver="tensorstore"
        ),
    )
    assert isinstance(arr, TensorStoreArray)
    arr[:] = np.arange(64, dtype="float32").reshape(8, 8)
    assert float(np.asarray(arr[1, 0])) == 8.0


def test_create_an_array_inside_a_group(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g.zarr")
    zarr.open_group(root, mode="w")
    group = abczarr.open(root, mode="a", driver="tensorstore")
    assert isinstance(group, TensorStoreGroup)
    arr = group.create_array("img", shape=(4,), dtype="int32", chunks=(4,))
    assert isinstance(arr, TensorStoreArray)
    arr[:] = np.arange(4, dtype="int32")
    assert np.asarray(arr[:]).tolist() == [0, 1, 2, 3]
    assert "img" in list(group.keys())


def test_store_writes_a_dask_array(tmp_path: pathlib.Path) -> None:
    # da.to_zarr rejects a tensorstore, but store() works through da.store
    da = pytest.importorskip("dask.array")
    arr = abczarr.create(
        str(tmp_path / "d.zarr"),
        ArrayConfig(
            shape=(8, 8), dtype="float32", chunks=(4, 4), driver="tensorstore"
        ),
    )
    arr.store(da.zeros((8, 8), dtype="float32", chunks=(4, 4)) + 3.0)
    assert float(np.asarray(arr[0, 0])) == 3.0
    assert float(np.asarray(arr[7, 7])) == 3.0


def test_both_node_kinds_share_a_common_base(tmp_path: pathlib.Path) -> None:
    # array and group are both TensorStoreNodes, so open returns one type
    root = str(tmp_path / "g.zarr")
    zarr.open_group(root, mode="w").create_array(
        "a", shape=(2,), chunks=(2,), dtype="u1"
    )
    group = abczarr.open(root, mode="r", driver="tensorstore")
    array = abczarr.open(root + "/a", mode="r", driver="tensorstore")
    assert isinstance(group, TensorStoreNode)
    assert isinstance(array, TensorStoreNode)


def test_attrs_write_through(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g.zarr")
    zarr.open_group(root, mode="w").create_array(
        "a", shape=(2,), chunks=(2,), dtype="u1"
    )
    node = abczarr.open(root + "/a", mode="a", driver="tensorstore")
    node.attrs["unit"] = "micrometer"
    reopened = abczarr.open(root + "/a", mode="r", driver="tensorstore")
    assert reopened.attrs["unit"] == "micrometer"


# --------------------------------------------------------------------------
# a group at a URL is detected as a group -- the metadata peek reads
# zarr.json through a (sync/async) PathBasedStore, so every scheme works,
# not only local paths
# --------------------------------------------------------------------------


def test_group_at_a_memory_url_opens_as_a_group() -> None:
    # a tensorstore group at an fsspec memory:// URL is detected as a group,
    # not mistaken for an array -- the sync peek reads through PathBasedStore
    import uuid

    pytest.importorskip("fsspec")
    url = "memory://" + uuid.uuid4().hex + "/g.zarr"
    group = zarr.open_group(url, mode="w")
    array = group.create_array("a", shape=(4,), chunks=(4,), dtype="int32")
    array[:] = np.arange(4)

    node = abczarr.open(url, mode="r", driver="tensorstore")
    assert isinstance(node, TensorStoreGroup)
    assert list(node.keys()) == ["a"]


def test_group_at_a_memory_url_opens_as_a_group_async() -> None:
    # the async peek reads through AsyncPathBasedStore, so an async open of a
    # group at a memory:// URL lands on the async path group, not an array
    import asyncio
    import uuid

    pytest.importorskip("fsspec")
    from abczarr.abc.asynchronous import AsyncPathGroup

    url = "memory://" + uuid.uuid4().hex + "/g.zarr"
    group = zarr.open_group(url, mode="w")
    group.create_array("a", shape=(4,), chunks=(4,), dtype="int32")

    async def go() -> object:
        return await abczarr.open(
            url, mode="r", asynchronous=True, driver="tensorstore"
        )

    node = asyncio.run(go())
    assert isinstance(node, AsyncPathGroup)


def test_group_via_a_pathlike_is_detected(tmp_path: pathlib.Path) -> None:
    # a location need not be a str -- a PathLike (pathlib.Path) is peeked the
    # same way, so a group opened through a Path is detected as a group
    root = tmp_path / "g.zarr"
    group = zarr.open_group(str(root), mode="w")
    group.create_array("a", shape=(4,), chunks=(4,), dtype="int32")

    node = abczarr.open(root, mode="r", driver="tensorstore")  # Path, not str
    assert isinstance(node, TensorStoreGroup)
    assert list(node.keys()) == ["a"]


# --------------------------------------------------------------------------
# a Zarr v2 group -- its metadata peek reads .zgroup, not only zarr.json, so
# creating and re-opening a v2 group through TensorStore succeeds instead of
# being mistaken for a (missing) v3 array
# --------------------------------------------------------------------------


def test_create_a_v2_group(tmp_path: pathlib.Path) -> None:
    # the reported bug: writing the .zgroup and re-opening it through the
    # driver would peek only zarr.json, mistake the group for a v3 array, and
    # raise an opaque backend NOT_FOUND. The peek now sees the v2 metadata.
    root = str(tmp_path / "g2.zarr")
    node = abczarr.create_group(root, zarr_version=2, driver="tensorstore")
    assert isinstance(node, TensorStoreGroup)
    assert node.zarr_version == 2


def test_a_v2_group_reopens_as_a_v2_group(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g2.zarr")
    abczarr.create_group(root, zarr_version=2, driver="tensorstore")
    reopened = abczarr.open(root, mode="r", driver="tensorstore")
    assert isinstance(reopened, TensorStoreGroup)
    assert reopened.zarr_version == 2


# --------------------------------------------------------------------------
# a Zarr v2 array -- opened and created through TensorStore's native ``zarr``
# driver (v3 goes through ``zarr3``), so a v2 array reads and writes rather
# than raising an opaque backend error
# --------------------------------------------------------------------------


def test_create_write_read_a_v2_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.create(
        str(tmp_path / "a2.zarr"),
        ArrayConfig(
            shape=(8,), dtype="int32", chunks=(4,), zarr_version=2,
            driver="tensorstore",
        ),
    )
    assert isinstance(arr, TensorStoreArray)
    assert arr.zarr_version == 2
    assert arr.dtype == np.dtype("int32")
    assert arr.chunks == (4,)
    arr[:4] = np.arange(4, dtype="int32")
    assert np.asarray(arr[:4]).tolist() == [0, 1, 2, 3]


def test_a_v2_array_reopens_as_a_v2_array(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "a2.zarr")
    arr = abczarr.create(
        root,
        ArrayConfig(
            shape=(4,), dtype="int16", chunks=(4,), zarr_version=2,
            driver="tensorstore",
        ),
    )
    arr[:] = np.arange(4, dtype="int16")

    reopened = abczarr.open(root, mode="r", driver="tensorstore")
    assert isinstance(reopened, TensorStoreArray)
    assert reopened.zarr_version == 2
    assert reopened.dtype == np.dtype("int16")
    assert np.asarray(reopened[:]).tolist() == [0, 1, 2, 3]


def test_a_v2_array_round_trips_dtype_compressor_and_fill(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "a2.zarr")
    arr = abczarr.create(
        root,
        ArrayConfig(
            shape=(8,), dtype="int32", chunks=(4,), zarr_version=2,
            fill_value=7, driver="tensorstore",
        ),
    )
    arr[:4] = np.arange(4, dtype="int32")
    reopened = abczarr.open(root, mode="r", driver="tensorstore")
    meta = reopened.metadata.to_json()
    assert meta["dtype"] == "<i4"
    assert meta["fill_value"] == 7
    assert meta["compressor"]["id"] == "zstd"
    # the fill value shows in the untouched half
    assert np.asarray(reopened[:]).tolist() == [0, 1, 2, 3, 7, 7, 7, 7]


def test_a_v2_array_attributes_round_trip(tmp_path: pathlib.Path) -> None:
    # TensorStore's zarr driver writes only .zarray, so a v2 array's user
    # attributes are persisted to .zattrs and read back from there
    root = str(tmp_path / "a2.zarr")
    arr = abczarr.create(
        root,
        ArrayConfig(
            shape=(4,), dtype="int32", chunks=(4,), zarr_version=2,
            attributes={"unit": "micrometer"}, driver="tensorstore",
        ),
    )
    assert dict(arr.attrs) == {"unit": "micrometer"}
    reopened = abczarr.open(root, mode="r", driver="tensorstore")
    assert dict(reopened.attrs) == {"unit": "micrometer"}


def test_reads_a_blosc_compressed_v2_array(tmp_path: pathlib.Path) -> None:
    # a v2 array written by zarr-python with a blosc compressor reads through
    # TensorStore's zarr driver
    numcodecs = pytest.importorskip("numcodecs")
    root = str(tmp_path / "g.zarr")
    group = zarr.open_group(root, mode="w", zarr_format=2)
    array = group.create_array(
        "img", shape=(4,), chunks=(4,), dtype="uint8",
        compressors=numcodecs.Blosc(cname="lz4", clevel=5),
    )
    array[:] = np.arange(4)

    node = abczarr.open(root + "/img", mode="r", driver="tensorstore")
    assert isinstance(node, TensorStoreArray)
    assert node.zarr_version == 2
    assert np.asarray(node[:]).tolist() == [0, 1, 2, 3]
    assert node.metadata.to_json()["compressor"]["id"] == "blosc"
