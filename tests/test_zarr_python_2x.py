"""The zarr-python driver running against a zarr-python 2.x install.

These tests exercise the library, not just the v2 on-disk format: they run
only where the installed ``zarr`` is a 2.x release. A 2.x install has no
native async surface, no sharding, and no ``create_array`` or ``metadata``
accessor, so the driver adapts the open, create, metadata, and async paths.
The v2-format-through-zarr-3 path is covered separately by
``test_zarr_python_v2``.
"""

import asyncio
import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

if not zarr.__version__.startswith("2."):
    pytest.skip("requires a zarr-python 2.x install", allow_module_level=True)

import abczarr  # noqa: E402
from abczarr.abc.asynchronous import (  # noqa: E402
    AsyncZarrArray,
    AsyncZarrGroup,
)
from abczarr.abc.capabilities import Support  # noqa: E402
from abczarr.drivers.zarr_python import (  # noqa: E402
    ZarrPythonArray,
    ZarrPythonDriver,
    ZarrPythonGroup,
)
from abczarr.metadata.base import ArrayMetadata  # noqa: E402


def _store(tmp_path: pathlib.Path) -> str:
    """A real zarr 2 store with an array and a subgroup."""
    root = str(tmp_path / "data.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_dataset(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32"
    )
    array[:] = np.arange(64).reshape(8, 8)
    array.attrs["unit"] = "um"
    group.create_group("sub")
    return root


def test_driver_is_available_on_zarr_2() -> None:
    driver = ZarrPythonDriver()
    assert driver.available
    assert driver._major == 2


def test_capabilities_reflect_zarr_2() -> None:
    driver = ZarrPythonDriver()
    # zarr 2 synthesizes async and has no sharding or v3 codecs.
    assert driver.capability("async") is Support.SYNTHESIZED
    assert driver.capability("sharding") is Support.NONE
    assert driver.capability("codecs_v2") is Support.NATIVE
    # a numcodecs compressor is provided; a v3 feature is not.
    assert driver.capability("v2:codec:blosc") is Support.NATIVE
    assert driver.capability("v3:codec:zstd") is Support.NONE


def test_open_read_and_navigate(tmp_path: pathlib.Path) -> None:
    node = abczarr.open(_store(tmp_path), mode="r")
    assert isinstance(node, ZarrPythonGroup)
    assert sorted(node.keys()) == ["img", "sub"]
    array = node["img"]
    assert isinstance(array, ZarrPythonArray)
    assert array.shape == (8, 8)
    assert array.chunks == (4, 4)
    assert array.shards is None
    assert array.zarr_version == 2
    assert np.asarray(array[:2, :2]).tolist() == [[0.0, 1.0], [8.0, 9.0]]


def test_metadata_and_attrs(tmp_path: pathlib.Path) -> None:
    array = abczarr.open(_store(tmp_path), mode="r")["img"]
    meta = array.metadata
    assert isinstance(meta, ArrayMetadata)
    assert meta.shape == (8, 8)
    assert array.attrs["unit"] == "um"


def test_attr_write_through(tmp_path: pathlib.Path) -> None:
    root = _store(tmp_path)
    node = abczarr.open(root, mode="a")["img"]
    node.attrs["scale"] = 0.5
    reopened = abczarr.open(root, mode="r")["img"]
    assert reopened.attrs["scale"] == 0.5
    assert reopened.attrs["unit"] == "um"
    del node.attrs["scale"]
    assert "scale" not in abczarr.open(root, mode="r")["img"].attrs


def test_node_reports_synthesized_async(tmp_path: pathlib.Path) -> None:
    array = abczarr.open(_store(tmp_path), mode="r")["img"]
    assert array.supports("async")
    assert not array.supports("async", native=True)


def test_create_writes_v2_format(tmp_path: pathlib.Path) -> None:
    # The default config asks for zarr_version 3, which a zarr 2 backend
    # cannot write, so the array is created in the v2 format instead.
    root = str(tmp_path / "new.zarr")
    array = abczarr.open(root, mode="w", shape=(4, 4), dtype="int16")
    assert isinstance(array, ZarrPythonArray)
    assert array.shape == (4, 4)
    assert array.zarr_version == 2
    import json

    with open(root + "/.zarray") as handle:
        assert json.load(handle)["zarr_format"] == 2


def test_create_group_and_nested_members(tmp_path: pathlib.Path) -> None:
    group = abczarr.open(str(tmp_path / "g.zarr"), mode="w")
    assert isinstance(group, ZarrPythonGroup)
    assert group.zarr_version == 2
    child = group.create_array("c", shape=(3,), dtype="u1", chunks=(3,))
    group.create_group("sg")
    assert child.zarr_version == 2
    assert sorted(group.keys()) == ["c", "sg"]


def test_async_open_and_read(tmp_path: pathlib.Path) -> None:
    root = _store(tmp_path)

    async def run() -> object:
        node = await abczarr.open(root, mode="r", asynchronous=True)
        assert isinstance(node, AsyncZarrGroup)
        array = await node.getitem("img")
        assert isinstance(array, AsyncZarrArray)
        return await array.getitem((slice(0, 2), slice(0, 2)))

    block = asyncio.run(run())
    assert np.asarray(block).tolist() == [[0.0, 1.0], [8.0, 9.0]]


def test_async_create(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "mk.zarr")

    async def run() -> object:
        array = await abczarr.open(
            root, mode="w", shape=(3,), dtype="u1", asynchronous=True
        )
        assert isinstance(array, AsyncZarrArray)
        await array.setitem(Ellipsis, np.array([1, 2, 3], dtype="u1"))
        return await array.getitem(Ellipsis)

    assert np.asarray(asyncio.run(run())).tolist() == [1, 2, 3]
