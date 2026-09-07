"""Creating from array data, and writing OME metadata on create.

[create][abczarr.api.create] accepts an array as *data*, taking the new
array's shape and dtype from it unless a config or a keyword overrides them,
and writing the data into the array. It also accepts *ome*, which writes
OME-Zarr metadata on the new node. [create_group][abczarr.api.create_group]
and [create_array][abczarr.api.create_array] pass *ome* through to
[create][abczarr.api.create]. These tests pin all of that, sync and async.
"""

import asyncio
import pathlib

import numpy as np
import pytest

import abczarr
from abczarr.ome import ImageConfig

# a real backend is needed to create and read arrays
pytest.importorskip("zarr")


def _read(node: object) -> "np.ndarray":
    return np.asarray(node[:, :])


# --- create from data ------------------------------------------------------


def test_create_takes_shape_and_dtype_from_data(
    tmp_path: pathlib.Path,
) -> None:
    data = np.arange(16, dtype="int16").reshape(4, 4)
    arr = abczarr.create(str(tmp_path / "a.zarr"), data=data)
    assert arr.shape == (4, 4)
    assert arr.dtype == np.dtype("int16")
    assert np.array_equal(_read(arr), data)


def test_a_dtype_keyword_overrides_the_data_dtype(
    tmp_path: pathlib.Path,
) -> None:
    arr = abczarr.create(
        str(tmp_path / "a.zarr"),
        data=np.ones((3, 3), "float64"),
        dtype="uint8",
    )
    assert arr.dtype == np.dtype("uint8")


def test_a_config_shape_overrides_the_data_shape(
    tmp_path: pathlib.Path,
) -> None:
    # the config's shape wins; the data fills the whole array here
    config = abczarr.ArrayConfig(shape=(2, 2))
    arr = abczarr.create(
        str(tmp_path / "a.zarr"), config, data=np.zeros((2, 2), "i1")
    )
    assert arr.shape == (2, 2)


def test_a_nested_list_is_turned_into_an_array(
    tmp_path: pathlib.Path,
) -> None:
    arr = abczarr.create(str(tmp_path / "a.zarr"), data=[[1, 2], [3, 4]])
    assert arr.shape == (2, 2)
    assert np.array_equal(_read(arr), [[1, 2], [3, 4]])


def test_create_with_data_and_a_group_config_is_an_error(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(TypeError, match="creates an array"):
        abczarr.create(
            str(tmp_path / "a.zarr"),
            abczarr.GroupConfig(),
            data=np.zeros((2, 2)),
        )


def test_create_with_neither_config_nor_data_is_an_error(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(TypeError):
        abczarr.create(str(tmp_path / "a.zarr"))


# --- writing OME metadata on create ----------------------------------------


def test_create_group_writes_ome_from_an_image_config(
    tmp_path: pathlib.Path,
) -> None:
    ome = ImageConfig(axes=["y", "x"], scale=[2.0, 2.0])
    group = abczarr.create(
        str(tmp_path / "g.zarr"), abczarr.GroupConfig(), ome=ome
    )
    assert group.ome is not None
    assert group.ome.version == "0.5"
    axes = [a.name for a in group.ome.multiscales[0].axes]
    assert axes == ["y", "x"]


def test_create_writes_ome_from_a_typed_object(
    tmp_path: pathlib.Path,
) -> None:
    ome = ImageConfig(axes=["y", "x"]).to_ome(version="0.5")
    group = abczarr.create(
        str(tmp_path / "g.zarr"), abczarr.GroupConfig(), ome=ome
    )
    assert group.ome is not None
    assert group.ome.version == "0.5"


def test_create_group_helper_passes_ome_through(
    tmp_path: pathlib.Path,
) -> None:
    group = abczarr.create_group(
        str(tmp_path / "g.zarr"), ome=ImageConfig(axes=["y", "x"])
    )
    assert group.ome is not None


# --- async -----------------------------------------------------------------


def test_async_create_stores_data(tmp_path: pathlib.Path) -> None:
    data = np.arange(9, dtype="int16").reshape(3, 3)

    async def go() -> object:
        return await abczarr.create(
            str(tmp_path / "a.zarr"), data=data, asynchronous=True
        )

    arr = asyncio.run(go())
    assert arr.shape == (3, 3)
    reopened = abczarr.open(str(tmp_path / "a.zarr"), mode="r")
    assert np.array_equal(_read(reopened), data)


def test_async_create_writes_ome(tmp_path: pathlib.Path) -> None:
    async def go() -> object:
        return await abczarr.create(
            str(tmp_path / "g.zarr"),
            abczarr.GroupConfig(),
            ome=ImageConfig(axes=["y", "x"]),
            asynchronous=True,
        )

    group = asyncio.run(go())
    assert group.ome is not None
    assert group.ome.version == "0.5"
