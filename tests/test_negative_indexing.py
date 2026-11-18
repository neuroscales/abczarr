"""NumPy-style indexing, negative positions included, on every driver.

Each read and write is compared against the same operation on a NumPy
array. A driver whose backend is not installed is skipped.
"""

import asyncio
import importlib.util
import pathlib

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import abczarr  # noqa: E402

SHAPE = (4, 5)

INDICES = [
    pytest.param(-1, id="-1"),
    pytest.param(slice(-2, None), id="-2:"),
    pytest.param(slice(None, -1), id=":-1"),
    pytest.param(slice(None, None, -1), id="::-1"),
    pytest.param(slice(-3, -1), id="-3:-1"),
    pytest.param((Ellipsis, -1), id="...,-1"),
    pytest.param(np.array([-1, 0]), id="[-1,0]"),
    pytest.param((slice(None), [-1, 0]), id=":,[-1,0]"),
    pytest.param((slice(None, None, -2), -2), id="::-2,-2"),
    pytest.param(np.array([True, False, True, False]), id="mask"),
]


def _driver(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not installed")
    return name


@pytest.fixture(params=["tensorstore", "zarrista"])
def driver(request: pytest.FixtureRequest) -> str:
    return _driver(request.param)


def _make(tmp_path: pathlib.Path) -> "tuple[str, np.ndarray]":
    path = str(tmp_path / "a.zarr")
    ref = np.arange(20, dtype="float32").reshape(SHAPE)
    array = zarr.create_array(
        path, shape=SHAPE, chunks=(2, 2), dtype="float32"
    )
    array[:] = ref
    return path, ref


@pytest.mark.parametrize("index", INDICES)
def test_read_matches_numpy(
    tmp_path: pathlib.Path, driver: str, index: object
) -> None:
    path, ref = _make(tmp_path)
    arr = abczarr.open_array(path, mode="r", driver=driver)
    np.testing.assert_array_equal(np.asarray(arr[index]), ref[index])
    assert np.asarray(arr[index]).shape == ref[index].shape


@pytest.mark.parametrize("index", INDICES)
def test_write_matches_numpy(
    tmp_path: pathlib.Path, driver: str, index: object
) -> None:
    path, ref = _make(tmp_path)
    arr = abczarr.open_array(path, mode="a", driver=driver)
    value = -1 - np.arange(ref[index].size, dtype="float32")
    value = value.reshape(ref[index].shape)
    arr[index] = value
    ref[index] = value
    np.testing.assert_array_equal(np.asarray(arr[...]), ref)


@pytest.mark.parametrize("index", [-5, 4, (0, -6), np.array([-5])])
def test_out_of_bounds_raises(
    tmp_path: pathlib.Path, driver: str, index: object
) -> None:
    path, _ = _make(tmp_path)
    arr = abczarr.open_array(path, mode="a", driver=driver)
    with pytest.raises(IndexError):
        arr[index]
    with pytest.raises(IndexError):
        arr[index] = 0


@pytest.mark.parametrize("index", INDICES)
def test_tensorstore_async_matches_numpy(
    tmp_path: pathlib.Path, index: object
) -> None:
    driver = _driver("tensorstore")
    path, ref = _make(tmp_path)
    arr = abczarr.open_array(path, mode="a", driver=driver).as_async()

    async def go() -> np.ndarray:
        got = await arr.getitem(index)
        await arr.setitem(index, np.zeros_like(ref[index]))
        return np.asarray(got)

    got = asyncio.run(go())
    np.testing.assert_array_equal(got, ref[index])
    ref[index] = 0
    reopened = abczarr.open_array(path, mode="r", driver=driver)
    np.testing.assert_array_equal(np.asarray(reopened[...]), ref)
