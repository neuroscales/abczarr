"""The async node surface: the coroutine twins of arrays and groups.

Three things are checked:

* **parity** -- the sync and async node classes expose matching surfaces, so
  the two colors cannot silently drift (runs with no backend installed);
* **native round-trips** -- tensorstore and zarr-python read and write and
  fan out concurrently through their own coroutine machinery;
* **honest capability reporting** -- `"async"` is `NATIVE` for a backend with
  a real coroutine surface and `SYNTHESIZED` for the thread-pool default.

pytest-asyncio is not a dependency, so each coroutine is driven with
``asyncio.run`` from a plain synchronous test.
"""

import asyncio
import inspect
import pathlib

import numpy as np
import pytest

import abczarr
from abczarr.abc.array import ZarrArray
from abczarr.abc.asyncnode import (
    AsyncZarrArray,
    AsyncZarrGroup,
    ThreadedAsyncArray,
    ThreadedAsyncGroup,
)
from abczarr.abc.capabilities import Support

# --------------------------------------------------------------------------
# parity: the sync and async surfaces stay in lockstep (no backend needed)
# --------------------------------------------------------------------------

#: Read-only properties that never block, so they are the same, synchronous,
#: on both colors.
_SHARED_PROPERTIES = {
    "store_path", "native", "metadata", "attrs", "zarr_version",
}
_ARRAY_PROPERTIES = _SHARED_PROPERTIES | {
    "ndim", "shape", "dtype", "chunks", "shards",
}


def _is_property(cls: type, name: str) -> bool:
    return isinstance(getattr(cls, name, None), property)


def test_array_read_only_properties_match() -> None:
    # every non-blocking property on the sync array is a property on the
    # async twin too, and vice versa
    for name in _ARRAY_PROPERTIES:
        assert _is_property(ZarrArray, name), name
        assert _is_property(AsyncZarrArray, name), name


def test_array_io_is_methods_not_dunders() -> None:
    # the async twin trades __getitem__/__setitem__ for awaitable
    # getitem/setitem -- an assignment expression cannot be awaited
    assert inspect.iscoroutinefunction(AsyncZarrArray.getitem)
    assert inspect.iscoroutinefunction(AsyncZarrArray.setitem)
    assert "__setitem__" not in AsyncZarrArray.__dict__
    # the sync array keeps the dunder surface
    assert callable(ZarrArray.__getitem__)
    assert callable(ZarrArray.__setitem__)


def test_group_surface_parity() -> None:
    # the sync group's members each have an async counterpart
    assert _is_property(AsyncZarrGroup, "zarr_version")
    assert inspect.iscoroutinefunction(AsyncZarrGroup.getitem)
    assert inspect.iscoroutinefunction(AsyncZarrGroup.create_array)
    assert inspect.iscoroutinefunction(AsyncZarrGroup.create_group)
    # members are iterated asynchronously: keys is abstract on the base and
    # an async generator on a concrete twin
    assert hasattr(AsyncZarrGroup, "keys")
    assert hasattr(AsyncZarrGroup, "__aiter__")
    assert inspect.isasyncgenfunction(ThreadedAsyncGroup.keys)


def test_shared_property_names_are_identical() -> None:
    # the shared, non-blocking surface is the same on both node colors
    sync = {n for n in _SHARED_PROPERTIES if _is_property(abczarr.ZarrNode, n)}
    asyncd = {
        n for n in _SHARED_PROPERTIES if _is_property(abczarr.AsyncZarrNode, n)
    }
    assert sync == asyncd == _SHARED_PROPERTIES


# --------------------------------------------------------------------------
# zarr-python: a native coroutine surface
# --------------------------------------------------------------------------


def _zp_array(tmp_path: pathlib.Path) -> str:
    zarr = pytest.importorskip("zarr")
    root = str(tmp_path / "zp.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32"
    )
    array[:] = np.arange(64).reshape(8, 8)
    return root


def test_zarr_python_async_roundtrip(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    from abczarr.drivers.zarr_python import AsyncZarrPythonArray

    root = _zp_array(tmp_path)
    arr = abczarr.open(root + "/img", mode="a", asynchronous=True)
    assert isinstance(arr, AsyncZarrPythonArray)
    assert arr.supports("async", native=True)
    assert arr.shape == (8, 8)

    async def go() -> float:
        await arr.setitem(
            (slice(0, 4), slice(0, 4)), np.ones((4, 4), "float32")
        )
        block = await arr.getitem((slice(0, 4), slice(0, 4)))
        return float(np.asarray(block).sum())

    assert asyncio.run(go()) == 16.0


def test_zarr_python_concurrent_fanout(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    root = _zp_array(tmp_path)
    arr = abczarr.open(root + "/img", mode="r", asynchronous=True)
    regions = [
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(4, 8)),
        (slice(4, 8), slice(0, 4)),
        (slice(4, 8), slice(4, 8)),
    ]

    async def go() -> list:
        return await asyncio.gather(*(arr.getitem(r) for r in regions))

    blocks = asyncio.run(go())
    assert len(blocks) == 4
    total = sum(np.asarray(b).sum() for b in blocks)
    assert total == np.arange(64).sum()


def test_zarr_python_async_group(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    from abczarr.drivers.zarr_python import (
        AsyncZarrPythonArray,
        AsyncZarrPythonGroup,
    )

    root = _zp_array(tmp_path)
    grp = abczarr.open(root, mode="a", asynchronous=True)
    assert isinstance(grp, AsyncZarrPythonGroup)
    assert grp.supports("async", native=True)

    async def go() -> None:
        names = [name async for name in grp]
        assert "img" in names

        child = await grp.getitem("img")
        assert isinstance(child, AsyncZarrPythonArray)

        made = await grp.create_array("made", (4, 4), "uint8")
        assert isinstance(made, AsyncZarrPythonArray)
        await made.setitem(
            (slice(0, 4), slice(0, 4)), np.full((4, 4), 7, "uint8")
        )
        got = await made.getitem((slice(0, 4), slice(0, 4)))
        assert int(np.asarray(got)[0, 0]) == 7

        sub = await grp.create_group("sub")
        assert isinstance(sub, AsyncZarrPythonGroup)

    asyncio.run(go())


def test_zarr_python_color_conversion(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    root = _zp_array(tmp_path)
    sync = abczarr.open(root + "/img", mode="r")
    assert isinstance(sync, ZarrArray)
    async_arr = sync.as_async()
    assert isinstance(async_arr, AsyncZarrArray)
    # as_sync returns the same backend handle we started from
    assert async_arr.as_sync() is sync
    assert async_arr.native is sync.native


# --------------------------------------------------------------------------
# tensorstore: a native coroutine surface (awaits its own futures)
# --------------------------------------------------------------------------


def _ts_array(tmp_path: pathlib.Path) -> str:
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("tensorstore")
    root = str(tmp_path / "ts.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32"
    )
    array[:] = np.arange(64).reshape(8, 8)
    return root


def test_tensorstore_async_roundtrip(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("tensorstore")
    from abczarr.drivers.tensorstore import AsyncTensorStoreArray

    root = _ts_array(tmp_path)
    arr = abczarr.open(
        root + "/img", mode="a", asynchronous=True, driver="tensorstore"
    )
    assert isinstance(arr, AsyncTensorStoreArray)
    assert arr.supports("async", native=True)
    assert arr.shape == (8, 8)

    corner = np.arange(64).reshape(8, 8)[0:4, 0:4].sum()

    async def go() -> None:
        block = await arr.getitem((slice(0, 4), slice(0, 4)))
        assert np.asarray(block).sum() == corner
        await arr.setitem(
            (slice(0, 4), slice(0, 4)), np.zeros((4, 4), "float32")
        )
        again = await arr.getitem((slice(0, 4), slice(0, 4)))
        assert np.asarray(again).sum() == 0.0

    asyncio.run(go())


def test_tensorstore_concurrent_fanout(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("tensorstore")
    root = _ts_array(tmp_path)
    arr = abczarr.open(
        root + "/img", mode="r", asynchronous=True, driver="tensorstore"
    )
    regions = [
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(4, 8)),
        (slice(4, 8), slice(0, 4)),
        (slice(4, 8), slice(4, 8)),
    ]

    async def go() -> list:
        return await asyncio.gather(*(arr.getitem(r) for r in regions))

    blocks = asyncio.run(go())
    total = sum(np.asarray(b).sum() for b in blocks)
    assert total == np.arange(64).sum()


def test_tensorstore_group_children_are_native(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("tensorstore")
    from abczarr.drivers.tensorstore import AsyncTensorStoreArray

    root = _ts_array(tmp_path)
    grp = abczarr.open(
        root, mode="r", asynchronous=True, driver="tensorstore"
    )
    # tensorstore has no group object, so the group is thread-synthesized...
    assert isinstance(grp, ThreadedAsyncGroup)
    assert grp.capability("async") is Support.SYNTHESIZED

    # ...but a child array it opens comes back in tensorstore's native color
    async def go() -> object:
        return await grp.getitem("img")

    child = asyncio.run(go())
    assert isinstance(child, AsyncTensorStoreArray)
    assert child.supports("async", native=True)


# --------------------------------------------------------------------------
# the thread-synthesized default, reported honestly
# --------------------------------------------------------------------------


def test_thread_synth_default_reports_synthesized(
    tmp_path: pathlib.Path,
) -> None:
    # zarrista ships no coroutine surface abczarr wires up, so it falls to
    # the thread-pool default -- and says so
    pytest.importorskip("zarrista")
    zarr = pytest.importorskip("zarr")
    root = str(tmp_path / "zr.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32"
    )
    array[:] = np.arange(64).reshape(8, 8)

    arr = abczarr.open(
        root + "/img", mode="r", asynchronous=True, driver="zarrista"
    )
    assert isinstance(arr, ThreadedAsyncArray)
    assert arr.capability("async") is Support.SYNTHESIZED
    assert not arr.supports("async", native=True)
    assert arr.supports("async")  # synthesized still counts as supported
    corner = np.arange(64).reshape(8, 8)[0:4, 0:4].sum()

    async def go() -> object:
        return await arr.getitem((slice(0, 4), slice(0, 4)))

    block = asyncio.run(go())
    assert np.asarray(block).sum() == corner


# --------------------------------------------------------------------------
# the async helpers underneath the thread-synth default
# --------------------------------------------------------------------------


def test_run_sync_runs_in_a_worker_thread() -> None:
    import threading

    from abczarr._core.asyncutils import run_sync

    def where() -> str:
        return threading.current_thread().name

    async def go() -> str:
        return await run_sync(where)

    name = asyncio.run(go())
    # it ran on the dedicated pool, not the main thread
    assert name.startswith("abczarr-async")


def test_concurrent_map_caps_the_fan_out() -> None:
    from abczarr._core.asyncutils import concurrent_map

    live = 0
    peak = 0

    async def task(n: int) -> int:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return n

    async def go() -> list:
        return await concurrent_map(
            [(i,) for i in range(20)], task, limit=3
        )

    result = asyncio.run(go())
    assert result == list(range(20))
    # the semaphore held the in-flight count to the limit
    assert peak <= 3
