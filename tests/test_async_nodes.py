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
from abczarr.abc.async_array import AsyncZarrArray, ThreadedAsyncArray
from abczarr.abc.async_group import (
    AsyncPathGroup,
    AsyncZarrGroup,
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


def test_driver_open_has_a_coroutine_twin() -> None:
    # every driver's sync open has an async twin: open is a plain method,
    # open_async is a coroutine, and the concrete drivers keep both
    from abczarr.drivers.base import Driver
    from abczarr.drivers.tensorstore import TensorStoreDriver
    from abczarr.drivers.zarr_python import ZarrPythonDriver
    from abczarr.drivers.zarrista import ZarristaDriver

    assert not inspect.iscoroutinefunction(Driver.open)
    assert inspect.iscoroutinefunction(Driver.open_async)
    # the native drivers override open_async; zarrista inherits the
    # thread-bridged default -- either way it stays a coroutine
    for driver in (TensorStoreDriver, ZarrPythonDriver, ZarristaDriver):
        assert inspect.iscoroutinefunction(driver.open_async)


def test_driver_create_has_a_coroutine_twin() -> None:
    # create/create_from_metadata are plain methods; their async twins are
    # coroutines, and every concrete driver keeps both colors
    from abczarr.drivers.base import Driver
    from abczarr.drivers.tensorstore import TensorStoreDriver
    from abczarr.drivers.zarr_python import ZarrPythonDriver
    from abczarr.drivers.zarrista import ZarristaDriver

    assert not inspect.iscoroutinefunction(Driver.create)
    assert not inspect.iscoroutinefunction(Driver.create_from_metadata)
    assert inspect.iscoroutinefunction(Driver.create_async)
    assert inspect.iscoroutinefunction(Driver.create_from_metadata_async)
    for driver in (TensorStoreDriver, ZarrPythonDriver, ZarristaDriver):
        assert inspect.iscoroutinefunction(driver.create_async)
        assert inspect.iscoroutinefunction(driver.create_from_metadata_async)


def test_update_attributes_parity() -> None:
    # both colors expose update_attributes; the write differs only in that the
    # async one is a coroutine (an assignment expression cannot be awaited, so
    # the async attrs mapping has no per-key setter -- same reason as setitem)
    assert callable(abczarr.ZarrNode.update_attributes)
    assert not inspect.iscoroutinefunction(abczarr.ZarrNode.update_attributes)
    assert inspect.iscoroutinefunction(
        abczarr.AsyncZarrNode.update_attributes
    )
    # attrs stays a read surface on both colors
    assert _is_property(abczarr.ZarrNode, "attrs")
    assert _is_property(abczarr.AsyncZarrNode, "attrs")


# --------------------------------------------------------------------------
# the async open is an awaitable: open(asynchronous=True) returns a coroutine,
# not a node -- the node arrives only once it is awaited
# --------------------------------------------------------------------------


def test_async_open_returns_an_awaitable(tmp_path: pathlib.Path) -> None:
    # open(asynchronous=True) hands back a coroutine you await, not a node
    pytest.importorskip("zarr")
    root = _zp_array(tmp_path)
    pending = abczarr.open(root + "/img", mode="r", asynchronous=True)
    assert inspect.isawaitable(pending)
    assert not isinstance(pending, (ZarrArray, AsyncZarrArray))

    arr = asyncio.run(_await(pending))  # the node arrives only once awaited
    assert isinstance(arr, AsyncZarrArray)


def test_async_open_array_and_group_are_awaitables(
    tmp_path: pathlib.Path,
) -> None:
    # the array/group variants are awaitables too, and each checks its kind
    pytest.importorskip("zarr")
    root = _zp_array(tmp_path)
    for pending in (
        abczarr.open_array(root + "/img", mode="r", asynchronous=True),
        abczarr.open_group(root, mode="r", asynchronous=True),
    ):
        assert inspect.isawaitable(pending)
        pending.close()  # a probe we do not await; close it cleanly

    arr = asyncio.run(
        _await(abczarr.open_array(root + "/img", mode="r", asynchronous=True))
    )
    assert isinstance(arr, AsyncZarrArray)
    grp = asyncio.run(
        _await(abczarr.open_group(root, mode="r", asynchronous=True))
    )
    assert isinstance(grp, AsyncZarrGroup)


async def _await(pending: object) -> object:
    return await pending  # type: ignore[misc]


def test_async_open_is_genuinely_async_over_memory(
    tmp_path: pathlib.Path,
) -> None:
    # a genuinely async open works against an fsspec memory:// store, whose
    # I/O never touches a local file -- the open awaits the metadata read and
    # round-trips through the async surface
    import uuid

    zarr = pytest.importorskip("zarr")
    pytest.importorskip("fsspec")
    url = "memory://" + uuid.uuid4().hex + "/zp.zarr"
    group = zarr.open_group(url, mode="w")
    array = group.create_array(
        "img", shape=(8, 8), chunks=(4, 4), dtype="float32"
    )
    array[:] = np.arange(64).reshape(8, 8)

    async def go() -> float:
        arr = await abczarr.open(url + "/img", mode="a", asynchronous=True)
        assert isinstance(arr, AsyncZarrArray)
        await arr.setitem(
            (slice(0, 4), slice(0, 4)), np.ones((4, 4), "float32")
        )
        block = await arr.getitem((slice(0, 4), slice(0, 4)))
        return float(np.asarray(block).sum())

    assert asyncio.run(go()) == 16.0


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

    async def go() -> float:
        arr = await abczarr.open(root + "/img", mode="a", asynchronous=True)
        assert isinstance(arr, AsyncZarrPythonArray)
        assert arr.supports("async", native=True)
        assert arr.shape == (8, 8)
        await arr.setitem(
            (slice(0, 4), slice(0, 4)), np.ones((4, 4), "float32")
        )
        block = await arr.getitem((slice(0, 4), slice(0, 4)))
        return float(np.asarray(block).sum())

    assert asyncio.run(go()) == 16.0


def test_zarr_python_concurrent_fanout(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    root = _zp_array(tmp_path)
    regions = [
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(4, 8)),
        (slice(4, 8), slice(0, 4)),
        (slice(4, 8), slice(4, 8)),
    ]

    async def go() -> list:
        arr = await abczarr.open(root + "/img", mode="r", asynchronous=True)
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

    async def go() -> None:
        grp = await abczarr.open(root, mode="a", asynchronous=True)
        assert isinstance(grp, AsyncZarrPythonGroup)
        assert grp.supports("async", native=True)
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
    corner = np.arange(64).reshape(8, 8)[0:4, 0:4].sum()

    async def go() -> None:
        arr = await abczarr.open(
            root + "/img", mode="a", asynchronous=True, driver="tensorstore"
        )
        assert isinstance(arr, AsyncTensorStoreArray)
        assert arr.supports("async", native=True)
        assert arr.shape == (8, 8)
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
    regions = [
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(4, 8)),
        (slice(4, 8), slice(0, 4)),
        (slice(4, 8), slice(4, 8)),
    ]

    async def go() -> list:
        arr = await abczarr.open(
            root + "/img", mode="r", asynchronous=True, driver="tensorstore"
        )
        return await asyncio.gather(*(arr.getitem(r) for r in regions))

    blocks = asyncio.run(go())
    total = sum(np.asarray(b).sum() for b in blocks)
    assert total == np.arange(64).sum()


def test_tensorstore_async_group_is_a_real_path_group(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("tensorstore")
    from abczarr.drivers.tensorstore import AsyncTensorStoreArray

    root = _ts_array(tmp_path)

    async def go() -> object:
        grp = await abczarr.open(
            root, mode="r", asynchronous=True, driver="tensorstore"
        )
        # tensorstore has no group object, so its async group IS the async
        # path group -- listing/navigation run on an AsyncStore, not by
        # threading the sync group. A path group synthesizes group semantics
        # over a store, so its own "async" is SYNTHESIZED even though its
        # array children are NATIVE.
        assert isinstance(grp, AsyncPathGroup)
        assert grp.capability("async") is Support.SYNTHESIZED
        assert grp.supports("async")  # synthesized still counts as supported
        # listing goes through the async store
        names = [name async for name in grp]
        assert "img" in names
        # ...and a child array comes back in tensorstore's native color
        return await grp.getitem("img")

    child = asyncio.run(go())
    assert isinstance(child, AsyncTensorStoreArray)
    assert child.supports("async", native=True)


def test_async_path_group_lists_and_navigates_nested(
    tmp_path: pathlib.Path,
) -> None:
    # a real async path group over a nested hierarchy: members are listed
    # and opened through the async store, and subgroups stay async path
    # groups (so the whole tree is reachable asynchronously)
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("tensorstore")
    root = str(tmp_path / "nested.zarr")
    group = zarr.open_group(root, mode="w")
    group.create_array("img", shape=(4, 4), chunks=(2, 2), dtype="float32")
    sub = group.create_group("sub")
    sub.create_array("inner", shape=(4, 4), chunks=(2, 2), dtype="uint8")

    async def go() -> None:
        grp = await abczarr.open(
            root, mode="r", asynchronous=True, driver="tensorstore"
        )
        assert isinstance(grp, AsyncPathGroup)
        names = sorted([name async for name in grp])
        assert names == ["img", "sub"]
        subgrp = await grp.getitem("sub")
        assert isinstance(subgrp, AsyncPathGroup)
        inner_names = [name async for name in subgrp]
        assert inner_names == ["inner"]
        inner = await subgrp.getitem("inner")
        assert inner.shape == (4, 4)

    asyncio.run(go())


def test_async_path_group_creates_children(tmp_path: pathlib.Path) -> None:
    # the real async path group can create arrays and subgroups too
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("tensorstore")
    from abczarr.drivers.tensorstore import AsyncTensorStoreArray

    root = str(tmp_path / "make.zarr")
    zarr.open_group(root, mode="w")

    async def go() -> None:
        grp = await abczarr.open(
            root, mode="a", asynchronous=True, driver="tensorstore"
        )
        assert isinstance(grp, AsyncPathGroup)
        arr = await grp.create_array("img", (8, 8), "float32")
        assert isinstance(arr, AsyncTensorStoreArray)
        await arr.setitem(
            (slice(0, 4), slice(0, 4)), np.ones((4, 4), "float32")
        )
        got = await arr.getitem((slice(0, 4), slice(0, 4)))
        assert np.asarray(got).sum() == 16.0

        sub = await grp.create_group("sub")
        assert isinstance(sub, AsyncPathGroup)
        names = sorted([name async for name in grp])
        assert names == ["img", "sub"]

    asyncio.run(go())


def test_threaded_async_group_is_the_generic_fallback() -> None:
    # a group that is neither natively async nor path-based falls to the
    # thread-pool default, honestly reported as synthesized
    from abczarr.abc.group import ZarrGroup
    from abczarr.abc.node import ZarrNode

    class _FakeGroup(ZarrGroup):
        _CAPABILITIES = {"sharding": Support.NATIVE}

        @property
        def metadata(self) -> None:
            return None

        @property
        def zarr_version(self) -> int:
            return 3

        def __getitem__(self, key: str) -> ZarrNode:
            raise KeyError(key)

        def __setitem__(self, key: str, value: ZarrNode) -> None:
            ...

        def __delitem__(self, key: str) -> None:
            ...

        def create_group(
            self, name: str, overwrite: bool = False
        ) -> ZarrGroup:
            raise NotImplementedError

        def _create_array(self, name: str, config: object) -> None:
            raise NotImplementedError

    twin = _FakeGroup("/store").as_async()
    assert isinstance(twin, ThreadedAsyncGroup)
    assert twin.capability("async") is Support.SYNTHESIZED
    assert twin.capability("sharding") is Support.NATIVE  # delegated to sync


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

    corner = np.arange(64).reshape(8, 8)[0:4, 0:4].sum()

    async def go() -> object:
        arr = await abczarr.open(
            root + "/img", mode="r", asynchronous=True, driver="zarrista"
        )
        assert isinstance(arr, ThreadedAsyncArray)
        assert arr.capability("async") is Support.SYNTHESIZED
        assert not arr.supports("async", native=True)
        assert arr.supports("async")  # synthesized still counts as supported
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


# --------------------------------------------------------------------------
# async create: create(..., asynchronous=True) is an awaitable resolving to
# the async node, mirroring async open
# --------------------------------------------------------------------------


def test_async_create_returns_an_awaitable(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    from abczarr.api.config import ArrayConfig

    pending = abczarr.create(
        str(tmp_path / "a.zarr"),
        ArrayConfig(shape=(4, 4), dtype="int16"),
        asynchronous=True,
    )
    assert inspect.isawaitable(pending)
    assert not isinstance(pending, (ZarrArray, AsyncZarrArray))
    arr = asyncio.run(_await(pending))
    assert isinstance(arr, AsyncZarrArray)


def test_zarr_python_async_create_roundtrip(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    from abczarr.api.config import ArrayConfig

    async def go() -> float:
        arr = await abczarr.create(
            str(tmp_path / "a.zarr"),
            ArrayConfig(shape=(4, 4), dtype="float32"),
            asynchronous=True,
        )
        assert isinstance(arr, AsyncZarrArray)
        assert arr.supports("async", native=True)
        await arr.setitem(
            (slice(0, 4), slice(0, 4)), np.ones((4, 4), "float32")
        )
        block = await arr.getitem((slice(0, 4), slice(0, 4)))
        return float(np.asarray(block).sum())

    assert asyncio.run(go()) == 16.0


def test_zarr_python_async_create_group(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")

    async def go() -> object:
        return await abczarr.create_group(
            str(tmp_path / "g.zarr"), asynchronous=True
        )

    grp = asyncio.run(go())
    assert isinstance(grp, AsyncZarrGroup)


def test_tensorstore_async_create_roundtrip(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    pytest.importorskip("tensorstore")
    from abczarr.api.config import ArrayConfig

    async def go() -> float:
        arr = await abczarr.create(
            str(tmp_path / "a.zarr"),
            ArrayConfig(shape=(4, 4), dtype="float32"),
            asynchronous=True,
            driver="tensorstore",
        )
        assert isinstance(arr, AsyncZarrArray)
        assert arr.supports("async", native=True)
        await arr.setitem(
            (slice(0, 4), slice(0, 4)), np.full((4, 4), 2.0, "float32")
        )
        block = await arr.getitem((slice(0, 4), slice(0, 4)))
        return float(np.asarray(block).sum())

    assert asyncio.run(go()) == 32.0


def test_tensorstore_async_create_fails_if_it_exists(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("tensorstore")
    from abczarr.api.config import ArrayConfig

    root = str(tmp_path / "a.zarr")

    async def make(overwrite: bool) -> object:
        return await abczarr.create(
            root,
            ArrayConfig(shape=(2, 2), dtype="int8", overwrite=overwrite),
            asynchronous=True,
            driver="tensorstore",
        )

    asyncio.run(make(False))
    with pytest.raises(FileExistsError):
        asyncio.run(make(False))
    # overwrite=True replaces it
    assert isinstance(asyncio.run(make(True)), AsyncZarrArray)


# --------------------------------------------------------------------------
# async open honours create modes too (local paths only, so this leg does not
# depend on the metadata-peek URL fix)
# --------------------------------------------------------------------------


def test_async_open_create_mode_makes_an_array(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")

    async def go() -> object:
        return await abczarr.open(
            str(tmp_path / "a.zarr"),
            mode="w", shape=(3,), dtype="uint8", asynchronous=True,
        )

    arr = asyncio.run(go())
    assert isinstance(arr, AsyncZarrArray)
    assert arr.shape == (3,)


def test_async_open_create_mode_makes_a_group(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")

    async def go() -> object:
        return await abczarr.open(
            str(tmp_path / "g.zarr"), mode="w", asynchronous=True
        )

    grp = asyncio.run(go())
    assert isinstance(grp, AsyncZarrGroup)


def test_async_open_w_dash_fails_if_it_exists(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    root = str(tmp_path / "g.zarr")

    async def make() -> object:
        return await abczarr.open(root, mode="w-", asynchronous=True)

    asyncio.run(make())
    with pytest.raises(FileExistsError):
        asyncio.run(make())


def test_async_open_a_opens_or_creates(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("zarr")
    root = str(tmp_path / "a.zarr")

    async def go() -> tuple:
        made = await abczarr.open(
            root, mode="a", shape=(4,), dtype="int8", asynchronous=True
        )
        again = await abczarr.open(root, mode="a", asynchronous=True)
        return made, again

    made, again = asyncio.run(go())
    assert isinstance(made, AsyncZarrArray)
    assert isinstance(again, AsyncZarrArray)
    assert again.shape == (4,)
