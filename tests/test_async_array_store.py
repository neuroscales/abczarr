"""AsyncZarrArray.store, the coroutine counterpart of ZarrArray.store.

The source is written over the write-unit grid (shard, else chunk) by a
bounded number of concurrent workers, without Dask. A recording fake array
checks the write grid and the concurrency bound without a backend; the
end-to-end tests run over zarr-python. pytest-asyncio is not a dependency,
so each coroutine is driven with ``asyncio.run``.
"""

import asyncio
import pathlib
import sys

import numpy as np
import numpy.typing as npt
import pytest
import typing_extensions as tx

from abczarr.abc.asynchronous import AsyncZarrArray, ThreadedAsyncArray
from abczarr.abc.sync import ZarrArray


class _RecordingArray(ZarrArray):
    """An in-memory array that records the selection of every write."""

    def __init__(
        self,
        shape: tx.Tuple[int, ...],
        chunks: tx.Tuple[int, ...],
        shards: tx.Optional[tx.Tuple[int, ...]] = None,
    ) -> None:
        self._shape = shape
        self._chunks = chunks
        self._shards = shards
        self.data = np.zeros(shape, "int32")
        self.writes: tx.List[tx.Any] = []
        super().__init__("/store")

    @property
    def metadata(self) -> None:
        return None

    @property
    def attrs(self) -> dict:
        return {}

    @property
    def zarr_version(self) -> int:
        return 3

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> tx.Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def chunks(self) -> tx.Tuple[int, ...]:
        return self._chunks

    @property
    def shards(self) -> tx.Optional[tx.Tuple[int, ...]]:
        return self._shards

    def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
        return self.data[index]

    def __setitem__(self, index: tx.Any, value: npt.ArrayLike) -> None:
        self.writes.append(index)
        self.data[index] = value


class _AsyncRecordingArray(AsyncZarrArray):
    """An async twin that tracks how many writes are in flight at once."""

    def __init__(
        self, sync: _RecordingArray, fail_at: tx.Optional[int] = None
    ) -> None:
        super().__init__(sync)
        self.in_flight = 0
        self.max_in_flight = 0
        self.cancelled = 0
        self.calls = 0
        self.fail_at = fail_at

    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        await asyncio.sleep(0)
        return self._sync[index]

    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        self.calls += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.calls == self.fail_at:
                raise RuntimeError("write failed")
            await asyncio.sleep(0.001)
            self._sync[index] = value
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1


def _no_dask(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes any import of that name fail.
    for name in list(sys.modules):
        if name == "dask" or name.startswith("dask."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.array", None)


def _sorted(writes: tx.List[tx.Any]) -> tx.List[tx.Any]:
    # Concurrent writes complete in no fixed order.
    return sorted(writes, key=lambda slc: [(s.start, s.stop) for s in slc])


# --------------------------------------------------------------------------
# the write grid (no backend)
# --------------------------------------------------------------------------


def test_store_is_a_coroutine_function() -> None:
    assert asyncio.iscoroutinefunction(AsyncZarrArray.store)


def test_store_writes_one_shard_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_dask(monkeypatch)
    sync = _RecordingArray((6, 5), chunks=(2, 2), shards=(4, 4))
    source = np.arange(30, dtype="int32").reshape(6, 5)
    asyncio.run(ThreadedAsyncArray(sync).store(source))
    np.testing.assert_array_equal(sync.data, source)
    # the last shard on each axis is partial
    assert _sorted(sync.writes) == [
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(4, 5)),
        (slice(4, 6), slice(0, 4)),
        (slice(4, 6), slice(4, 5)),
    ]
    assert "dask" not in sys.modules or sys.modules["dask"] is None


def test_store_falls_back_to_chunks_when_unsharded() -> None:
    sync = _RecordingArray((5,), chunks=(2,))
    asyncio.run(ThreadedAsyncArray(sync).store([1, 2, 3, 4, 5]))
    assert sync.data.tolist() == [1, 2, 3, 4, 5]
    assert _sorted(sync.writes) == [
        (slice(0, 2),),
        (slice(2, 4),),
        (slice(4, 5),),
    ]


def test_store_does_not_use_dask(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("dask.array")
    import dask.array as da

    def _fail(*args: tx.Any, **kwargs: tx.Any) -> None:
        raise AssertionError("dask was used")

    monkeypatch.setattr(da, "store", _fail)
    monkeypatch.setattr(da, "from_array", _fail)
    sync = _RecordingArray((4, 4), chunks=(2, 2))
    asyncio.run(ThreadedAsyncArray(sync).store(np.ones((4, 4), "int32")))
    assert (sync.data == 1).all()


def test_store_writes_a_0d_array_once() -> None:
    sync = _RecordingArray((), chunks=())
    asyncio.run(ThreadedAsyncArray(sync).store(np.asarray(7)))
    assert int(sync.data) == 7
    assert sync.writes == [()]


def test_store_writes_nothing_for_an_empty_array() -> None:
    sync = _RecordingArray((0, 3), chunks=(2, 2))
    asyncio.run(ThreadedAsyncArray(sync).store(np.zeros((0, 3))))
    assert sync.writes == []


@pytest.mark.parametrize("source", [np.ones((4, 3)), np.ones((4,)), 1.0])
def test_store_rejects_a_shape_mismatch(source: npt.ArrayLike) -> None:
    sync = _RecordingArray((4, 4), chunks=(2, 2))
    with pytest.raises(ValueError, match="shape"):
        asyncio.run(ThreadedAsyncArray(sync).store(source))
    assert sync.writes == []


def test_store_rejects_an_async_source_of_another_shape() -> None:
    source = ThreadedAsyncArray(_RecordingArray((4, 3), chunks=(2, 2)))
    sync = _RecordingArray((4, 4), chunks=(2, 2))
    with pytest.raises(ValueError, match="shape"):
        asyncio.run(ThreadedAsyncArray(sync).store(source))
    assert sync.writes == []


def test_store_rejects_a_concurrency_below_one() -> None:
    sync = _RecordingArray((4,), chunks=(2,))
    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(
            ThreadedAsyncArray(sync).store(np.ones(4), max_concurrency=0)
        )


@pytest.mark.parametrize("limit", [1, 3])
def test_store_bounds_the_writes_in_flight(limit: int) -> None:
    sync = _RecordingArray((8, 8), chunks=(2, 2))
    arr = _AsyncRecordingArray(sync)
    source = np.arange(64, dtype="int32").reshape(8, 8)
    asyncio.run(arr.store(source, max_concurrency=limit))
    np.testing.assert_array_equal(sync.data, source)
    assert len(sync.writes) == 16
    assert arr.max_in_flight == limit


def test_store_reads_an_async_source_with_await() -> None:
    src_sync = _RecordingArray((5, 4), chunks=(3, 3))
    src_sync.data[...] = np.arange(20).reshape(5, 4)
    reads: tx.List[tx.Any] = []

    class _Source(_AsyncRecordingArray):
        async def getitem(self, index: tx.Any) -> npt.ArrayLike:
            reads.append(index)
            return await super().getitem(index)

    sync = _RecordingArray((5, 4), chunks=(2, 2))
    asyncio.run(ThreadedAsyncArray(sync).store(_Source(src_sync)))
    np.testing.assert_array_equal(sync.data, src_sync.data)
    assert _sorted(reads) == _sorted(sync.writes)
    assert len(reads) == 6


def test_store_reads_only_the_written_region_of_a_lazy_source() -> None:
    reads: tx.List[tx.Any] = []
    data = np.full((4, 4), 3, "int32")

    class _Lazy:
        shape = (4, 4)
        ndim = 2
        dtype = data.dtype

        def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
            reads.append(index)
            return data[index]

    sync = _RecordingArray((4, 4), chunks=(2, 2))
    asyncio.run(ThreadedAsyncArray(sync).store(_Lazy()))
    assert (sync.data == 3).all()
    assert _sorted(reads) == _sorted(sync.writes)
    assert len(reads) == 4


def test_store_raises_and_cancels_the_other_writes() -> None:
    sync = _RecordingArray((8, 8), chunks=(2, 2))
    arr = _AsyncRecordingArray(sync, fail_at=6)

    async def go() -> None:
        before = asyncio.all_tasks()
        with pytest.raises(RuntimeError, match="write failed"):
            await arr.store(np.ones((8, 8), "int32"), max_concurrency=4)
        # no worker outlives the failed call
        assert asyncio.all_tasks() == before

    asyncio.run(go())
    assert arr.in_flight == 0
    assert arr.cancelled > 0
    assert len(sync.writes) < 16


# --------------------------------------------------------------------------
# end to end, over a real zarr array
# --------------------------------------------------------------------------

zarr = pytest.importorskip("zarr")

import abczarr  # noqa: E402
from abczarr.api.config import ArrayConfig  # noqa: E402


@pytest.mark.parametrize(
    "config",
    [
        ArrayConfig(shape=(7, 6), dtype="int32", chunks=(3, 4)),
        ArrayConfig(shape=(7, 6), dtype="int32", chunks=(2, 2), shards=(4, 4)),
    ],
    ids=["unsharded", "sharded"],
)
def test_store_writes_numpy_with_a_partial_last_chunk(
    tmp_path: pathlib.Path,
    config: ArrayConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_dask(monkeypatch)
    arr = abczarr.create(str(tmp_path / "a.zarr"), config)
    source = np.arange(42, dtype="int32").reshape(7, 6)
    asyncio.run(arr.as_async().store(source))
    np.testing.assert_array_equal(np.asarray(arr), source)


def test_store_copies_from_another_zarr_array(
    tmp_path: pathlib.Path,
) -> None:
    src = abczarr.create(
        str(tmp_path / "src.zarr"),
        ArrayConfig(shape=(9, 5), dtype="float32", chunks=(4, 4)),
    )
    expected = np.arange(45, dtype="float32").reshape(9, 5)
    src[...] = expected
    dst = abczarr.create(
        str(tmp_path / "dst.zarr"),
        ArrayConfig(
            shape=(9, 5), dtype="float32", chunks=(2, 2), shards=(4, 4)
        ),
    )
    asyncio.run(dst.as_async().store(src))
    np.testing.assert_array_equal(np.asarray(dst), expected)


def test_store_copies_from_an_async_zarr_array(
    tmp_path: pathlib.Path,
) -> None:
    src = abczarr.create(
        str(tmp_path / "src.zarr"),
        ArrayConfig(shape=(9, 5), dtype="float32", chunks=(4, 4)),
    )
    expected = np.arange(45, dtype="float32").reshape(9, 5)
    src[...] = expected
    dst = abczarr.create(
        str(tmp_path / "dst.zarr"),
        ArrayConfig(shape=(9, 5), dtype="float32", chunks=(3, 2)),
    )
    asyncio.run(dst.as_async().store(src.as_async()))
    np.testing.assert_array_equal(np.asarray(dst), expected)


def test_store_over_a_native_async_array(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "a.zarr")
    abczarr.create(
        root,
        ArrayConfig(shape=(7, 6), dtype="int32", chunks=(2, 2), shards=(4, 4)),
    )
    source = np.arange(42, dtype="int32").reshape(7, 6)

    async def go() -> None:
        arr = await abczarr.open(root, mode="a", asynchronous=True)
        await arr.store(source)

    asyncio.run(go())
    np.testing.assert_array_equal(
        np.asarray(abczarr.open(root, mode="r")), source
    )
