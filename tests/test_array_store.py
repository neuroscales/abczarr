"""ZarrArray.store for a source that is not a Dask array.

Such a source is written serially, one write unit (shard, else chunk) at a
time, and the write needs no dask. A recording fake array checks the write
grid without a backend; the end-to-end tests run over zarr-python.
"""

import pathlib
import sys

import numpy as np
import numpy.typing as npt
import pytest
import typing_extensions as tx

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


def _no_dask(monkeypatch: pytest.MonkeyPatch) -> None:
    # A None entry in sys.modules makes any import of that name fail.
    for name in list(sys.modules):
        if name == "dask" or name.startswith("dask."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "dask", None)
    monkeypatch.setitem(sys.modules, "dask.array", None)


# --------------------------------------------------------------------------
# the serial path (no backend, no dask)
# --------------------------------------------------------------------------


def test_store_writes_one_shard_at_a_time_without_dask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_dask(monkeypatch)
    with pytest.raises(ImportError):
        import dask.array  # noqa: F401
    arr = _RecordingArray((6, 5), chunks=(2, 2), shards=(4, 4))
    source = np.arange(30, dtype="int32").reshape(6, 5)
    arr.store(source)
    np.testing.assert_array_equal(arr.data, source)
    # the last shard on each axis is partial
    assert arr.writes == [
        (slice(0, 4), slice(0, 4)),
        (slice(0, 4), slice(4, 5)),
        (slice(4, 6), slice(0, 4)),
        (slice(4, 6), slice(4, 5)),
    ]


def test_store_falls_back_to_chunks_when_unsharded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_dask(monkeypatch)
    arr = _RecordingArray((5,), chunks=(2,))
    arr.store([1, 2, 3, 4, 5])
    assert arr.data.tolist() == [1, 2, 3, 4, 5]
    assert arr.writes == [(slice(0, 2),), (slice(2, 4),), (slice(4, 5),)]


def test_store_reads_only_the_written_region_of_a_lazy_source() -> None:
    source = _RecordingArray((4, 4), chunks=(4, 4))
    source.data[...] = 3
    reads: tx.List[tx.Any] = []
    getitem = source.__getitem__

    class _Lazy:
        shape = (4, 4)

        def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
            reads.append(index)
            return getitem(index)

    arr = _RecordingArray((4, 4), chunks=(2, 2))
    arr.store(_Lazy())
    assert (arr.data == 3).all()
    assert reads == arr.writes
    assert len(reads) == 4


def test_store_writes_a_0d_array_once() -> None:
    arr = _RecordingArray((), chunks=())
    arr.store(np.asarray(7))
    assert int(arr.data) == 7
    assert arr.writes == [()]


@pytest.mark.parametrize("source", [np.ones((4, 3)), np.ones((4,)), 1.0])
def test_store_rejects_a_shape_mismatch(source: npt.ArrayLike) -> None:
    arr = _RecordingArray((4, 4), chunks=(2, 2))
    with pytest.raises(ValueError, match="shape"):
        arr.store(source)
    assert arr.writes == []


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
    tmp_path: pathlib.Path, config: ArrayConfig
) -> None:
    arr = abczarr.create(str(tmp_path / "a.zarr"), config)
    source = np.arange(42, dtype="int32").reshape(7, 6)
    arr.store(source)
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
    dst.store(src)
    np.testing.assert_array_equal(np.asarray(dst), expected)


def test_store_does_not_call_dask_for_a_numpy_source(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    da = pytest.importorskip("dask.array")

    def _fail(*args: tx.Any, **kwargs: tx.Any) -> None:
        raise AssertionError("dask.array.store was called")

    monkeypatch.setattr(da, "store", _fail)
    arr = abczarr.create(
        str(tmp_path / "a.zarr"),
        ArrayConfig(shape=(5, 5), dtype="int32", chunks=(2, 2)),
    )
    arr.store(np.full((5, 5), 4, dtype="int32"))
    assert (np.asarray(arr) == 4).all()
