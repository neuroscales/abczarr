"""The async array interface: the coroutine twin of the Zarr array.

[AsyncZarrArray][abczarr.abc.async_array.AsyncZarrArray] reads and writes
through **methods**, not `[]`: `await array.getitem(index)` and
`await array.setitem(index, value)`. An assignment expression cannot be
awaited, so `__setitem__` cannot be a coroutine -- the same choice
zarr-python made. The read-only properties (`shape`, `dtype`, `chunks`, ...)
never block, so they stay synchronous.
"""

__all__ = [
    "AsyncZarrArray",
    "ThreadedAsyncArray",
]

# stdlib
from abc import abstractmethod

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.asyncutils import run_sync

# locals
from .array import ZarrArray
from .async_node import AsyncZarrNode


class AsyncZarrArray(AsyncZarrNode):
    """The coroutine twin of a [ZarrArray][abczarr.abc.array.ZarrArray].

    Read and write it with `await`:

    !!! example
        ```python
        block = await array.getitem((slice(0, 64), slice(0, 64)))
        await array.setitem((slice(0, 64), slice(0, 64)), block * 2)
        ```
    """

    def __init__(self, sync: ZarrArray) -> None:
        self._sync = sync

    def as_sync(self) -> ZarrArray:
        """The synchronous [ZarrArray][abczarr.abc.array.ZarrArray] twin."""
        return self._sync

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        return self._sync.ndim

    @property
    def shape(self) -> tz.Shape:
        """The shape of the array."""
        return self._sync.shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the array."""
        return self._sync.dtype

    @property
    def chunks(self) -> tz.Shape:
        """The chunk shape of the array."""
        return self._sync.chunks

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        """The shard shape of the array, or `None` if it is not sharded."""
        return self._sync.shards

    @abstractmethod
    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        """Read data from the array at *index* (a NumPy-style selection)."""
        ...

    @abstractmethod
    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        """Write *value* at *index* (a NumPy-style selection)."""
        ...


class ThreadedAsyncArray(AsyncZarrArray):
    """An [AsyncZarrArray][abczarr.abc.async_array.AsyncZarrArray] that runs a
    sync array's reads and writes in a bounded thread pool.

    The default async array for a backend that has no coroutine surface of
    its own. It reports `"async"` as `Support.SYNTHESIZED`.
    """

    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        return await run_sync(self._sync.__getitem__, index)

    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        await run_sync(self._sync.__setitem__, index, value)
