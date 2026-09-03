"""The Zarr array interface: n-dimensional array data."""

__all__ = [
    "ZarrArray",
]

# stdlib
from abc import abstractmethod

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# locals
from .node import ZarrNode

if tx.TYPE_CHECKING:
    import dask.array as da

    from .asyncnode import AsyncZarrArray


class ZarrArray(ZarrNode):
    """An n-dimensional Zarr array.

    Read and write it like a NumPy array, with NumPy-style
    selections:

    !!! example
        ```python
        array[0, :10]
        array[...] = data
        ```
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tz.Shape:
        """The shape of the array."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The data type of the array."""
        ...

    @property
    @abstractmethod
    def chunks(self) -> tz.Shape:
        """The chunk shape of the array.

        Raises when the array's chunk grid is not regular.
        """
        ...

    @property
    @abstractmethod
    def shards(self) -> tx.Optional[tz.Shape]:
        """The shard shape of the array, or `None` if it is not
        sharded.

        Raises when the array's shard grid is not regular.
        """
        ...

    @abstractmethod
    def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
        """Read data from the array at *index* (a NumPy-style
        selection)."""
        ...

    @abstractmethod
    def __setitem__(self, index: tx.Any, value: npt.ArrayLike) -> None:
        """Write *value* at *index* (a NumPy-style selection)."""
        ...

    def __array__(
        self, dtype: tx.Optional[npt.DTypeLike] = None
    ) -> npt.ArrayLike:
        """Convert this array to a NumPy array."""
        return np.asarray(self[()], dtype=dtype)

    def as_async(self) -> "AsyncZarrArray":
        """The coroutine twin of this array, over the same backend handle.

        The default runs this array's reads and writes in a bounded thread
        pool and reports `"async"` as `Support.SYNTHESIZED`. A driver whose
        backend has a native coroutine surface overrides this to return a
        `Support.NATIVE` twin that awaits the backend's own futures.

        !!! example
            ```python
            block = await array.as_async().getitem((slice(0, 8),))
            ```
        """
        from .asyncnode import ThreadedAsyncArray

        return ThreadedAsyncArray(self)

    def to_dask(
        self, chunks: tx.Union[str, tz.ShapeLike, None] = None
    ) -> "da.Array":
        """Convert this array to a Dask array.

        Parameters
        ----------
        chunks : optional
            The Dask block size. `"shards"` uses the write unit (the shard
            when sharded, otherwise the chunk); `"chunks"` uses the chunk;
            or pass an explicit block shape. The default aligns to the write
            unit, which reads a shard once rather than once per inner chunk.
            Align to `"chunks"` instead when you mean to read from the array,
            or to `"shards"` when you mean to write back into it.
        """
        import dask.array as da

        if chunks is None or chunks == "shards":
            chunks = self.shards or self.chunks
        elif chunks == "chunks":
            chunks = self.chunks
        return da.from_array(self, chunks=chunks)

    def store(
        self,
        source: npt.ArrayLike,
        *,
        lock: tx.Union[bool, str] = "auto",
    ) -> None:
        """Write *source* into this array, block by block.

        *source* is any array-like with a matching shape. A Dask array
        is written one block at a time, so a source too large to hold
        in memory never is; a plain array is written in one go. This is
        the write counterpart of
        [to_dask][abczarr.abc.array.ZarrArray.to_dask], and it works
        for every backend (unlike `dask.array.to_zarr`, which requires
        a native `zarr.Array`).

        !!! example
            ```python
            array.store(dask_array)
            ```

        Parameters
        ----------
        source : array-like
            The data to write. Its shape must match this array's.
        lock : bool or str, optional
            Serialize concurrent block writes. The default, `"auto"`, locks
            only when the source's blocks do not line up with this array's
            write unit, since blocks that each fall on whole chunks never
            write the same chunk at once. Pass `True` or `False` to decide
            it yourself.
        """
        import dask.array as da

        darr = da.asarray(source)
        if lock == "auto":
            unit = self.shards or self.chunks
            lock = not _blocks_align_to(darr.chunks, unit)
        da.store(darr, self, lock=lock)


def _blocks_align_to(
    dask_chunks: tx.Sequence[tx.Sequence[int]], unit: tz.ShapeLike
) -> bool:
    """Whether Dask blocks fall on whole *unit*-sized chunks.

    *dask_chunks* is a Dask array's `.chunks` (per axis, the block sizes);
    *unit* is the array's write unit. True when every interior block
    boundary lands on a multiple of the unit size, so no two blocks ever
    write the same chunk and a lock is unnecessary. Conservative: anything
    it cannot prove aligned counts as not aligned.
    """
    unit = tuple(unit)
    if len(dask_chunks) != len(unit):
        return False
    for axis_blocks, size in zip(dask_chunks, unit):
        if not size:
            return False
        offset = 0
        # the final boundary is the array end; a partial last chunk there
        # is still written by a single block, so it need not align
        for block in tuple(axis_blocks)[:-1]:
            offset += block
            if offset % size:
                return False
    return True
