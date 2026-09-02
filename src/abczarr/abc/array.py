"""The Zarr array interface: n-dimensional array data."""

__all__ = [
    "ZarrArrayConfig",
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


class ZarrArrayConfig(tx.TypedDict, total=False):
    """Options for creating a Zarr array.

    Every key is optional. Pass a `config` dict to a group's
    `create_array`, or the same keys as keyword arguments, to set
    the chunking, sharding, and compression of a new array.
    """

    chunk: tz.ShapeLike
    shard: tx.Optional[tz.ShapeLike]
    compressor: tx.Optional[tz.CompressorType]
    compressor_options: tx.Mapping[str, tx.Any]
    dimension_separator: tz.DimensionSeparator
    order: tz.MemoryOrder
    fill_value: tx.Optional[tz.JSONNumber]


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

    def to_dask(self) -> "da.Array":
        """Convert this array to a Dask array.

        Dask blocks align to the write unit -- the shard when the
        array is sharded, otherwise the chunk -- so a read never
        re-fetches the same shard once per inner chunk.
        """
        import dask.array as da

        return da.from_array(self, chunks=self.shards or self.chunks)
