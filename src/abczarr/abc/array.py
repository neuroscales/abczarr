__all__ = [
    "ZarrArrayConfig",
    "ZarrArray",
]

# stdlib
from abc import abstractmethod

# dependencies
import dask.array as da
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# locals
from .node import ZarrNode


class ZarrArrayConfig(tx.TypedDict, total=False):
    """Configuration for creating a Zarr Array."""

    chunk: tz.ShapeLike
    shard: tx.Optional[tz.ShapeLike]
    compressor: tx.Optional[tz.CompressorType]
    compressor_options: tx.Mapping[str, tx.Any]
    dimension_separator: tz.DimensionSeparator
    order: tz.MemoryOrder
    fill_value: tx.Optional[tx.Number]


class ZarrArray(ZarrNode):
    """Abstract interface for a Zarr array (n-dimensional data)."""

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tz.Shape:
        """Shape of the array."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """Data type of the array."""
        ...

    @property
    @abstractmethod
    def chunks(self) -> tz.Shape:
        """Chunk shape for the array.

        If the chunk grid is not regular, accessing this property should
        raise an exception.
        """
        ...

    @property
    @abstractmethod
    def shards(self) -> tx.Optional[tz.Shape]:
        """
        Shard shape, if supported; otherwise None.

        If the shard grid is not regular, accessing this property should
        raise an exception.
        """
        ...

    @abstractmethod
    def __getitem__(self, key: str) -> npt.ArrayLike:
        """Read data from the array."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: npt.ArrayLike) -> None:
        """Write data to the array."""
        ...

    def __array__(
        self, dtype: tx.Optional[npt.DTypeLike] = None
    ) -> npt.ArrayLike:
        """Convert this Zarr array to a NumPy array."""
        return np.asarray(self[()], dtype=dtype)

    def to_dask(self) -> da.Array:
        """Convert this Zarr array to a Dask array."""
        return da.from_array(self, chunks=self.chunks)
