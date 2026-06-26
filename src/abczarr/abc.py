"""Abstract base classes for ZarrIO interfaces."""

# stdlib
import logging
import os
from abc import ABC, abstractmethod
from numbers import Number

# dependencies
import dask.array as da
import numpy as np
import numpy.typing as npt
import typing_extensions as tx
from dask.diagnostics import ProgressBar

# locals
from . import typing as tz
from .config import ZarrConfig
from .generate_pyramid import (
    compute_next_level,
    default_levels,
    next_level_shape,
)
from .ome import write_ome_metadata
from .path import Path

# optionals
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x: tx.Iterable, *a, **kw) -> tx.Iterable:
        return x


class ZarrArrayConfig(tx.TypedDict):
    """Configuration for creating a Zarr Array."""

    chunk: tz.Shape
    shard: tx.Optional[tz.Shape]
    compressor: tx.Optional[tz.CompressorType]
    compressor_options: tx.Mapping[str, tx.Any]
    dimension_separator: tz.DimensionSeparator
    order: tz.ArrayOrder
    fill_value: tx.Optional[Number]


class ZarrNode(ABC):
    """Base class for any Zarr-like object (group or array)."""

    def __init__(self, store_path: tz.PathLike) -> None:
        if isinstance(store_path,(str, bytes)):
            store_path = Path(store_path)
        self._store_path = store_path

    @property
    def store_path(self) -> os.PathLike:
        """Path to the Zarr store for this node."""
        return self._store_path

    @property
    @abstractmethod
    def attrs(self) -> tz.Attributes:
        """Access metadata/attributes for this node."""
        ...

    @property
    @abstractmethod
    def zarr_version(self) -> tz.ZarrVersion:
        """Get the Zarr format version."""
        ...

    # @abstractmethod
    # def __repr__(self) -> str:
    #     """Return the string representation of the Zarr node."""
    #     ...

    # @abstractmethod
    # def __str__(self) -> str:
    #     """Return the string representation of the Zarr node."""
    #     ...


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
        """Chunk shape for the array."""
        ...

    @property
    @abstractmethod
    def shards(self) -> tx.Optional[tz.Shape]:
        """Shard shape, if supported; otherwise None."""
        ...

    @abstractmethod
    def __getitem__(self, key: str) -> npt.ArrayLike:
        """Read data from the array."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: npt.ArrayLike) -> None:
        """Write data to the array."""
        ...

    def to_dask(self) -> da.Array:
        """Convert this Zarr array to a Dask array."""
        return da.from_array(self, chunks=self.chunks)


class ZarrGroup(ZarrNode):
    """Abstract interface for a Zarr group (container of arrays/subgroups)."""

    @classmethod
    @abstractmethod
    def from_config(cls, zarr_config: ZarrConfig) -> "ZarrGroup":
        """Create a Zarr group from a configuration object."""
        ...

    @abstractmethod
    def __getitem__(self, key: str) -> ZarrNode:
        """Get a subgroup or array by name within this group."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: ZarrNode) -> None:
        """Set a subgroup or array by name within this group."""

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        """Delete a subgroup or array by name within this group."""
        ...

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        """Create or open a subgroup within this group."""
        ...

    @abstractmethod
    def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        zarr_config: ZarrConfig = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrArray:
        """Create a new array within this group."""
        ...

    @abstractmethod
    def create_array_from_base(
        self,
        name: str,
        shape: tz.ShapeLike,
        data: tx.Optional[npt.ArrayLike] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrArray:
        """Create a new array using metadata of an existing array."""
        ...

    def generate_pyramid(
        self,
        levels: int = -1,
        ndim: int = 3,
        mode: tz.PyramidMode = "mean",
        no_pyramid_axis: tx.Optional[int] = None,
    ) -> list[list[int]]:
        """
        Generate the levels of a pyramid in an existing Zarr.

        Parameters
        ----------
        levels : int
            Number of additional levels to generate. By default, stop when
            all dimensions are smaller than their corresponding chunk size.
        ndim : int
            Number of spatial dimensions.
        mode : {"mean", "median"}
            Function to be used for down-sampling, either a callable or
            mean or median.
        no_pyramid_axis : int | None
            Axis to leave unsampled.

        Returns
        -------
        shapes : list[list[int]]
            Shapes of each level, from finest to coarsest.
        """
        logger = logging.getLogger("PyramidGeneration")
        base = self["0"]
        batch_shape, spatial_shape = base.shape[:-ndim], base.shape[-ndim:]
        all_shapes = [spatial_shape]
        chunk_size = base.chunks[-ndim:]
        if isinstance(mode, tx.Callable):
            window = mode
        else:
            window_func = {"median": da.nanmedian, "mean": da.nanmean}
            if mode not in window_func:
                raise ValueError(f"Unsupported mode: {mode}")
            window = window_func[mode]

        if levels == -1:
            levels = default_levels(spatial_shape, chunk_size, no_pyramid_axis)

        for lvl in tqdm(range(1, levels + 1)):
            spatial_shape = next_level_shape(spatial_shape, no_pyramid_axis)
            all_shapes.append(spatial_shape)
            full_shape = (*batch_shape, *spatial_shape)
            logger.info(f"Compute level {lvl} with shape {spatial_shape}")
            arr = self.create_array_from_base(str(lvl), full_shape)
            dat = self[str(lvl - 1)].to_dask()
            dat = compute_next_level(dat, ndim, no_pyramid_axis, window)
            dat = dat.rechunk(arr.shards or arr.chunks).persist()
            with ProgressBar():
                dat.store(arr)

        return all_shapes

    def write_ome_metadata(
        self,
        axes: tx.Sequence[tz.AxisName],
        space_scale: tz.OneOrMore[float] = 1.0,
        time_scale: float = 1.0,
        space_unit: str = "micrometer",
        time_unit: str = "second",
        name: str = "",
        pyramid_aligns: tz.OneOrMore[tx.Union[str, int]] = 2,
        levels: tx.Optional[int] = None,
        no_pool: tx.Optional[int] = None,
        multiscales_type: str = "",
        ome_version: tz.OMEVersion = "0.4",
    ) -> None:
        """
        Write OME-compatible metadata into this group.

        Parameters
        ----------
        axes : list[str]
            Name of each dimension, in Zarr order (t, c, z, y, x)
        space_scale : float | list[float]
            Finest-level voxel size, in Zarr order (z, y, x)
        time_scale : float
            Time scale
        space_unit : str
            Unit of spatial scale (assumed identical across dimensions)
        time_unit : str
            Unit of timescale
        name : str
            Name attribute
        pyramid_aligns : float | list[float] | {"center", "edge"}
            Whether the pyramid construction aligns the edges or the centers
            of the corner voxels. If a (list of) number, assume that a moving
            window of that size was used.
        levels : int
            Number of existing levels. Default: find out automatically.
        no_pool: int
            Index of the spatial dimension that was not down-sampled
            when generating pyramid levels.
        multiscales_type: str
            Override the type field in multiscale attribute.
        ome_version: {"0.4", "0.5"}
            Version of the OME-Zarr specification to use

        Returns
        -------
        None.
        """
        write_ome_metadata(
            self,
            space_scale=space_scale,
            time_scale=time_scale,
            space_unit=space_unit,
            time_unit=time_unit,
            axes=axes,
            name=name,
            pyramid_aligns=pyramid_aligns,
            levels=levels,
            no_pool=no_pool,
            multiscales_type=multiscales_type,
            ome_version=ome_version,
        )
