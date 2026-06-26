"""ZarrIO Implementation using the zarr-python library."""

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx
import zarr
from zarr.core.array import CompressorsLike
from zarr.core.chunk_key_encodings import (
    ChunkKeyEncodingLike,
    ChunkKeyEncodingParams,
)

# locals
from .. import typing as tz
from ..abc import ZarrArray, ZarrArrayConfig, ZarrGroup, ZarrNode
from ..config import ZarrConfig
from ..helpers import _compute_zarr_layout


class ZarrPythonNode(ZarrNode): ...


class ZarrPythonArray(ZarrArray, ZarrPythonNode):
    """Zarr Array implementation using the zarr-python library."""

    def __init__(self, array: zarr.Array) -> None:
        """
        Initialize the ZarrPythonArray with a zarr.Array.

        Parameters
        ----------
        array : zarr.Array
            Underlying Zarr array.
        """
        super().__init__(str(array.store_path))
        self._array = array

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        return self._array.ndim

    @property
    def shape(self) -> tz.Shape:
        """Shape of the array."""
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the array."""
        return self._array.dtype

    @property
    def chunks(self) -> tz.Shape:
        """Chunk shape for the array."""
        return self._array.chunks

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        """Shard shape, if supported; otherwise None."""
        return getattr(self._array, "shards", None)

    @property
    def attrs(self) -> tz.Attributes:
        """Access metadata/attributes for this node."""
        return self._array.attrs

    @property
    def zarr_version(self) -> int:
        """Get the Zarr format version."""
        return self._array.metadata.zarr_format

    def __getitem__(self, key: str) -> npt.ArrayLike:
        """Read data from the array."""
        return self._array[key]

    def __setitem__(self, key: str, value: npt.ArrayLike) -> None:
        """Write data to the array."""
        self._array[key] = value

    def __getattr__(self, name: str) -> tx.Any:
        """Delegate any unknown attributes to the underlying array."""
        if name == "_array":
            return self._array
        if hasattr(self._array, name):
            return getattr(self._array, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @classmethod
    def open(cls, *args, **kwargs) -> "ZarrPythonArray":
        """Open a Zarr array."""
        return cls(zarr.open_array(*args, **kwargs))

    @classmethod
    def open_array(cls, *args, **kwargs) -> "ZarrPythonArray":
        """Open a Zarr array."""
        return cls(zarr.open_array(*args, **kwargs))


class ZarrPythonGroup(ZarrGroup, ZarrPythonNode):
    """Zarr Group implementation using the zarr-python library."""

    def __init__(self, zarr_group: zarr.Group) -> None:
        """
        Initialize the ZarrPythonGroup with a zarr.Group.

        Parameters
        ----------
        zarr_group : zarr.Group
            Underlying Zarr Python group.
        """
        super().__init__(str(zarr_group.store_path))

        self._zgroup = zarr_group

    @classmethod
    def from_config(
        cls, out: tz.PathLike, zarr_config: ZarrConfig
    ) -> "ZarrPythonGroup":
        """Create a Zarr group from a configuration object."""
        store = zarr.storage.LocalStore(out)
        return cls(
            zarr.group(
                store=store,
                overwrite=zarr_config.overwrite,
                zarr_format=zarr_config.zarr_version,
            )
        )

    @property
    def attrs(self) -> tz.Attributes:
        """Access metadata/attributes for this node."""
        return self._zgroup.attrs

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        """Get the Zarr format version."""
        return self._zgroup.metadata.zarr_format

    def keys(self) -> tx.Iterator[str]:
        """Get the names of all subgroups and arrays in this group."""
        yield from self._zgroup.keys()

    def __getitem__(self, key: str) -> ZarrPythonNode:
        """Get a subgroup or array by name within this group."""
        if key not in self._zgroup:
            raise KeyError(
                f"Key '{key}' not found in group '{self.store_path}'"
            )
        item = self._zgroup[key]
        if isinstance(item, zarr.Group):
            return ZarrPythonGroup(item)
        elif isinstance(item, zarr.Array):
            return ZarrPythonArray(item)
        else:
            raise TypeError(f"Unsupported item type: {type(item)}")

    def __setitem__(self, key: str, value: ZarrPythonNode) -> None:
        """Set a subgroup or array by name within this group."""
        if isinstance(value, ZarrPythonGroup):
            self._zgroup[key] = value._zgroup
        elif isinstance(value, ZarrPythonArray):
            self._zgroup[key] = value._array
        else:
            raise TypeError(f"Unsupported item type: {type(value)}")

    def __delitem__(self, key: str) -> None:
        """Delete a subgroup or array by name within this group."""
        del self._zgroup[key]

    def __iter__(self) -> tx.Iterator[str]:
        """Iterate over the names of all subgroups and arrays in this group."""
        yield from self.keys()

    def __getattr__(self, name: str) -> tx.Any:
        """Delegate attribute access to the underlying Zarr group."""
        return getattr(self._zgroup, name)

    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        """Create or open a subgroup within this group."""
        subgroup = self._zgroup.create_group(name, overwrite=overwrite)
        return ZarrPythonGroup(subgroup)

    def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        zarr_config: ZarrConfig = None,
        data: tx.Optional[npt.ArrayLike] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrPythonArray:
        """Create a new array within this group."""
        if zarr_config is None:
            arr = self._zgroup.create_array(
                name, shape=shape, dtype=dtype, **kwargs
            )
            if data is not None:
                arr[:] = data
            return ZarrPythonArray(arr)

        compressor = zarr_config.compressor
        compressor_opt = zarr_config.compressor_opt
        chunk, shard = _compute_zarr_layout(shape, dtype, zarr_config)
        # TODO: implement fill_value
        opt = {
            "chunks": chunk,
            "shards": shard,
            "order": zarr_config.order,
            "dtype": np.dtype(dtype).str,
            "fill_value": None,
            "compressors": _make_compressor(
                compressor, zarr_config.zarr_version, **compressor_opt
            ),
            "overwrite": zarr_config.overwrite,
        }

        chunk_key_encoding = _make_chunk_key_encoding(
            zarr_config.dimension_separator, zarr_config.zarr_version
        )
        if chunk_key_encoding:
            opt["chunk_key_encoding"] = chunk_key_encoding
        arr = self._zgroup.create_array(name=name, shape=shape, **opt)
        if data:
            arr[:] = data
        return ZarrPythonArray(arr)

    def create_array_from_base(
        self,
        name: str,
        shape: tz.ShapeLike,
        data: tx.Optional[npt.ArrayLike] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrPythonArray:
        """Create a new array using the properties from a base_level object."""

        # this is very hacky, otherwise the inherited class will use
        # their override
        base_level = ZarrPythonGroup.__getitem__(self, "0")
        array = base_level._array
        opts = dict(
            dtype=base_level.dtype,
            chunks=base_level.chunks,
            shards=_getattr(base_level, "shards"),
            filters=_getattr(array, "filters"),
            compressors=_getattr(array, "compressors"),
            fill_value=_getattr(array, "fill_value"),
            order=_getattr(array, "order"),
            attributes=_getattr(_getattr(array, "metadata"), "attributes"),
            overwrite=True,
        )

        # Handle extra options based on metadata type
        meta = getattr(base_level, "metadata", None)
        if meta is not None:
            if hasattr(meta, "dimension_separator"):
                opts["chunk_key_encoding"] = _make_chunk_key_encoding(
                    meta.dimension_separator, 2
                )
            if hasattr(meta, "chunk_key_encoding"):
                opts["chunk_key_encoding"] = meta.chunk_key_encoding
            if hasattr(base_level, "serializer"):
                opts["serializer"] = base_level.serializer
            if hasattr(meta, "dimension_names"):
                opts["dimension_names"] = meta.dimension_names

        # Remove None values
        opts = {k: v for k, v in opts.items() if v is not None}
        opts.update(kwargs)
        arr = self._zgroup.create_array(name=name, shape=shape, **opts)
        if data is not None:
            arr[:] = data
        return ZarrPythonArray(arr)

    @classmethod
    def open(cls, *args, **kwargs) -> "ZarrPythonGroup":
        """Open a Zarr group."""
        return cls(zarr.open_group(*args, **kwargs))

    @classmethod
    def open_group(cls, *args, **kwargs) -> "ZarrPythonGroup":
        """Open a Zarr group."""
        return cls(zarr.open_group(*args, **kwargs))


def _make_compressor(
    name: str | None, zarr_version: tz.ZarrVersion, **prm: dict
) -> CompressorsLike:
    """Build compressor object from name and options."""
    if not isinstance(name, str):
        return name
    if name == "none":
        return None

    if zarr_version == 2:
        import numcodecs

        compressor_map = {"blosc": numcodecs.Blosc, "zlib": numcodecs.Zstd}
    elif zarr_version == 3:
        import zarr.codecs

        compressor_map = {
            "blosc": zarr.codecs.BloscCodec,
            "zlib": zarr.codecs.ZstdCodec,
        }
    else:
        raise ValueError()
    name = name.lower()

    if name not in compressor_map:
        raise ValueError("Unknown compressor", name)
    Compressor = compressor_map[name]

    return Compressor(**prm)


def _make_chunk_key_encoding(
    dimension_separator: tz.DimensionSeparator, zarr_version: tz.ZarrVersion
) -> ChunkKeyEncodingLike:
    dimension_separator = dimension_separator
    if dimension_separator == "." and zarr_version == 2:
        return None
    if dimension_separator == "/" and zarr_version == 3:
        return None
    return ChunkKeyEncodingParams(
        name="default" if zarr_version == 3 else "v2",
        separator=dimension_separator,
    )


def _getattr(x: object, name: str) -> tx.Any:
    """Get attribute from object, or None if not present."""
    return getattr(x, name, None)
