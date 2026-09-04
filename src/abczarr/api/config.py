"""Describe a Zarr array or group to create.

An [ArrayConfig][abczarr.api.config.ArrayConfig] or
[GroupConfig][abczarr.api.config.GroupConfig] is a reusable description of
what to create. It carries the coarse choices (chunking, sharding,
compression, the
format version) and lowers them to the exact metadata a driver writes. Hand
one to [create][abczarr.api.create] to make the array or group it describes,
or spread its fields into a group's `create_array` as keyword arguments.

`"auto"` on a field means "let the target version decide": an auto chunk size
fits a byte budget, an auto compressor is the version's default, an auto fill
value is the dtype's zero. One config then lowers correctly to either format
version.
"""

__all__ = [
    "ZarrConfig",
    "GroupConfig",
    "ArrayConfig",
    "OMEZarrConfig",
    "ZarrOptions",
    "GroupOptions",
    "ArrayOptions",
]

# stdlib
from collections import abc as _abc

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# core
from .._core import typing as tz
from .._core.attrs import autodefine, evolve, field, fields
from .._core.dtypes import to_zarr3 as dtype_to_zarr3
from .._core.sharding import ChunkSpec, auto_chunk, auto_shard
from ..metadata.base import ArrayMetadata


@autodefine
class ZarrConfig:
    """The choices shared by everything abczarr creates.

    `ArrayConfig` and `GroupConfig` build on this; it is not created on its
    own.

    Parameters
    ----------
    zarr_version : int
        The Zarr format version to write.
    overwrite : bool
        Replace whatever is already at the target location instead of raising.
    driver : str, optional
        The name of the driver to create with. `None` lets abczarr pick one
        that can write the result.
    attributes : mapping
        User attributes to store on the node.
    """

    zarr_version: tz.ZarrVersion = 3
    overwrite: bool = False
    driver: tx.Optional[str] = None
    attributes: tz.JsonDict = field(factory=dict)

    # -- mapping protocol, so a config can be spread as ``**config`` and
    #    passed through ``dict(config)`` wherever a plain options mapping is
    #    expected.
    def keys(self) -> tx.List[str]:
        """The config's field names, so `dict(config)` and `**config` work."""
        return [f.name for f in fields(type(self))]

    def __getitem__(self, key: str) -> tx.Any:
        if key not in self.keys():
            raise KeyError(key)
        return getattr(self, key)


@autodefine
class GroupConfig(ZarrConfig):
    """A description of a group to create.

    A group holds no data of its own, so it adds nothing to
    [ZarrConfig][abczarr.api.config.ZarrConfig].
    """


@autodefine
class ArrayConfig(ZarrConfig):
    """A description of an array to create.

    Set `shape` and `dtype` on the config, or supply them when it is lowered.
    The chunking, sharding, and compression fields accept `"auto"`, which is
    resolved against the shape, dtype, and format version.

    Parameters
    ----------
    shape : tuple of int, optional
        The array's shape. Required by the time the config is lowered.
    dtype : numpy dtype, optional
        The array's data type. Required by the time the config is lowered.
    dimension_names : tuple of str, optional
        A name for each axis, used by a chunk mapping and written to v3
        metadata.
    chunks : int, "auto", None, sequence or mapping
        The chunk size per axis. `"auto"` (or `None`) fits `max_chunk_bytes`;
        `-1` means the whole axis. A short sequence repeats its last entry.
    shards : int, "auto", sequence, mapping or None
        The shard size per axis, or `None` for no sharding (Zarr v3 only).
    max_chunk_bytes : int
        The target chunk size in bytes when a chunk size is `"auto"`.
    max_shard_bytes : int
        The target shard size in bytes when a shard size is `"auto"`.
    compression_ratio : float
        The compression factor the byte budgets assume.
    compressor : str, mapping, "auto" or None
        The compressor. `"auto"` picks abczarr's default, which is `zstd` (the
        Zarr v3 spec names no default; this follows zarr-python 3.x, which
        tensorstore also reads). `None` (or `"none"`) is no compression; a
        mapping is a codec spec passed through untouched.
    compressor_options : mapping
        Options for a named compressor.
    filters : tuple of mapping
        Codecs applied before the main compressor, passed through as specs.
    fill_value : number, "auto" or None
        The value read where nothing was written. `"auto"` is the dtype's
        zero.
    order : {"C", "F"}
        The in-memory layout. Stored in Zarr v2 metadata; on v3 it is a
        runtime memory-layout preference (not written to `zarr.json`).
    dimension_separator : {"/", ".", "auto"}
        The separator between chunk indices in a key, `"auto"` (or `None`)
        being `/`. A v2 concept; v3 carries it in the chunk-key encoding.
    """

    shape: tx.Optional[tz.Shape] = None
    dtype: tx.Optional[np.dtype] = None
    dimension_names: tx.Optional[tx.Tuple[tx.Optional[str], ...]] = None
    chunks: tx.Optional[ChunkSpec] = "auto"
    shards: tx.Optional[ChunkSpec] = None
    max_chunk_bytes: int = 8 * 1024**2
    max_shard_bytes: int = 2 * 1024**3
    compression_ratio: float = 1.8
    compressor: tx.Union[str, tz.JsonDict, None] = "auto"
    compressor_options: tz.JsonDict = field(factory=dict)
    filters: tx.Tuple[tz.JsonDict, ...] = ()
    fill_value: tx.Union[tz.BuiltinNumber, str, None] = "auto"
    order: tz.MemoryOrder = "C"
    dimension_separator: tx.Union[tz.DimensionSeparator, str] = "auto"

    def __attrs_post_init__(self) -> None:
        if self.zarr_version < 3 and self.shards:
            raise ValueError("sharding requires Zarr v3")

    def resolve(
        self, data: tx.Optional[npt.ArrayLike] = None, **overrides: tx.Any
    ) -> tx.Self:
        """Return a copy with `"auto"` chunking and sharding worked out.

        Takes the shape and dtype from *data* if given, else from *overrides*,
        else from the config. Raises if neither the config nor the call
        supplies a shape and a dtype.
        """
        shape = overrides.get("shape", self.shape)
        dtype = overrides.get("dtype", self.dtype)
        names = overrides.get("dimension_names", self.dimension_names)
        if data is not None:
            shape = getattr(data, "shape", shape)
            dtype = getattr(data, "dtype", dtype)
        if shape is None or dtype is None:
            raise ValueError(
                "ArrayConfig needs a shape and a dtype to create an array"
            )
        shape = tuple(shape)
        dtype = np.dtype(dtype)

        # a None chunk size means "auto" (shards keep None for "no sharding")
        chunk_spec = "auto" if self.chunks is None else self.chunks
        chunks = auto_chunk(
            shape,
            _normalize_axis(chunk_spec),
            names=names or (),
            itemsize=dtype,
            maxsize=self.max_chunk_bytes,
            compression_ratio=self.compression_ratio,
        )
        shards = self.shards
        if shards is not None:
            shards, chunks = auto_shard(
                shape,
                _normalize_axis(shards),
                _normalize_axis(chunk_spec),
                names=names or (),
                itemsize=dtype,
                maxsize=self.max_shard_bytes,
                compression_ratio=self.compression_ratio,
            )
        return evolve(
            self,
            shape=shape,
            dtype=dtype,
            dimension_names=names,
            chunks=tuple(chunks),
            shards=tuple(shards) if shards is not None else None,
        )

    def to_metadata(self) -> ArrayMetadata:
        """Lower the config to the metadata a driver writes.

        Resolves `"auto"` fields, builds Zarr v3 metadata, and converts it to
        `zarr_version` if that is not 3.
        """
        config = self.resolve()
        compressors = _compressor_codecs(
            config.compressor, config.compressor_options
        )
        _fill_blosc_typesize(compressors, config.dtype)
        filters = [dict(f) for f in config.filters]
        # the data's serializer carries the dtype's endianness (none for a
        # single-byte dtype); the shard index is uint64, so it is little.
        bytes_codec = _bytes_codec(config.dtype)
        index_bytes = {"name": "bytes", "configuration": {"endian": "little"}}
        separator = _resolve_separator(config.dimension_separator)

        if config.shards:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": config.shards},
            }
            codecs = [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": config.chunks,
                    "codecs": [*filters, bytes_codec, *compressors],
                    "index_codecs": [index_bytes, {"name": "crc32c"}],
                    "index_location": "end",
                },
            }]
        else:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": config.chunks},
            }
            codecs = [*filters, bytes_codec, *compressors]

        metadata = {
            "zarr_format": 3,
            "shape": config.shape,
            "data_type": dtype_to_zarr3(config.dtype),
            "chunk_grid": chunk_grid,
            "codecs": codecs,
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": separator},
            },
            "fill_value": _resolve_fill(config.fill_value, config.dtype),
            "attributes": dict(config.attributes),
        }
        result = ArrayMetadata.from_dict(metadata).to_version(
            config.zarr_version
        )
        # order is Zarr v2 metadata; v3 treats it as a runtime memory layout,
        # not stored in zarr.json, so it is only set here for v2.
        if config.zarr_version == 2 and config.order != "C":
            result = evolve(result, order=config.order)
        return result

    # -- resolved pieces, for a driver that creates natively rather than from
    #    a written metadata document.

    def compressor_codecs(self) -> tx.List[tz.JsonDict]:
        """The compressor as a list of zero or one v3 codec specs.

        `"auto"` becomes the version default (zstd); `None` (or `"none"`) is
        no compressor; a mapping passes through.
        """
        return _compressor_codecs(self.compressor, self.compressor_options)

    def resolved_fill_value(self) -> tx.Any:
        """The fill value, with `"auto"` turned into the dtype's zero."""
        return _resolve_fill(self.fill_value, self.dtype)

    def resolved_separator(self) -> str:
        """The chunk-key separator, with `"auto"` (or `None`) resolved."""
        return _resolve_separator(self.dimension_separator)


@autodefine
class OMEZarrConfig(ArrayConfig):
    """A description of an OME-Zarr image to create.

    Extends [ArrayConfig][abczarr.api.config.ArrayConfig] with the pyramid and
    axis choices OME-Zarr adds: how many resolution levels to build, which
    axis to hold at full resolution, and whether to give each channel or time
    point its own chunk or shard.

    Parameters
    ----------
    chunk_channels : bool
        Give each channel its own chunk.
    chunk_time : bool
        Give each time point its own chunk.
    shard_channels : bool
        Give each channel its own shard.
    shard_time : bool
        Give each time point its own shard.
    no_time : bool
        Read a fourth axis as channel rather than time.
    no_pyramid_axis : str, optional
        A spatial axis to leave un-downsampled across pyramid levels.
    levels : int
        The number of pyramid levels, or -1 for as many as fit a chunk.
    ome_version : str
        The OME-Zarr specification version to write.
    """

    chunk_channels: bool = False
    chunk_time: bool = True
    shard_channels: bool = False
    shard_time: bool = False
    no_time: bool = False
    no_pyramid_axis: tx.Optional[tz.SpatialAxisName] = None
    levels: int = -1
    ome_version: tz.OMEVersion = "0.4"


class ZarrOptions(tx.TypedDict, total=False):
    """The keyword form of [ZarrConfig][abczarr.api.config.ZarrConfig]."""

    zarr_version: tz.ZarrVersion
    overwrite: bool
    driver: tx.Optional[str]
    attributes: tz.JsonDict


class GroupOptions(ZarrOptions, total=False):
    """The keyword form of [GroupConfig][abczarr.api.config.GroupConfig]."""


class ArrayOptions(ZarrOptions, total=False):
    """The keyword form of [ArrayConfig][abczarr.api.config.ArrayConfig].

    Every [ArrayConfig][abczarr.api.config.ArrayConfig] field except `shape`
    and `dtype`, which a group's `create_array` takes as positional arguments.
    """

    dimension_names: tx.Optional[tx.Tuple[tx.Optional[str], ...]]
    chunks: ChunkSpec
    shards: tx.Optional[ChunkSpec]
    max_chunk_bytes: int
    max_shard_bytes: int
    compression_ratio: float
    compressor: tx.Union[str, tz.JsonDict, None]
    compressor_options: tz.JsonDict
    filters: tx.Tuple[tz.JsonDict, ...]
    fill_value: tx.Union[tz.BuiltinNumber, str, None]
    order: tz.MemoryOrder
    dimension_separator: tx.Union[tz.DimensionSeparator, str]


def _normalize_axis(spec: tx.Any) -> tx.Any:
    """Map the dask-style `-1` (a whole axis) to the sizing layer's `0`."""
    if spec is None or spec == "auto":
        return spec
    if isinstance(spec, int):
        return 0 if spec == -1 else spec
    if isinstance(spec, _abc.Mapping):
        return {k: (0 if v == -1 else v) for k, v in spec.items()}
    return [(0 if (isinstance(x, int) and x == -1) else x) for x in spec]


def _compressor_codecs(
    compressor: tx.Any, options: tx.Optional[tz.JsonDict]
) -> tx.List[tz.JsonDict]:
    """The compressor as a list of zero or one codec specs."""
    if compressor in (None, "none", "raw"):
        return []
    if compressor == "auto":
        compressor = "zstd"
    if isinstance(compressor, _abc.Mapping):
        return [dict(compressor)]
    return [{
        "name": str(compressor).lower(),
        "configuration": dict(options or {}),
    }]


def _bytes_codec(dtype: npt.DTypeLike) -> tz.JsonDict:
    """The array-to-bytes codec for *dtype*.

    A multi-byte dtype carries its endianness; a single-byte dtype, for which
    endianness does not apply, carries none (the codec is written as a bare
    name).
    """
    order = np.dtype(dtype).byteorder
    if order == "|":
        return {"name": "bytes"}
    return {
        "name": "bytes",
        "configuration": {"endian": "big" if order == ">" else "little"},
    }


def _fill_blosc_typesize(
    codecs: tx.List[tz.JsonDict], dtype: npt.DTypeLike
) -> None:
    """Fill a blosc codec's required `typesize` from the dtype's item size.

    The v3 blosc codec requires `typesize` (a positive integer) unless
    `shuffle` is `"noshuffle"`, where it is ignored. Set it from the dtype
    when `shuffle` is active and it is not already given; otherwise leave it
    out (the codec omits an unset one rather than writing null).
    """
    itemsize = int(np.dtype(dtype).itemsize)
    for codec in codecs:
        if codec.get("name") != "blosc":
            continue
        configuration = codec.setdefault("configuration", {})
        if (
            configuration.get("shuffle", "shuffle") != "noshuffle"
            and "typesize" not in configuration
        ):
            configuration["typesize"] = itemsize


def _resolve_fill(fill: tx.Any, dtype: npt.DTypeLike) -> tx.Any:
    """Turn an `"auto"` fill value into the dtype's zero."""
    if isinstance(fill, str) and fill == "auto":
        return np.zeros((), np.dtype(dtype)).item()
    return fill


def _resolve_separator(separator: tx.Any) -> str:
    """Turn an `"auto"` (or `None`) separator into the v3 default."""
    return "/" if separator in ("auto", None) else separator
