__all__ = [
    "ChunkGrid",
    "RegularChunkGrid",
    "RectilinearChunkGrid",
    "ChunkKeyEncoding",
    "DefaultChunkKeyEncoding",
    "V2ChunkKeyEncoding",
    "ArrayMetadata",
]
# stdlib
from warnings import warn

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, update, field, eq_safenan
from abczarr._core.dtypes import asdtype
from abczarr._core.metadata import register_subclass

# metadata
from abczarr.metadata import base

# locals
from .base import ArrayMetadataV3
from .extensions import MustUnderstandExtension, ExtraField, TypedConfig
from .codecs import Codec, CompressorCodec, ShardingCodec, BytesCodec
from .dtypes import DType


# ----------------------------------------------------------------------
#   CHUNK GRID
# ----------------------------------------------------------------------


@autofrozen
class ChunkGrid(MustUnderstandExtension):
    ...


@autofrozen(extra_items=False)
class RegularChunkGridConfig(TypedConfig):
    chunk_shape: tz.Shape


@register_subclass(name="regular")
@autofrozen
class RegularChunkGrid(ChunkGrid):
    name: tx.Literal["regular"]
    configuration: RegularChunkGridConfig


@autofrozen(extra_items=False)
class RectilinearChunkGridConfig(TypedConfig):
    kind: tx.Literal["inline"]
    chunk_shapes: tz.Shape


@register_subclass(name="rectilinear")
@autofrozen
class RectilinearChunkGrid(ChunkGrid):
    name: tx.Literal["rectilinear"]
    configuration: RectilinearChunkGridConfig


# ----------------------------------------------------------------------
#   CHUNK KEY ENCODING
# ----------------------------------------------------------------------


@autofrozen(extra_items=tz.FrozenJSON)
class ChunkKeyEncodingConfig(TypedConfig):
    ...


@autofrozen(extra_items=False)
class CommonChunkKeyEncodingConfig(ChunkKeyEncodingConfig):
    separator: tz.DimensionSeparator = "/"


@autofrozen
class ChunkKeyEncoding(MustUnderstandExtension):
    name: str
    configuration: ChunkKeyEncodingConfig

    def __new___(cls, name: str, *a, **k) -> tx.Self:
        if cls is ChunkKeyEncoding:
            if name == "default":
                return super().__new__(DefaultChunkKeyEncoding)
            elif name == "v2":
                return super().__new__(V2ChunkKeyEncoding)
        return super().__new__(cls)


@autofrozen(field_transformer=update(separator={"default": "/"}))
class DefaultChunkKeyEncodingConfig(CommonChunkKeyEncodingConfig):
    ...


@register_subclass(name="default")
@autofrozen
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    name: tx.Literal["default"]
    configuration: DefaultChunkKeyEncodingConfig


@autofrozen(field_transformer=update(separator={"default": "."}))
class V2ChunkKeyEncodingConfig(CommonChunkKeyEncodingConfig):
    ...


@register_subclass(name="v2")
@autofrozen
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    name: tx.Literal["v2"]
    configuration: V2ChunkKeyEncodingConfig


# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


_AxisNames = tx.Tuple[tx.Optional[str], ...]


@register_subclass(zarr_format=3, node_type="array")
@autofrozen(kw_only=True, extra_items=ExtraField)
class ArrayMetadata(ArrayMetadataV3):

    # --- Required ----
    shape: tz.Shape
    data_type: DType
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: tx.Optional[tz.BuiltinNumber] = field(eq=eq_safenan)
    codecs: tx.Tuple[Codec, ...]

    # --- Optional ----
    attributes: tz.FrozenJSONDict
    dimension_names: tx.Optional[_AxisNames]
    storage_transformers: tx.Tuple[tz.FrozenJSONDict, ...]

    # --- Conversion ---

    def to_version(self, version: tz.ZarrVersion) -> base.ArrayMetadata:
        if version == 1:
            return _to_v1(self)
        if version == 2:
            return _to_v2(self)
        if version == 3:
            return self
        else:
            raise ValueError(f"Unsupported version: {version}")


# ----------------------------------------------------------------------
#   CONVERTERS
# ----------------------------------------------------------------------


def _pop_next(
    seq: tx.List[tx.Type[Codec]], cls: tx.Type[Codec]
) -> tx.Optional[Codec]:
    """
    Pop the next codec of the given type from the list, if any.
    """
    for i, c in enumerate(seq):
        if isinstance(c, cls):
            return seq.pop(i)
    return None


def _to_v1(self: ArrayMetadata) -> base.ArrayMetadata:
    from abczarr.metadata import v1

    if self.chunk_grid.name != "regular":
        raise ValueError("Only regular chunk grids are supported in Zarr v1")
    chunk_grid = tx.cast(RegularChunkGrid, self.chunk_grid)
    chunk_shape = chunk_grid.configuration.chunk_shape

    # Data type
    dtype = asdtype(self.data_type)

    # Preprocess codecs
    filters = list(self.codecs)

    sharding = _pop_next(filters, ShardingCodec)
    if sharding:
        chunk_shape = sharding.configuration.chunk_shape
        filters.extend(sharding.configuration.codecs)

    compression = _pop_next(filters, CompressorCodec)
    if compression:
        compression = tx.cast(CompressorCodec, compression)
        compression = compression.to_version(1)

    endian = _pop_next(filters, BytesCodec)
    if endian:
        endian = tx.cast(BytesCodec, endian)
        endian = endian.configuration.endian

    # If remaining filters, warn
    if filters:
        warn(f"Ignoring filters imcompatible with Zarr v1: {filters}")

    # Preprocess compressor
    compression_opts = None
    if compression:
        compression_opts = compression.to_version(1)
        compression = compression_opts.id

    # Fix endianness
    if endian:
        endian = {"big": ">", "little": "<"}.get(endian, dtype.byteorder)
        dtype = dtype.newbyteorder(endian)

    return v1.ArrayMetadata(
        shape=self.shape,
        chunks=chunk_shape,
        dtype=self.data_type.to_version(1),
        compression=compression,
        compression_opts=compression_opts,
        fill_value=self.fill_value,
    )


def _to_v2(self: ArrayMetadata) -> base.ArrayMetadata:
    from abczarr.metadata import v2

    if self.chunk_grid.name != "regular":
        raise ValueError("Only regular chunk grids are supported in Zarr v2")
    chunk_grid = tx.cast(RegularChunkGrid, self.chunk_grid)
    chunk_shape = chunk_grid.configuration.chunk_shape

    # Separator
    separator = getattr(
        self.chunk_key_encoding.configuration, "separator", "."
    )

    # Data type
    dtype = asdtype(self.data_type)

    # Preprocess codecs
    filters = list(self.codecs)

    sharding = _pop_next(filters, ShardingCodec)
    if sharding:
        chunk_shape = sharding.configuration.chunk_shape
        filters.extend(sharding.configuration.codecs)

    compressor = _pop_next(filters, CompressorCodec)
    if compressor:
        compressor = tx.cast(CompressorCodec, compressor)
        compressor = compressor.to_version(2)

    endian = _pop_next(filters, BytesCodec)
    if endian:
        endian = tx.cast(BytesCodec, endian)
        endian = endian.configuration.endian

    # Convert remaining filters
    filters = [f.to_version(2) for f in filters]

    # Preprocess compressor
    if compressor:
        compressor = compressor.to_version(2)

    # Fix endianness
    if endian:
        endian = {"big": ">", "little": "<"}.get(endian, dtype.byteorder)
        dtype = dtype.newbyteorder(endian)

    return v2.ArrayMetadata(
        shape=self.shape,
        chunks=chunk_shape,
        dtype=dtype,
        compressor=compressor,
        fill_value=self.fill_value,
        filters=filters,
        dimension_separator=separator
    )
