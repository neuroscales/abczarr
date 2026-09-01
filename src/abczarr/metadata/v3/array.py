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

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, eq_safenan, field, update
from abczarr._core.dtypes import asdtype
from abczarr._core.metadata import register_subclass

# metadata
from abczarr.metadata import base

# locals
from .base import ArrayMetadataV3
from .codecs import BytesCodec, Codec, CompressorCodec, ShardingCodec
from .dtypes import DType
from .extensions import ExtraField, MustUnderstandExtension, TypedConfig

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

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: base.ConversionPolicy = "lossy",
    ) -> base.ArrayMetadata:
        if version == 1:
            # route through v2 -- v1 and v2 share the numcodecs model
            return _to_v2(self, policy).to_version(1, policy)
        if version == 2:
            return _to_v2(self, policy)
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


def _is_serializer(codec: Codec) -> bool:
    """Whether *codec* is the array-to-bytes step of the v3 pipeline."""
    return isinstance(codec, BytesCodec) or getattr(codec, "name", None) == (
        "bytes"
    )


def _to_v2(
    self: ArrayMetadata, policy: base.ConversionPolicy = "lossy"
) -> base.ArrayMetadata:
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

    # The v3 codec pipeline is ordered: array->array codecs (v2 filters),
    # then one array->bytes serializer, then bytes->bytes codecs (v2
    # compressor). Split on the serializer by position rather than by
    # isinstance -- a generic codec (zstd, zlib, ...) is a plain Codec, not a
    # CompressorCodec, so a type check would misroute it into the filters.
    codecs = list(self.codecs)

    sharding = _pop_next(codecs, ShardingCodec)
    if sharding:
        # v2 has no shard grid; keep the inner chunk shape and drop the
        # sharding structure per the policy.
        base.report_loss(policy, "sharding", 2)
        chunk_shape = sharding.configuration.chunk_shape
        codecs.extend(sharding.configuration.codecs)

    split = next(
        (i for i, c in enumerate(codecs) if _is_serializer(c)), None
    )
    if split is None:
        endian = None
        pre = [c for c in codecs if not isinstance(c, CompressorCodec)]
        post = [c for c in codecs if isinstance(c, CompressorCodec)]
    else:
        endian = codecs[split].configuration.endian
        pre = codecs[:split]
        post = codecs[split + 1:]

    # array->array codecs become v2 filters -- a distinct hierarchy from the
    # v2 compressor, so route them through the filter registry rather than
    # leaving them as v2 codecs.
    filters = [v2.Filter.from_dict(c.to_version(2).to_dict()) for c in pre]

    # v2 holds a single bytes->bytes compressor; any extra is a loss
    compressor = None
    if post:
        compressor = post[0].to_version(2)
        if len(post) > 1:
            base.report_loss(policy, "codecs", 2)

    # v2 folds the byte order back into the dtype
    if endian:
        endian = {"big": ">", "little": "<"}.get(endian, dtype.byteorder)
        dtype = dtype.newbyteorder(endian)

    # "annotate": stash the full source so the down-conversion is reversible.
    attributes = self.attributes
    if policy == "annotate":
        attributes = {**attributes, base.SOURCE_ATTR: self.to_dict()}

    return v2.ArrayMetadata(
        shape=self.shape,
        chunks=chunk_shape,
        dtype=dtype,
        compressor=compressor,
        fill_value=self.fill_value,
        filters=filters,
        dimension_separator=separator,
        attributes=attributes,
    )
