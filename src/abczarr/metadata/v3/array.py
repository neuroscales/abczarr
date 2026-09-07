"""Zarr v3 array metadata.

Zarr v3 describes an array with three things: a chunk grid, which
says how the array is divided into chunks, a chunk-key encoding,
which says how a chunk's index maps to its key in the store, and an
ordered pipeline of codecs applied to each chunk.
[ArrayMetadata.to_version][abczarr.metadata.v3.array.ArrayMetadata.to_version]
converts to v2 or v1 by folding the codec pipeline back into v2's
compressor, filters and byte-order-bearing dtype. That conversion
requires a regular chunk grid.
"""

__all__ = [
    "ChunkGrid",
    "RegularChunkGridConfig",
    "RegularChunkGrid",
    "RectilinearChunkGridConfig",
    "RectilinearChunkGrid",
    "ChunkKeyEncodingConfig",
    "CommonChunkKeyEncodingConfig",
    "ChunkKeyEncoding",
    "DefaultChunkKeyEncodingConfig",
    "DefaultChunkKeyEncoding",
    "V2ChunkKeyEncodingConfig",
    "V2ChunkKeyEncoding",
    "ArrayMetadata",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, eq_safenan, field, update
from abczarr._core.dtypes import asdtype
from abczarr._core.features import feature_key
from abczarr._core.metadata import register_subclass
from abczarr.errors import UnsupportedConversion

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
    """How an array is divided into chunks.

    Use [RegularChunkGrid][abczarr.metadata.v3.array.RegularChunkGrid]
    for a fixed chunk shape, or
    [RectilinearChunkGrid][abczarr.metadata.v3.array.RectilinearChunkGrid]
    for chunks that vary in size along an axis.
    """


@autofrozen(extra_items=False)
class RegularChunkGridConfig(TypedConfig):
    """Holds the regular chunk grid's parameters: the shared chunk
    shape.
    """

    chunk_shape: tz.Shape
    """The shape of every chunk, one entry per dimension of the
    array."""


@register_subclass(name="regular")
@autofrozen
class RegularChunkGrid(ChunkGrid):
    """A chunk grid where every chunk has the same shape.

    `configuration.chunk_shape` gives that shape, one entry per
    dimension of the array. This is the only chunk grid Zarr v2 and
    v1 can represent, so it is required for conversion to those
    versions.
    """

    name: tx.Literal["regular"]
    """Always ``"regular"``."""
    configuration: RegularChunkGridConfig
    """The chunk grid's parameters."""


@autofrozen(extra_items=False)
class RectilinearChunkGridConfig(TypedConfig):
    """Holds the rectilinear chunk grid's parameters: the per-axis
    chunk shapes.
    """

    kind: tx.Literal["inline"]
    """Always ``"inline"``."""
    chunk_shapes: tz.Shape
    """The chunk shapes along the axis that varies."""


@register_subclass(name="rectilinear")
@autofrozen
class RectilinearChunkGrid(ChunkGrid):
    """A chunk grid whose chunk sizes vary along an axis.

    Unlike
    [RegularChunkGrid][abczarr.metadata.v3.array.RegularChunkGrid],
    which fixes one shape for every chunk, this grid's chunks need
    not all be the same size. It has no representation in Zarr v2 or
    v1.
    """

    name: tx.Literal["rectilinear"]
    """Always ``"rectilinear"``."""
    configuration: RectilinearChunkGridConfig
    """The chunk grid's parameters."""


# ----------------------------------------------------------------------
#   CHUNK KEY ENCODING
# ----------------------------------------------------------------------


@autofrozen(extra_items=tz.FrozenJson)
class ChunkKeyEncodingConfig(TypedConfig):
    """This class is the base for a chunk-key encoding's own
    configuration parameters."""


@autofrozen(extra_items=False)
class CommonChunkKeyEncodingConfig(ChunkKeyEncodingConfig):
    """Holds the parameter every chunk-key encoding shares: the
    separator joining a chunk index into its key.
    """

    separator: tz.DimensionSeparator = "/"
    """The character joining a chunk index into its store key."""


@autofrozen
class ChunkKeyEncoding(MustUnderstandExtension):
    """How a chunk's index maps to its key in the store.

    Use
    [the default encoding][abczarr.metadata.v3.array.DefaultChunkKeyEncoding]
    for v3's own scheme, or
    [V2ChunkKeyEncoding][abczarr.metadata.v3.array.V2ChunkKeyEncoding]
    to keep the key layout Zarr v2 uses. A v2 array converts to
    `V2ChunkKeyEncoding`, and a v3 array must use it to convert back
    to v2 or v1.
    """

    name: str
    """The name of the encoding, such as ``"default"`` or ``"v2"``."""
    configuration: ChunkKeyEncodingConfig
    """The encoding's own parameters."""

    # Construction dispatches to the right subclass by name through the
    # metadata registry (Metadata.__new__), so a "default"/"v2" name yields
    # a DefaultChunkKeyEncoding / V2ChunkKeyEncoding without a hand-written
    # __new__ here.


@autofrozen(field_transformer=update(separator={"default": "/"}))
class DefaultChunkKeyEncodingConfig(CommonChunkKeyEncodingConfig):
    """Holds the default chunk-key encoding's parameters: the
    separator, defaulting to ``"/"``."""


@register_subclass(name="default")
@autofrozen
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    """Zarr v3's own chunk-key layout.

    A chunk index like `(1, 2)` becomes the key `c/1/2`, joined by
    `configuration.separator` (`/` by default).
    """

    name: tx.Literal["default"]
    """Always ``"default"``."""
    configuration: DefaultChunkKeyEncodingConfig
    """The encoding's parameters."""


@autofrozen(field_transformer=update(separator={"default": "."}))
class V2ChunkKeyEncodingConfig(CommonChunkKeyEncodingConfig):
    """Holds the v2 chunk-key encoding's parameters: the separator,
    defaulting to ``"."``."""


@register_subclass(name="v2")
@autofrozen
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    """Zarr v2's chunk-key layout, usable from a v3 array.

    A chunk index like `(1, 2)` becomes the key `1.2`, joined by
    `configuration.separator` (`.` by default, v2's own default). A
    v3 array must use this encoding to convert to v2 or v1. A v2
    array converts to this encoding, not to
    [the default one][abczarr.metadata.v3.array.DefaultChunkKeyEncoding].
    """

    name: tx.Literal["v2"]
    """Always ``"v2"``."""
    configuration: V2ChunkKeyEncodingConfig
    """The encoding's parameters."""


# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


_AxisNames = tx.Tuple[tx.Optional[str], ...]

# A fill value is a scalar, or -- for a complex dtype -- the two-element
# ``[real, imag]`` array the Zarr v3 spec uses to encode a complex number
# (JSON has no complex literal). The authored ``array.schema`` allows both.
_ComplexFillValue = tx.Tuple[tz.BuiltinReal, tz.BuiltinReal]
_FillValue = tx.Union[tz.BuiltinNumber, _ComplexFillValue]


@register_subclass(zarr_format=3, node_type="array")
@autofrozen(kw_only=True, extra_items=ExtraField)
class ArrayMetadata(ArrayMetadataV3):
    """A Zarr v3 array's metadata: shape, dtype, chunking, codecs.

    Corresponds to the contents of `zarr.json` for an array node.
    `codecs` is an ordered pipeline: zero or more array-to-array
    codecs, exactly one array-to-bytes codec (a plain serializer, or
    a `ShardingCodec` that groups several chunks into one shard
    file), then zero or more bytes-to-bytes codecs such as a
    compressor.

    !!! example
        ```pycon
        >>> from abczarr.metadata import v1
        >>> meta = v1.ArrayMetadata.from_json({
        ...     "zarr_format": 1,
        ...     "shape": [10],
        ...     "chunks": [5],
        ...     "dtype": "<f8",
        ...     "compression": "zlib",
        ...     "compression_opts": {"level": 1},
        ...     "fill_value": 0,
        ...     "order": "C",
        ...     "attributes": {},
        ... })
        >>> v3_meta = meta.to_version(3)
        >>> v3_meta.chunk_grid.configuration.chunk_shape
        (5,)
        >>> [codec.name for codec in v3_meta.codecs]
        ['bytes', 'zlib']

        ```
    """

    # --- Required ----
    shape: tz.Shape
    """The array's shape, one entry per dimension."""
    data_type: DType
    """The array's Zarr v3 data type."""
    chunk_grid: ChunkGrid
    """How the array is divided into chunks."""
    chunk_key_encoding: ChunkKeyEncoding
    """How a chunk's index maps to its key in the store."""
    fill_value: tx.Optional[_FillValue] = field(eq=eq_safenan)
    """The value an unwritten element of the array reads as. A complex
    fill value is a two-element ``(real, imag)`` pair."""
    codecs: tx.Tuple[Codec, ...]
    """The codec pipeline applied to each chunk: zero or more
    array-to-array codecs, exactly one array-to-bytes codec, then zero
    or more bytes-to-bytes codecs."""

    # --- Optional ----
    attributes: tz.FrozenJsonDict
    """The array's user-defined attributes."""
    dimension_names: tx.Optional[_AxisNames]
    """An optional name for each dimension of the array. An entry may
    be `None` when that dimension is unnamed."""
    storage_transformers: tx.Tuple[tz.FrozenJsonDict, ...]
    """Extra transformations a store applies to the array's chunks,
    beyond the codec pipeline."""

    # --- Serialization ---

    def to_json(self) -> tz.JsonDict:
        """Serialize this metadata to the contents of `zarr.json`.

        `dimension_names` and `storage_transformers` are omitted from
        the document when they carry nothing, instead of written as
        `null` or an empty list, matching how the Zarr v3
        specification represents an array with neither.

        Returns
        -------
        dict
            The JSON-compatible representation of this metadata.
        """
        data = super().to_json()
        if data.get("dimension_names") is None:
            data.pop("dimension_names", None)
        if not data.get("storage_transformers"):
            data.pop("storage_transformers", None)
        return data

    # --- Conversion ---

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: base.ConversionPolicy = "lossy",
    ) -> base.ArrayMetadata:
        """Convert this array's metadata to another Zarr version.

        Requires a
        [RegularChunkGrid][abczarr.metadata.v3.array.RegularChunkGrid]
        -- v2 and v1 have no other kind. Sharding is unwrapped into
        its inner codecs (subject to `policy`, since the sharding
        structure itself is then lost), the codec pipeline is split
        back into v2's filters, byte order and compressor around its
        one array-to-bytes codec, and an `order` other than ``"C"``
        or more than one bytes-to-bytes codec is likewise subject to
        `policy`.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.
        policy : ConversionPolicy
            How to treat a field the target version can't hold.

        Returns
        -------
        ArrayMetadata
            Equivalent metadata for `version`. Converting to 3
            returns this object unchanged.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3, or if `chunk_grid` is not
            a `RegularChunkGrid`.
        UnsupportedConversion
            If `policy` is ``"strict"`` and a field cannot be
            represented in `version`.
        """
        if version == 1:
            # route through v2 -- v1 and v2 share the numcodecs model
            return _to_v2(self, policy).to_version(1, policy)
        if version == 2:
            return _to_v2(self, policy)
        if version == 3:
            return self
        else:
            raise ValueError(f"Unsupported version: {version}")

    def required_features(self) -> tx.FrozenSet[str]:
        """Report the features a driver needs to read or write this array.

        Returns
        -------
        frozenset of str
            One key each for the chunk grid, the chunk-key encoding
            and the data type, plus one per codec in `codecs`,
            including the codecs nested inside a `ShardingCodec`, and
            one per named storage transformer.
        """
        feats = {
            feature_key("v3", "chunk_grid", self.chunk_grid.name),
            feature_key(
                "v3", "chunk_key_encoding", self.chunk_key_encoding.name
            ),
            feature_key("v3", "data_type", self.data_type.name),
        }
        for codec in self.codecs:
            _collect_codec_features(codec, feats)
        for transformer in self.storage_transformers:
            name = transformer.get("name")
            if name:
                feats.add(feature_key("v3", "storage_transformer", name))
        return frozenset(feats)


# ----------------------------------------------------------------------
#   FEATURES
# ----------------------------------------------------------------------


def _collect_codec_features(codec: Codec, feats: tx.Set[str]) -> None:
    """Add `codec`'s feature key to `feats`, recursing into a sharding
    codec's inner and index codecs so a nested codec is named too."""
    name = getattr(codec, "name", None)
    if name:
        feats.add(feature_key("v3", "codec", name))
    config = getattr(codec, "configuration", None)
    for attr in ("codecs", "index_codecs"):
        for inner in getattr(config, attr, None) or ():
            _collect_codec_features(inner, feats)


# ----------------------------------------------------------------------
#   CONVERTERS
# ----------------------------------------------------------------------


# The v3 variable-length data types and the v2 vlen filter each maps to. A
# vlen filter on an |O array is what tags it as variable-length string
# ("vlen-utf8") or bytes ("vlen-bytes"), which numpy object alone cannot hold.
_VLEN_DTYPE_TO_FILTER = {"string": "vlen-utf8", "bytes": "vlen-bytes"}


def _pop_next(
    seq: tx.List[tx.Type[Codec]], cls: tx.Type[Codec]
) -> tx.Optional[Codec]:
    """Remove and return the first instance of `cls` in `seq`, or `None`
    if `seq` holds none."""
    for i, c in enumerate(seq):
        if isinstance(c, cls):
            return seq.pop(i)
    return None


def _is_serializer(codec: Codec) -> bool:
    """Report whether `codec` is the array-to-bytes step of the v3
    pipeline."""
    return isinstance(codec, BytesCodec) or getattr(codec, "name", None) == (
        "bytes"
    )


def _to_v2(
    self: ArrayMetadata, policy: base.ConversionPolicy = "lossy"
) -> base.ArrayMetadata:
    """Convert `self` to a v2 `ArrayMetadata`, applying `policy` to
    whatever v2 cannot represent."""
    from abczarr.metadata import v2

    if self.chunk_grid.name != "regular":
        # A non-regular grid (such as rectilinear) has no v2/v1 form. Unlike
        # a dropped field, it leaves no chunk shape to build a valid array
        # from, so the conversion cannot proceed under any policy. This is a
        # documented limitation, not a policy-governed loss, so it raises a
        # named error instead of a bare ValueError.
        raise UnsupportedConversion("chunk_grid", 2)
    chunk_grid = tx.cast(RegularChunkGrid, self.chunk_grid)
    chunk_shape = chunk_grid.configuration.chunk_shape

    # Chunk-key encoding: only a V2ChunkKeyEncoding maps onto v2's flat key
    # layout. A "default" (or any non-v2) encoding prefixes chunk keys with
    # "c/", which v2 cannot express, so keeping only its separator produces a
    # .zarray that addresses chunks where the data is not. That is a
    # policy-governed loss: reported (and raised under "strict"), while the
    # lenient path keeps today's separator-only behaviour.
    if not isinstance(self.chunk_key_encoding, V2ChunkKeyEncoding):
        base._report_loss(policy, "chunk_key_encoding", 2)

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
        base._report_loss(policy, "sharding", 2)
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
    filters = [v2.Filter.from_json(c.to_version(2).to_json()) for c in pre]

    # A variable-length string/bytes array is numpy object (|O) in v2 with a
    # vlen filter (vlen-utf8 / vlen-bytes). The v3 pipeline carries that codec
    # as its array->bytes serializer, which flows into the filters above; guard
    # against a pipeline that omitted it so the string / bytes tag numpy object
    # drops is never lost.
    vlen_id = _VLEN_DTYPE_TO_FILTER.get(getattr(self.data_type, "name", None))
    if vlen_id and not any(getattr(f, "id", None) == vlen_id for f in filters):
        filters.insert(0, v2.Filter.from_json({"id": vlen_id}))

    # v2 holds a single bytes->bytes compressor; any extra is a loss
    compressor = None
    if post:
        compressor = post[0].to_version(2)
        if len(post) > 1:
            base._report_loss(policy, "codecs", 2)

    # v2 folds the byte order back into the dtype
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
        dimension_separator=separator,
        attributes=self.attributes,
    )
