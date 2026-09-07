__all__ = [
    "BitroundCodec",
    "CastValueCodec",
    "ConditionalCodec",
    "N5DefaultCodec",
    "PackBitsCodec",
    "ScaleOffsetCodec",
    "VLenBytesCodec",
    "VLenUTF8Codec",
    "ReshapeCodec",
    "ZfpCodec",
    "ZstdCodec",
]

# stdlib
import re

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto import autofrozen

# locals
from abczarr.metadata.base import Metadata, register_subclass

from .base import (
    ArrayToArrayCodec,
    ArrayToBytesCodec,
    Codec,
    CodecConfigImpl,
    CompressorCodec,
)
from .builtin import BytesCodec, TransposeCodec


@autofrozen
class BitroundConfig(CodecConfigImpl):
    """Bitround codec parameters: how many mantissa bits to keep."""

    keepbits: int = 1


@register_subclass(name=re.compile(r"(?:bitround|numcodecs\.bitround)"))
@autofrozen
class BitroundCodec(ArrayToArrayCodec):
    """Rounds a float's mantissa to ``keepbits`` bits, zeroing the rest.

    A lossy transform that improves the compressibility of floating-point
    data by discarding low-order precision the data does not need.
    """

    name: tx.Literal["bitround", "numcodecs.bitround"]
    configuration: BitroundConfig


_ScalarMapItem = tx.Union[tz.JsonScalar, str]
_ScalarMap = tx.Tuple[tx.Tuple[_ScalarMapItem, _ScalarMapItem], ...]


@autofrozen
class ScalarMap(Metadata):
    """An explicit value-to-value mapping used by a cast-value codec.

    ``encode`` and ``decode`` each list ``(from, to)`` pairs, mapping
    individual values that fall outside the ordinary numeric cast --
    for example a sentinel or a small set of categorical codes.
    """

    encode: _ScalarMap
    decode: _ScalarMap


@autofrozen
class CastValueConfig(CodecConfigImpl):
    """Cast-value codec parameters: target type, rounding, value mapping."""

    data_type: np.dtype
    rounding: tx.Literal[
        "nearest-even",
        "towards-zero",
        "towards-positive",
        "towards-negative",
        "nearest-away",
    ] = "nearest-even"
    out_of_range: tx.Optional[tx.Literal["clamp", "wrap"]] = None
    scalar_map: ScalarMap


@register_subclass(name="cast_value")
@autofrozen
class CastValueCodec(ArrayToArrayCodec):
    """Casts an array's values to ``data_type``, and back on decode.

    ``rounding`` and ``out_of_range`` control how a value that does not
    fit the target type exactly is rounded and clamped or wrapped;
    ``scalar_map`` maps any values that need an explicit, non-numeric
    mapping instead.
    """

    name: tx.Literal["cast_value"]
    configuration: CastValueConfig


@autofrozen
class ConditionalConfig(CodecConfigImpl):
    """Conditional codec parameters: the candidate codecs to choose among."""

    codecs: tx.Tuple[Codec, ...]


@register_subclass(name="conditional")
@autofrozen
class ConditionalCodec(Codec):
    """Selects one of several candidate codecs based on the array applied to.

    Lets a single pipeline entry vary by a property of the array, such
    as its data type, instead of naming one codec unconditionally.
    """

    name: tx.Literal["conditional"]
    configuration: ConditionalConfig


class N5DefaultCodecList(list):
    """The fixed codec chain of an ``n5_default`` codec.

    A transpose codec, then a big-endian bytes codec, then any number of
    trailing codecs. Each element is converted to the concrete codec type
    on construction, so the stored elements are codec objects rather than
    the raw dicts they were read from.

    Parameters
    ----------
    codecs : iterable
        The codecs to store, as codec objects or as JSON-like mappings.
        The first is read as a transpose codec, the second as a big-endian
        bytes codec, and the rest as arbitrary codecs.

    Raises
    ------
    ValueError
        If fewer than two codecs are given, or the second codec is
        little-endian.
    """

    def __init__(self, codecs: tx.Iterable[Codec]) -> None:
        codecs = list(codecs)
        if len(codecs) < 2:
            raise ValueError(
                f"N5DefaultCodecList must have at least 2 codecs, "
                f"got {len(codecs)}"
            )
        first, second, *rest = codecs
        first = TransposeCodec(**first)
        second = BytesCodec(**second)
        if second.configuration.endian == "little":
            raise ValueError(
                f"N5DefaultCodecList second codec must be big-endian, "
                f"got {second.configuration.endian}"
            )
        rest = [Codec(**c) for c in rest]
        super().__init__([first, second, *rest])


@autofrozen
class N5DefaultConfig(CodecConfigImpl):
    """N5-default codec parameters: its fixed codec chain."""

    codecs: N5DefaultCodecList


@register_subclass(name="n5_default")
@autofrozen
class N5DefaultCodec(Codec):
    """The codec chain an N5 array uses by default, for interop with N5.

    Fixed to a transpose codec followed by a big-endian bytes codec,
    with any number of trailing codecs, matching how the N5 format lays
    out a chunk.
    """

    name: tx.Literal["n5_default"]
    configuration: N5DefaultConfig


@autofrozen
class PackBitsConfig(CodecConfigImpl):
    """Packbits codec parameters: padding placement and used bit range."""

    padding_encoding: tx.Literal["first_byte", "last_byte", "none"] = "none"
    first_bit: tx.Optional[int]
    last_bit: tx.Optional[int]


@register_subclass(name="packbits")
@autofrozen
class PackBitsCodec(ArrayToBytesCodec):
    """Packs a boolean array down to one bit per element for storage."""

    name: tx.Literal["packbits"]
    configuration: PackBitsConfig


@autofrozen
class ScaleOffsetConfig(CodecConfigImpl):
    """Scale-offset codec parameters: the scale and offset to apply."""

    offset: tz.JsonNumber
    scale: tz.JsonNumber


@register_subclass(name="scale_offset")
@autofrozen
class ScaleOffsetCodec(ArrayToArrayCodec):
    """Quantizes values as ``(value - offset) * scale``, and reverses it back.

    A lossy transform useful for storing a bounded floating-point range
    in fewer bits.
    """

    name: tx.Literal["scale_offset"]
    configuration: ScaleOffsetConfig


@register_subclass(name="vlen-bytes")
@autofrozen
class VLenBytesCodec(ArrayToBytesCodec):
    """Serializes an array of variable-length byte strings to bytes."""

    name: tx.Literal["vlen-bytes"]


@register_subclass(name="vlen-utf8")
@autofrozen
class VLenUTF8Codec(ArrayToBytesCodec):
    """Serializes an array of variable-length UTF-8 strings to bytes."""

    name: tx.Literal["vlen-utf8"]


#: One axis of a reshape target: a size (with -1 for "the rest"), or a group
#: of sizes to split that axis into.
_ReshapeAxis = tx.Union[int, tx.Tuple[int, ...]]


@autofrozen
class ReshapeConfig(CodecConfigImpl):
    """Reshape codec parameters: the target shape."""

    shape: tx.Tuple[_ReshapeAxis, ...]


@register_subclass(name="reshape")
@autofrozen
class ReshapeCodec(ArrayToArrayCodec):
    """Reshapes an array to ``shape`` before the rest of the pipeline.

    Reversed on decode, so the array's logical shape is unchanged; an
    axis size of ``-1`` stands for whatever size makes the reshape fit,
    and a group of sizes splits that axis into several.
    """

    name: tx.Literal["reshape"]
    configuration: ReshapeConfig


#: zfp's five modes; each carries only its own parameters.
_ZfpMode = tx.Literal[
    "reversible", "expert", "fixed_accuracy", "fixed_rate", "fixed_precision"
]


@autofrozen
class ZfpConfig(CodecConfigImpl):
    """ZFP codec parameters: the mode, and that mode's own parameters.

    Only the parameters that belong to ``mode`` apply: ``expert`` takes
    ``minbits``/``maxbits``/``maxprec``/``minexp``, ``fixed_accuracy``
    takes ``tolerance``, ``fixed_rate`` takes ``rate``,
    ``fixed_precision`` takes ``precision``, and ``reversible`` takes
    none.
    """

    # zfp picks a mode, and each mode carries only its own parameters: expert
    # takes minbits/maxbits/maxprec/minexp; fixed_accuracy takes tolerance;
    # fixed_rate takes rate; fixed_precision takes precision; reversible takes
    # none. The parameters of the other modes stay None and are omitted on
    # serialization, so each mode round-trips exactly as the spec writes it.
    mode: _ZfpMode
    minbits: tx.Optional[int] = None
    maxbits: tx.Optional[int] = None
    maxprec: tx.Optional[int] = None
    minexp: tx.Optional[int] = None
    tolerance: tx.Optional[float] = None
    rate: tx.Optional[float] = None
    precision: tx.Optional[int] = None


@register_subclass(name="zfp")
@autofrozen
class ZfpCodec(ArrayToBytesCodec):
    """ZFP compression for floating-point arrays.

    ``configuration.mode`` selects the accuracy/size trade-off;
    compression is lossy except in ``reversible`` mode.
    """

    name: tx.Literal["zfp"]
    configuration: ZfpConfig


@autofrozen
class ZstdConfig(CodecConfigImpl):
    """Zstd codec parameters: compression level and optional checksum."""

    # The v3 zstd codec schema requires `level` (checksum is optional) and
    # declares no defaults, so an implementation picks its own. abczarr
    # defaults level to 0, matching zarr-python, and writes both fields.
    # Zstd levels run from -131072 to 22 (wider than the 0-9 the core
    # compressors use, and negative for the fast modes), so the type stays
    # a plain int.
    level: int = 0
    checksum: bool = False

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 3:
            return self
        if version == 2:
            from abczarr.metadata import v2
            # v2's numcodecs zstd carries only the level
            return v2.ZstdCodec(id="zstd", level=self.level)
        if version == 1:
            from abczarr.metadata import v1
            # v1's numcodecs zstd carries only the level
            return v1.ZstdCodecOptions(level=self.level)
        raise ValueError(f"Unsupported version: {version}")


@register_subclass(name="zstd")
@autofrozen
class ZstdCodec(CompressorCodec):
    """Zstandard compression, at a configurable compression level.

    ``configuration.checksum``, when set, appends a checksum of each
    frame for integrity.
    """

    name: tx.Literal["zstd"]
    configuration: ZstdConfig

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 3:
            return self
        return self.configuration.to_version(version)
