__all__ = [
    "BitroundCodec",
    "CastValueCodec",
    "ConditionalCodec",
    "N5DefaultCodec",
    "PackBitsCodec",
    "ScaleOffsetCodec",
    "VLenBytesCodec",
    "VLenUTF8Codec",
]

# stdlib
import re

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autofrozen

# locals
from abczarr.metadata.base import register_subclass, Metadata
from .base import CodecConfigImpl, Codec, ArrayToArrayCodec, ArrayToBytesCodec
from .builtin import TransposeCodec, BytesCodec


@autofrozen
class BitroundConfig(CodecConfigImpl):
    keepbits: int = 1


@register_subclass(name=re.compile(r"(?:bitround|numcodecs\.bitround)"))
@autofrozen
class BitroundCodec(ArrayToArrayCodec):
    name: tx.Literal["bitround", "numcodecs.bitround"]
    configuration: BitroundConfig


_ScalarMapItem = tx.Union[tz.JSONScalar, str]
_ScalarMap = tx.Tuple[tx.Tuple[_ScalarMapItem, _ScalarMapItem], ...]


@autofrozen
class ScalarMap(Metadata):
    encode: _ScalarMap
    decode: _ScalarMap


@autofrozen
class CastValueConfig(CodecConfigImpl):
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
    name: tx.Literal["cast_value"]
    configuration: CastValueConfig


@autofrozen
class ConditionalConfig(CodecConfigImpl):
    codecs: tx.Tuple[Codec]


@register_subclass(name="conditional")
@autofrozen
class ConditionalCodec(Codec):
    name: tx.Literal["conditional"]
    configuration: ConditionalConfig


@autofrozen
class N5TransposeCodec(TransposeCodec):
    endian: tx.Literal["big"]


class N5DefaultCodecList(list):

    def __new__(cls, codecs: tx.Iterable[Codec]):
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
        return super().__new__(cls, [first, second, *rest])


@autofrozen
class N5DefaultConfig(CodecConfigImpl):
    codecs: N5DefaultCodecList


@register_subclass(name="n5_default")
@autofrozen
class N5DefaultCodec(Codec):
    name: tx.Literal["n5_default"]
    configuration: N5DefaultConfig


@autofrozen
class N5DefaultCodecListConfig(CodecConfigImpl):
    codecs: N5DefaultCodecList


@autofrozen
class PackBitsConfig(CodecConfigImpl):
    padding_encoding: tx.Literal["first_byte", "last_byte", "none"] = "none"
    first_bit: tx.Optional[int]
    last_bit: tx.Optional[int]


@register_subclass(name="packbits")
@autofrozen
class PackBitsCodec(ArrayToBytesCodec):
    name: tx.Literal["packbits"]
    configuration: PackBitsConfig


@autofrozen
class ScaleOffsetConfig(CodecConfigImpl):
    offset: tz.JSONNumber
    scale: tz.JSONNumber


@register_subclass(name="scale_offset")
@autofrozen
class ScaleOffsetCodec(ArrayToArrayCodec):
    name: tx.Literal["scale_offset"]
    configuration: ScaleOffsetConfig


@register_subclass(name="vlen-bytes")
@autofrozen
class VLenBytesCodec(ArrayToBytesCodec):
    name: tx.Literal["vlen-bytes"]


@register_subclass(name="vlen-utf8")
@autofrozen
class VLenUTF8Codec(ArrayToBytesCodec):
    name: tx.Literal["vlen-utf8"]
