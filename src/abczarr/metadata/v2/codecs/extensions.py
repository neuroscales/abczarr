__all__ = [
    "Bz2Codec",
    "LZMACodec",
    "LZ4Codec",
    "PCodec",
    "ZFPYCodec",
    "ZlibCodec",
    "ZstdCodec",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr._core.metadata import register_subclass
from abczarr.schemas.v2 import codecs

# locals
from .base import CodecImpl


@register_subclass(id="bz2")
@autofrozen
class Bz2Codec(CodecImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.Bz2CompressionLevel

    # attributes
    id: tx.Literal["bz2"]
    level: CompressionLevel


@register_subclass(id="lzma")
@autofrozen
class LZMACodec(CodecImpl):
    # type aliases
    Format: tx.ClassVar[tx.TypeAlias] = codecs.LZMAFormat
    Check: tx.ClassVar[tx.TypeAlias] = codecs.LZMACheck
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.LZMACompressionLevel

    # attributes
    id: tx.Literal["lzma"]
    format: Format
    check: Check
    preset: CompressionLevel
    filters: tx.Tuple[tz.FrozenJSONDict, ...]


@register_subclass(id="lz4")
@autofrozen
class LZ4Codec(CodecImpl):
    id: tx.Literal["lz4"]
    acceleration: int


@register_subclass(id="pcodec")
@autofrozen
class PCodec(CodecImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.PCodecCompressionLevel
    Mode: tx.ClassVar[tx.TypeAlias] = codecs.PCodecMode
    Delta: tx.ClassVar[tx.TypeAlias] = codecs.PCodecDelta
    Paging: tx.ClassVar[tx.TypeAlias] = codecs.PCodecPaging
    DeltaOrder: tx.ClassVar[tx.TypeAlias] = codecs.PCodecDeltaOrder

    # attributes
    id: tx.Literal["pcodec"]
    level: CompressionLevel
    mode_spec: Mode
    delta_spec: Delta
    paging_spec: Paging
    delta_encoding_order: DeltaOrder
    equal_pages_up_to: int


@register_subclass(id="zfpy")
@autofrozen
class ZFPYCodec(CodecImpl):
    # type aliases
    Mode: tx.ClassVar[tx.TypeAlias] = codecs.ZFPYMode

    # attributes
    id: tx.Literal["zfpy"]
    mode: Mode
    tolerance: float
    rate: int
    precision: int
    compression_kwargs: tz.FrozenJSONDict


@register_subclass(id="zlib")
@autofrozen
class ZlibCodec(CodecImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.ZlibCompressionLevel

    # attributes
    id: tx.Literal["zlib"]
    level: CompressionLevel


@register_subclass(id="zstd")
@autofrozen
class ZstdCodec(CodecImpl):
    # attributes
    id: tx.Literal["zstd"]
    level: int
