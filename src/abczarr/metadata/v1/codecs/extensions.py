__all__ = [
    "Bz2CodecOptions",
    "LZMACodecOptions",
    "LZ4CodecOptions",
    "PCodecOptions",
    "ZFPYCodecOptions",
    "ZlibCodecOptions",
    "ZstdCodecOptions",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr.metadata.base import register_subclass
from abczarr.schemas.v2 import codecs

# locals
from .base import CodecOptionsImpl


@register_subclass(id="bz2")
@autofrozen
class Bz2CodecOptions(CodecOptionsImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.Bz2CompressionLevel

    # attributes
    level: CompressionLevel

    # classvar
    id: tx.ClassVar[tx.Literal["bz2"]] = "bz2"


@register_subclass(id="lzma")
@autofrozen
class LZMACodecOptions(CodecOptionsImpl):
    # type aliases
    Format: tx.ClassVar[tx.TypeAlias] = codecs.LZMAFormat
    Check: tx.ClassVar[tx.TypeAlias] = codecs.LZMACheck
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.LZMACompressionLevel

    # attributes
    format: Format
    check: Check
    preset: CompressionLevel
    filters: tx.Tuple[tz.FrozenJSONDict, ...]

    # classvar
    id: tx.ClassVar[tx.Literal["lzma"]] = "lzma"


@register_subclass(id="lz4")
@autofrozen
class LZ4CodecOptions(CodecOptionsImpl):
    acceleration: int


@register_subclass(id="pcodec")
@autofrozen
class PCodecOptions(CodecOptionsImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.PCodecCompressionLevel
    Mode: tx.ClassVar[tx.TypeAlias] = codecs.PCodecMode
    Delta: tx.ClassVar[tx.TypeAlias] = codecs.PCodecDelta
    Paging: tx.ClassVar[tx.TypeAlias] = codecs.PCodecPaging
    DeltaOrder: tx.ClassVar[tx.TypeAlias] = codecs.PCodecDeltaOrder

    # attributes
    level: CompressionLevel
    mode_spec: Mode
    delta_spec: Delta
    paging_spec: Paging
    delta_encoding_order: DeltaOrder
    equal_pages_up_to: int

    # classvar
    id: tx.ClassVar[tx.Literal["pcodec"]] = "pcodec"


@register_subclass(id="zfpy")
@autofrozen
class ZFPYCodecOptions(CodecOptionsImpl):
    # type aliases
    Mode: tx.ClassVar[tx.TypeAlias] = codecs.ZFPYMode

    # attributes
    mode: Mode
    tolerance: float
    rate: int
    precision: int
    compression_kwargs: tz.FrozenJSONDict

    # classvar
    id: tx.ClassVar[tx.Literal["zfpy"]] = "zfpy"


@register_subclass(id="zlib")
@autofrozen
class ZlibCodecOptions(CodecOptionsImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.ZlibCompressionLevel

    # attributes
    level: CompressionLevel

    # classvar
    id: tx.ClassVar[tx.Literal["zlib"]] = "zlib"


@register_subclass(id="zstd")
@autofrozen
class ZstdCodecOptions(CodecOptionsImpl):
    # attributes
    level: int

    # classvar
    id: tx.ClassVar[tx.Literal["zstd"]] = "zstd"
