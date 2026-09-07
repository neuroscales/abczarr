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

from . import aliases as codecs

# locals
from .base import CodecImpl


@register_subclass(id="bz2")
@autofrozen
class Bz2Codec(CodecImpl):
    """Bzip2 compression, at a configurable compression level."""

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.Bz2CompressionLevel

    # attributes
    id: tx.Literal["bz2"]
    level: CompressionLevel


@register_subclass(id="lzma")
@autofrozen
class LZMACodec(CodecImpl):
    """LZMA compression (as used by ``.xz``/``.7z``), via Python's ``lzma``.

    ``filters`` carries the raw ``lzma`` filter chain when one is needed
    (for example to select a custom dictionary size); most uses only need
    ``format``, ``check`` and ``preset``.
    """

    # type aliases
    Format: tx.ClassVar = codecs.LZMAFormat
    Check: tx.ClassVar = codecs.LZMACheck
    CompressionLevel: tx.ClassVar = codecs.LZMACompressionLevel

    # attributes
    id: tx.Literal["lzma"]
    format: Format
    check: Check
    preset: CompressionLevel
    filters: tx.Tuple[tz.FrozenJsonDict, ...]


@register_subclass(id="lz4")
@autofrozen
class LZ4Codec(CodecImpl):
    """LZ4 compression: very fast, at the cost of a lower ratio.

    ``acceleration`` trades ratio for speed further: higher values
    compress faster and worse.
    """

    id: tx.Literal["lz4"]
    acceleration: int


@register_subclass(id="pcodec")
@autofrozen
class PCodec(CodecImpl):
    """Pcodec (``pco``): a compressor for numeric arrays.

    Models each chunk's distribution directly, optionally after delta
    encoding, rather than treating it as a byte stream -- it typically
    beats general-purpose compressors on numeric data while staying
    lossless.
    """

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.PCodecCompressionLevel
    Mode: tx.ClassVar = codecs.PCodecMode
    Delta: tx.ClassVar = codecs.PCodecDelta
    Paging: tx.ClassVar = codecs.PCodecPaging
    DeltaOrder: tx.ClassVar = codecs.PCodecDeltaOrder

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
    """ZFP compression for floating-point arrays, via the ``zfpy`` bindings.

    ``mode`` selects which of ``tolerance``, ``rate`` and ``precision``
    governs the accuracy/size trade-off; compression is lossy except in
    the fixed-accuracy mode with a tolerance of zero.
    """

    # type aliases
    Mode: tx.ClassVar = codecs.ZFPYMode

    # attributes
    id: tx.Literal["zfpy"]
    mode: Mode
    tolerance: float
    rate: int
    precision: int
    compression_kwargs: tz.FrozenJsonDict


@register_subclass(id="zlib")
@autofrozen
class ZlibCodec(CodecImpl):
    """Zlib compression: DEFLATE, at a configurable compression level."""

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.ZlibCompressionLevel

    # attributes
    id: tx.Literal["zlib"]
    level: CompressionLevel


@register_subclass(id="zstd")
@autofrozen
class ZstdCodec(CodecImpl):
    """Zstandard compression, at a configurable compression level."""

    # attributes
    id: tx.Literal["zstd"]
    level: int
