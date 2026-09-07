"""Extension codecs for Zarr v1, beyond the required blosc and gzip.

Each class holds the options of one numcodecs compressor as it is
carried in a Zarr v1 array's `compression_opts` field.
"""

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

from ...v2.codecs import aliases as codecs

# locals
from .base import CodecOptionsImpl


@register_subclass(id="bz2")
@autofrozen
class Bz2CodecOptions(CodecOptionsImpl):
    """Options for bzip2 compression."""

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.Bz2CompressionLevel

    # attributes
    level: CompressionLevel
    """The compression level, from 0 to 9."""

    # classvar
    id: tx.ClassVar[tx.Literal["bz2"]] = "bz2"


@register_subclass(id="lzma")
@autofrozen
class LZMACodecOptions(CodecOptionsImpl):
    """Options for LZMA compression, the compression used by
    `.xz`/`.7z` files, via Python's `lzma` module.

    `filters` carries the raw `lzma` filter chain for cases that need
    one, such as selecting a custom dictionary size. Most uses only
    need `format`, `check` and `preset`.
    """

    # type aliases
    Format: tx.ClassVar = codecs.LZMAFormat
    Check: tx.ClassVar = codecs.LZMACheck
    CompressionLevel: tx.ClassVar = codecs.LZMACompressionLevel

    # attributes
    format: Format
    """The LZMA container format to write."""
    check: Check
    """The integrity check to embed in the compressed stream."""
    preset: CompressionLevel
    """The compression level, from 0 to 9."""
    filters: tx.Tuple[tz.FrozenJsonDict, ...]
    """The raw `lzma` filter chain, when a custom one is needed."""

    # classvar
    id: tx.ClassVar[tx.Literal["lzma"]] = "lzma"


@register_subclass(id="lz4")
@autofrozen
class LZ4CodecOptions(CodecOptionsImpl):
    """Options for LZ4 compression, which is very fast at the cost of
    a lower compression ratio.
    """

    # attributes
    acceleration: int
    """A further speed-for-ratio trade-off. A higher value compresses
    faster and produces a lower ratio."""

    # classvar
    id: tx.ClassVar[tx.Literal["lz4"]] = "lz4"


@register_subclass(id="pcodec")
@autofrozen
class PCodecOptions(CodecOptionsImpl):
    """Options for Pcodec (`pco`), a compressor that models each
    chunk's value distribution directly instead of treating the
    chunk as a byte stream.
    """

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.PCodecCompressionLevel
    Mode: tx.ClassVar = codecs.PCodecMode
    Delta: tx.ClassVar = codecs.PCodecDelta
    Paging: tx.ClassVar = codecs.PCodecPaging
    DeltaOrder: tx.ClassVar = codecs.PCodecDeltaOrder

    # attributes
    level: CompressionLevel
    """The compression level."""
    mode_spec: Mode
    """How Pcodec chooses its numerical mode: ``"auto"`` or
    ``"classic"``."""
    delta_spec: Delta
    """How Pcodec chooses whether to delta-encode values before
    modeling them."""
    paging_spec: Paging
    """How Pcodec splits a chunk into pages for encoding."""
    delta_encoding_order: DeltaOrder
    """The order of delta encoding applied, when `delta_spec` selects
    one."""
    equal_pages_up_to: int
    """The maximum page size, in elements, when `paging_spec` is
    ``"equal_pages_up_to"``."""

    # classvar
    id: tx.ClassVar[tx.Literal["pcodec"]] = "pcodec"


@register_subclass(id="zfpy")
@autofrozen
class ZFPYCodecOptions(CodecOptionsImpl):
    """Options for ZFP compression of floating-point arrays, via the
    `zfpy` bindings.

    `mode` selects which of `tolerance`, `rate` and `precision`
    governs the accuracy/size trade-off. Compression is lossy,
    except in the fixed-accuracy mode with a tolerance of zero.
    """

    # type aliases
    Mode: tx.ClassVar = codecs.ZFPYMode

    # attributes
    mode: Mode
    """The accuracy mode ZFP compresses in."""
    tolerance: float
    """The maximum absolute error allowed, in fixed-accuracy mode."""
    rate: int
    """The number of bits per value, in fixed-rate mode."""
    precision: int
    """The number of bits of precision retained, in fixed-precision
    mode."""
    compression_kwargs: tz.FrozenJsonDict
    """Extra keyword arguments forwarded to the underlying `zfpy`
    call."""

    # classvar
    id: tx.ClassVar[tx.Literal["zfpy"]] = "zfpy"


@register_subclass(id="zlib")
@autofrozen
class ZlibCodecOptions(CodecOptionsImpl):
    """Options for DEFLATE compression (zlib)."""

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.ZlibCompressionLevel

    # attributes
    level: CompressionLevel
    """The compression level, from 0 to 9."""

    # classvar
    id: tx.ClassVar[tx.Literal["zlib"]] = "zlib"


@register_subclass(id="zstd")
@autofrozen
class ZstdCodecOptions(CodecOptionsImpl):
    """Options for Zstandard compression."""

    # attributes
    level: int
    """The compression level."""

    # classvar
    id: tx.ClassVar[tx.Literal["zstd"]] = "zstd"
