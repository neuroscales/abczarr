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
    """Options for bzip2 compression.

    Attributes
    ----------
    level : int
        The compression level, from 0 to 9.
    """

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.Bz2CompressionLevel

    # attributes
    level: CompressionLevel

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

    Attributes
    ----------
    format : int
        The LZMA container format to write.
    check : int
        The integrity check to embed in the compressed stream.
    preset : int
        The compression level, from 0 to 9.
    filters : tuple of dict
        The raw `lzma` filter chain, when a custom one is needed.
    """

    # type aliases
    Format: tx.ClassVar = codecs.LZMAFormat
    Check: tx.ClassVar = codecs.LZMACheck
    CompressionLevel: tx.ClassVar = codecs.LZMACompressionLevel

    # attributes
    format: Format
    check: Check
    preset: CompressionLevel
    filters: tx.Tuple[tz.FrozenJsonDict, ...]

    # classvar
    id: tx.ClassVar[tx.Literal["lzma"]] = "lzma"


@register_subclass(id="lz4")
@autofrozen
class LZ4CodecOptions(CodecOptionsImpl):
    """Options for LZ4 compression, which is very fast at the cost of
    a lower compression ratio.

    Attributes
    ----------
    acceleration : int
        A further speed-for-ratio trade-off. A higher value
        compresses faster and produces a lower ratio.
    """

    # attributes
    acceleration: int

    # classvar
    id: tx.ClassVar[tx.Literal["lz4"]] = "lz4"


@register_subclass(id="pcodec")
@autofrozen
class PCodecOptions(CodecOptionsImpl):
    """Options for Pcodec (`pco`), a compressor that models each
    chunk's value distribution directly instead of treating the
    chunk as a byte stream.

    Attributes
    ----------
    level : int
        The compression level.
    mode_spec : str
        How Pcodec chooses its numerical mode: ``"auto"`` or
        ``"classic"``.
    delta_spec : str
        How Pcodec chooses whether to delta-encode values before
        modeling them.
    paging_spec : str
        How Pcodec splits a chunk into pages for encoding.
    delta_encoding_order : int
        The order of delta encoding applied, when `delta_spec`
        selects one.
    equal_pages_up_to : int
        The maximum page size, in elements, when `paging_spec` is
        ``"equal_pages_up_to"``.
    """

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.PCodecCompressionLevel
    Mode: tx.ClassVar = codecs.PCodecMode
    Delta: tx.ClassVar = codecs.PCodecDelta
    Paging: tx.ClassVar = codecs.PCodecPaging
    DeltaOrder: tx.ClassVar = codecs.PCodecDeltaOrder

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
    """Options for ZFP compression of floating-point arrays, via the
    `zfpy` bindings.

    `mode` selects which of `tolerance`, `rate` and `precision`
    governs the accuracy/size trade-off. Compression is lossy,
    except in the fixed-accuracy mode with a tolerance of zero.

    Attributes
    ----------
    mode : int
        The accuracy mode ZFP compresses in.
    tolerance : float
        The maximum absolute error allowed, in fixed-accuracy mode.
    rate : int
        The number of bits per value, in fixed-rate mode.
    precision : int
        The number of bits of precision retained, in fixed-precision
        mode.
    compression_kwargs : dict
        Extra keyword arguments forwarded to the underlying `zfpy`
        call.
    """

    # type aliases
    Mode: tx.ClassVar = codecs.ZFPYMode

    # attributes
    mode: Mode
    tolerance: float
    rate: int
    precision: int
    compression_kwargs: tz.FrozenJsonDict

    # classvar
    id: tx.ClassVar[tx.Literal["zfpy"]] = "zfpy"


@register_subclass(id="zlib")
@autofrozen
class ZlibCodecOptions(CodecOptionsImpl):
    """Options for DEFLATE compression (zlib).

    Attributes
    ----------
    level : int
        The compression level, from 0 to 9.
    """

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.ZlibCompressionLevel

    # attributes
    level: CompressionLevel

    # classvar
    id: tx.ClassVar[tx.Literal["zlib"]] = "zlib"


@register_subclass(id="zstd")
@autofrozen
class ZstdCodecOptions(CodecOptionsImpl):
    """Options for Zstandard compression.

    Attributes
    ----------
    level : int
        The compression level.
    """

    # attributes
    level: int

    # classvar
    id: tx.ClassVar[tx.Literal["zstd"]] = "zstd"
