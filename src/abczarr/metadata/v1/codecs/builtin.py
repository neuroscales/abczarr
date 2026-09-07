"""The built-in Zarr v1 codecs every implementation should support.

Corresponds to the compressors the Zarr v1 specification names as
required: blosc and gzip.
"""
__all__ = [
    "BloscCodecOptions",
    "GzipCodecOptions",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autofrozen
from abczarr.metadata.base import register_subclass

from . import aliases as codecs

# locals
from .base import CodecOptionsImpl


@register_subclass(id="blosc")
@autofrozen
class BloscCodecOptions(CodecOptionsImpl):
    """Options for Blosc, a meta-compressor that shuffles bytes and
    then applies an inner compressor.
    """

    # type aliases
    CodecName: tx.ClassVar = codecs.BloscCodecName
    CompressionLevel: tx.ClassVar = codecs.BloscCompressionLevel
    Shuffle: tx.ClassVar = codecs.BloscShuffle

    # attributes
    cname: CodecName = "lz4"
    """The inner compressor Blosc applies: one of ``"blosclz"``,
    ``"lz4"``, ``"lz4hc"``, ``"snappy"``, ``"zlib"`` or ``"zstd"``."""
    clevel: CompressionLevel = 5
    """The compression level, from 0 to 9."""
    shuffle: Shuffle = 1
    """The byte-shuffle filter applied before compression: ``0`` for
    none, ``1`` for byte shuffle, ``2`` for bit shuffle, or ``-1`` to
    let Blosc choose automatically."""
    blocksize: int = 0
    """The block size Blosc compresses in, in bytes. ``0`` lets Blosc
    choose automatically."""
    typesize: tx.Optional[int] = None
    """The size, in bytes, of the array's element type. Blosc uses this
    to group same-position bytes together when shuffling."""

    # classvar
    id: tx.ClassVar[tx.Literal["blosc"]] = "blosc"


@register_subclass(id="gzip")
@autofrozen
class GzipCodecOptions(CodecOptionsImpl):
    """Options for DEFLATE compression (gzip)."""

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.GzipCompressionLevel

    # attributes
    level: CompressionLevel = 5
    """The compression level, from 0 to 9."""

    # classvar
    id: tx.ClassVar[tx.Literal["gzip"]] = "gzip"
