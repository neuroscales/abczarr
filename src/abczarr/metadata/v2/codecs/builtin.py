"""
This module contains the built-in codecs that all zarr implementations
SHOULD support, according to the specification.
"""
__all__ = [
    "BloscCodec",
    "GzipCodec",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr._core.metadata import Metadata, register_subclass

from . import aliases as codecs

# locals
from .base import CodecImpl


@register_subclass(id="blosc")
@autofrozen
class BloscCodec(CodecImpl):
    """Compresses data with Blosc, a meta-compressor that shuffles bytes
    and then applies an inner compressor.

    Blosc groups same-typed bytes together before handing them to
    ``cname`` (one of blosclz, lz4, lz4hc, snappy, zlib or zstd), and
    compresses the result in blocks so multiple threads can be used.
    """

    # type aliases
    CodecName: tx.ClassVar = codecs.BloscCodecName
    CompressionLevel: tx.ClassVar = codecs.BloscCompressionLevel
    Shuffle: tx.ClassVar = codecs.BloscShuffle

    # attributes
    id: tx.Literal["blosc"]
    cname: CodecName = "lz4"
    clevel: CompressionLevel = 5
    shuffle: Shuffle = 1
    blocksize: int = 0
    typesize: tx.Optional[int] = None

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata import v3
            SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")
            # v3 codecs are {name, configuration}: the settings go inside the
            # configuration, not as top-level kwargs.
            return v3.BloscCodec(
                configuration={
                    "cname": self.cname,
                    "clevel": self.clevel,
                    "shuffle": SHUFFLE[self.shuffle],
                    "blocksize": self.blocksize,
                    "typesize": self.typesize,
                },
            )
        if version == 1:
            from abczarr.metadata import v1
            SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")
            return v1.BloscCodecOptions(
                cname=self.cname,
                clevel=self.clevel,
                shuffle=self.shuffle,
                blocksize=self.blocksize,
            )
        else:
            raise ValueError(f"Unsupported version: {version}")


@register_subclass(id="gzip")
@autofrozen
class GzipCodec(CodecImpl):
    """Applies DEFLATE compression (gzip) at a configurable compression
    level."""

    # type aliases
    CompressionLevel: tx.ClassVar = codecs.GzipCompressionLevel

    # attributes
    id: tx.Literal["gzip"]
    level: CompressionLevel = 5
