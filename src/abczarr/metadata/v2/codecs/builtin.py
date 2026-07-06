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
from abczarr.schemas.v2 import codecs

# locals
from .base import CodecImpl


@register_subclass(id="blosc")
@autofrozen
class BloscCodec(CodecImpl):
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
            return v3.BloscCodec(
                cname=self.cname,
                clevel=self.clevel,
                shuffle=SHUFFLE[self.shuffle],
                blocksize=self.blocksize,
                typesize=self.typesize,
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
    # type aliases
    CompressionLevel: tx.ClassVar = codecs.GzipCompressionLevel

    # attributes
    id: tx.Literal["gzip"]
    level: CompressionLevel = 5
