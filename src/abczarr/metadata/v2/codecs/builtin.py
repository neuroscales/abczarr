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
from abczarr._core.attrs import autofrozen
from abczarr.schemas.v2 import codecs

# locals
from ...base import register_subclass
from .base import CodecImpl


@register_subclass(id="blosc")
@autofrozen
class BloscCodec(CodecImpl):
    # type aliases
    CodecName: tx.ClassVar[tx.TypeAlias] = codecs.BloscCodecName
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.BloscCompressionLevel
    Shuffle: tx.ClassVar[tx.TypeAlias] = codecs.BloscShuffle

    # attributes
    id: tx.Literal["blosc"]
    cname: CodecName = "lz4"
    clevel: CompressionLevel = 5
    shuffle: Shuffle = "shuffle"
    blocksize: int = 0
    typesize: tx.Optional[int] = None


@register_subclass(id="gzip")
@autofrozen
class GzipCodec(CodecImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.GzipCompressionLevel

    # attributes
    id: tx.Literal["gzip"]
    level: CompressionLevel = 5
