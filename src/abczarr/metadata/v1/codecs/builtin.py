"""
This module contains the built-in codecs that all zarr implementations
SHOULD support, according to the specification.
"""
__all__ = [
    "BloscCodecOptions",
    "GzipCodecOptions",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autofrozen
from abczarr.metadata.base import register_subclass
from abczarr.schemas.v1 import codecs

# locals
from .base import CodecOptionsImpl


@register_subclass(id="blosc")
@autofrozen
class BloscCodecOptions(CodecOptionsImpl):
    # type aliases
    CodecName: tx.ClassVar[tx.TypeAlias] = codecs.BloscCodecName
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.BloscCompressionLevel
    Shuffle: tx.ClassVar[tx.TypeAlias] = codecs.BloscShuffle

    # attributes
    cname: CodecName = "lz4"
    clevel: CompressionLevel = 5
    shuffle: Shuffle = "shuffle"
    blocksize: int = 0
    typesize: tx.Optional[int] = None


@register_subclass(id="gzip")
@autofrozen
class GzipCodecOptions(CodecOptionsImpl):
    # type aliases
    CompressionLevel: tx.ClassVar[tx.TypeAlias] = codecs.GzipCompressionLevel

    # attributes
    id: tx.Literal["gzip"]
    level: CompressionLevel = 5
