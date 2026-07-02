"""
This module contains the built-in codecs that all zarr implementations
SHOULD support, according to the specification.
"""
__all__ = [
    "BloscCodec",
    "BytesCodec",
    "CRC32CCodec",
    "GzipCodec",
    "ShardingCodec",
    "TransposeCodec",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autofrozen
from abczarr.schemas.v3 import codecs

# metadata
from abczarr.metadata.base import Metadata, register_subclass
from .base import (
    Codec, CodecConfigImpl,
    BytesToBytesCodec, ArrayToArrayCodec, ArrayToBytesCodec, CompressorCodec
)


@autofrozen
class BloscConfig(CodecConfigImpl):
    cname: codecs.BloscCodecName = "lz4"
    clevel: codecs.BloscCompressionLevel = 5
    shuffle: codecs.BloscShuffle = "shuffle"
    blocksize: int = 0
    typesize: tx.Optional[int] = None

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 3:
            return self
        if version == 2:
            from abczarr.metadata import v2
            SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")
            return v2.BloscCodec(
                cname=self.cname,
                clevel=self.clevel,
                shuffle=SHUFFLE.index(self.shuffle),
                blocksize=self.blocksize,
                typesize=self.typesize,
            )
        if version == 1:
            from abczarr.metadata import v1
            SHUFFLE = ("noshuffle", "shuffle", "bitshuffle")
            return v1.BloscCodecOptions(
                cname=self.cname,
                clevel=self.clevel,
                shuffle=SHUFFLE.index(self.shuffle),
                blocksize=self.blocksize,
            )
        else:
            raise ValueError(f"Unsupported version: {version}")


@register_subclass(name="blosc")
@autofrozen
class BloscCodec(CompressorCodec):
    name: tx.Literal["blosc"]
    configuration: BloscConfig

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 3:
            return self
        return self.configuration.to_version(version)


@autofrozen
class BytesConfig(CodecConfigImpl):
    endian: tx.Optional[tx.Literal["big", "little"]]


@register_subclass(name="bytes")
@autofrozen
class BytesCodec(ArrayToBytesCodec):
    name: tx.Literal["bytes"]
    configuration: BytesConfig


@autofrozen
class CRC32CConfig(CodecConfigImpl):
    ...


@register_subclass(name="crc32c")
@autofrozen
class CRC32CCodec(BytesToBytesCodec):
    name: tx.Literal["crc32c"]
    configuration: CRC32CConfig


@autofrozen
class GzipConfig(CodecConfigImpl):
    level: codecs.GzipCompressionLevel = 5


@register_subclass(name="gzip")
@autofrozen
class GzipCodec(CompressorCodec):
    name: tx.Literal["gzip"]
    configuration: GzipConfig


@autofrozen
class ShardingConfig(CodecConfigImpl):
    chunk_shape: tz.Shape
    codecs: tx.Tuple[Codec, ...]
    index_codecs: tx.Tuple[Codec, ...]
    index_location: tx.Literal["start", "end"] = "end"


@register_subclass(name="sharding_indexed")
@autofrozen
class ShardingCodec(ArrayToArrayCodec):
    name: tx.Literal["sharding_indexed"]
    configuration: ShardingConfig


@autofrozen
class TransposeConfig(CodecConfigImpl):
    order: tx.Tuple[int, ...]


@register_subclass(name="transpose")
@autofrozen
class TransposeCodec(ArrayToArrayCodec):
    name: tx.Literal["transpose"]
    configuration: TransposeConfig
