__all__ = [
    "Codec",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# locals
from .extensions import Extension, ExtensionWithConfig, Config

# typing
CompressionLevel = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class Codec(Extension):
    ...


class CodedWithConfig(ExtensionWithConfig, Codec):
    ...


class CodecConfig(Config):
    ...


BloscCodecName = tx.Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
BloscShuffle = tx.Literal["noshuffle", "shuffle", "bitshuffle"]
BloscCompressionLevel = CompressionLevel


class BloscConfig(CodecConfig):
    cname: tx.NotRequired[BloscCodecName]
    clevel: tx.NotRequired[BloscCompressionLevel]
    shuffle: tx.NotRequired[BloscShuffle]
    typesize: tx.NotRequired[int]
    blocksize: tx.NotRequired[int]


class BloscCodec(Codec):
    name: tx.Literal["blosc"]
    configuration: tx.NotRequired[BloscConfig]


class BytesConfig(CodecConfig):
    endian: tx.NotRequired[tx.Literal["big", "little"]]
    """Required for data types for which endianness is applicable."""


class BytesCodec(Codec):
    name: tx.Literal["bytes"]
    configuration: tx.NotRequired[BytesConfig]


class CRC32CConfig(CodecConfig):
    ...


class CRC32CCodec(Codec):
    name: tx.Literal["crc32c"]
    configuration: tx.NotRequired[CRC32CConfig]


GzipCompressionLevel = CompressionLevel


class GzipConfig(CodecConfig):
    level: tx.NotRequired[GzipCompressionLevel]


class GzipCodec(Codec):
    name: tx.Literal["gzip"]
    configuration: tx.NotRequired[GzipConfig]


class ShardingConfig(CodecConfig):
    chunk_shape: tz.Shape
    codecs: tz.BuiltinSequence[Codec]
    index_codecs: tz.BuiltinSequence[Codec]
    index_location: tx.Literal["start", "end"]


class ShardingCodec(CodedWithConfig):
    name: tx.Literal["sharding_indexed"]
    configuration: ShardingConfig


class TransposeConfig(CodecConfig):
    order: tz.BuiltinSequence[int]


class TransposeCodec(CodedWithConfig):
    name: tx.Literal["transpose"]
    configuration: TransposeConfig


ValidCodec = tx.Union[
    BloscCodec,
    BytesCodec,
    CRC32CCodec,
    GzipCodec,
    ShardingCodec,
    TransposeCodec,
]
