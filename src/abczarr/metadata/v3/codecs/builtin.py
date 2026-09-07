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
from abczarr._core.auto import autofrozen

# metadata
from abczarr.metadata.base import Metadata, register_subclass

from . import aliases as codecs
from .base import (
    ArrayToArrayCodec,
    ArrayToBytesCodec,
    BytesToBytesCodec,
    Codec,
    CodecConfigImpl,
    CompressorCodec,
)


@autofrozen
class BloscConfig(CodecConfigImpl):
    """Blosc codec parameters: compressor, level, shuffle, block size."""

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
    """The Blosc meta-compressor: a byte/bit shuffle plus an inner compressor.

    Shuffles same-typed bytes together before handing them to ``cname``,
    then compresses in blocks so it can use multiple threads.
    """

    name: tx.Literal["blosc"]
    configuration: BloscConfig

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        if version == 3:
            return self
        return self.configuration.to_version(version)


@autofrozen
class BytesConfig(CodecConfigImpl):
    """Bytes codec parameters: the byte order to serialize with."""

    endian: tx.Optional[tx.Literal["big", "little"]]


@register_subclass(name="bytes")
@autofrozen
class BytesCodec(ArrayToBytesCodec):
    """Serializes an array to its raw bytes, in ``endian`` byte order.

    The plain array-to-bytes codec: no compression, no transform, just
    the array's values laid out contiguously.
    """

    name: tx.Literal["bytes"]
    configuration: BytesConfig


@autofrozen
class CRC32CConfig(CodecConfigImpl):
    """The CRC-32C codec's configuration: empty, as it takes no parameters."""


@register_subclass(name="crc32c")
@autofrozen
class CRC32CCodec(BytesToBytesCodec):
    """Appends a CRC-32C checksum of the preceding bytes, for integrity.

    Verified and stripped on decode; catches corrupted or truncated
    chunk data, but neither compresses nor transforms it.
    """

    name: tx.Literal["crc32c"]
    configuration: CRC32CConfig


@autofrozen
class GzipConfig(CodecConfigImpl):
    """Gzip codec parameters: the compression level."""

    level: codecs.GzipCompressionLevel = 5


@register_subclass(name="gzip")
@autofrozen
class GzipCodec(CompressorCodec):
    """Gzip compression: DEFLATE, at a configurable compression level."""

    name: tx.Literal["gzip"]
    configuration: GzipConfig


@autofrozen
class ShardingConfig(CodecConfigImpl):
    """Sharding codec parameters: the inner chunking and its two codec chains.

    ``chunk_shape`` is the shape of the sub-chunks packed into each
    shard; ``codecs`` encodes each sub-chunk's data, and
    ``index_codecs`` encodes the index that locates them within the
    shard, stored at its start or end per ``index_location``.
    """

    chunk_shape: tz.Shape
    codecs: tx.Tuple[Codec, ...]
    index_codecs: tx.Tuple[Codec, ...]
    index_location: tx.Literal["start", "end"] = "end"


@register_subclass(name="sharding_indexed")
@autofrozen
class ShardingCodec(ArrayToArrayCodec):
    """Packs many sub-chunks into one storage object, indexed for lookup.

    Splits a chunk into smaller sub-chunks (``chunk_shape``), encodes
    each with its own codec chain, and stores them together in a single
    shard alongside an index that maps each sub-chunk to its offset and
    length -- reducing the number of files or objects a store holds for
    arrays with many small chunks.
    """

    name: tx.Literal["sharding_indexed"]
    configuration: ShardingConfig


@autofrozen
class TransposeConfig(CodecConfigImpl):
    """Transpose codec parameters: the permutation of axes to apply."""

    order: tx.Tuple[int, ...]


@register_subclass(name="transpose")
@autofrozen
class TransposeCodec(ArrayToArrayCodec):
    """Permutes an array's axes into ``order`` before the rest of the pipeline.

    Reversed on decode, so the array's logical shape is unchanged; used
    to control the memory layout the later codecs see, e.g. to make an
    axis contiguous for a compressor.
    """

    name: tx.Literal["transpose"]
    configuration: TransposeConfig
