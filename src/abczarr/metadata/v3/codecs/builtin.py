"""The built-in Zarr v3 codecs every implementation should support.

Corresponds to the codecs the Zarr v3 specification names as
required: blosc, bytes, crc32c, gzip, sharding and transpose.
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
    """Holds the Blosc codec's parameters: compressor, level, shuffle,
    and block size.

    Attributes
    ----------
    cname : str
        The inner compressor Blosc applies.
    clevel : int
        The compression level, from 0 to 9.
    shuffle : str
        The byte-shuffle filter applied before compression:
        ``"noshuffle"``, ``"shuffle"`` or ``"bitshuffle"``.
    blocksize : int
        The block size Blosc compresses in, in bytes. ``0`` lets
        Blosc choose automatically.
    typesize : int or None
        The size, in bytes, of the array's element type. Blosc uses
        this to group same-position bytes together when shuffling.
    """

    cname: codecs.BloscCodecName = "lz4"
    clevel: codecs.BloscCompressionLevel = 5
    shuffle: codecs.BloscShuffle = "shuffle"
    blocksize: int = 0
    typesize: tx.Optional[int] = None

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        """Convert this configuration to another Zarr version's Blosc
        codec.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        Metadata
            The equivalent Blosc codec for `version`.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3.
        """
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
    """Compresses data with Blosc, a meta-compressor that shuffles bytes
    and then applies an inner compressor.

    Blosc groups same-typed bytes together before handing them to
    `cname`, then compresses the result in blocks so multiple threads
    can be used.

    Attributes
    ----------
    name : str
        Always ``"blosc"``.
    configuration : BloscConfig
        The codec's parameters.
    """

    name: tx.Literal["blosc"]
    configuration: BloscConfig

    def to_version(self, version: tz.ZarrVersion) -> Metadata:
        """Convert this codec to another Zarr version.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        Metadata
            The equivalent Blosc codec for `version`.
        """
        if version == 3:
            return self
        return self.configuration.to_version(version)


@autofrozen
class BytesConfig(CodecConfigImpl):
    """Holds the bytes codec's parameters: the byte order to serialize
    with.

    Attributes
    ----------
    endian : str or None
        The byte order to serialize with, ``"big"`` or ``"little"``,
        or `None` for a dtype whose byte order does not matter, such
        as a single-byte type.
    """

    endian: tx.Optional[tx.Literal["big", "little"]]


@register_subclass(name="bytes")
@autofrozen
class BytesCodec(ArrayToBytesCodec):
    """Serializes an array to its raw bytes, in `configuration.endian`
    byte order.

    This is the plain array-to-bytes codec. It applies no compression
    and no transform, laying the array's values out contiguously.

    Attributes
    ----------
    name : str
        Always ``"bytes"``.
    configuration : BytesConfig
        The codec's parameters.
    """

    name: tx.Literal["bytes"]
    configuration: BytesConfig


@autofrozen
class CRC32CConfig(CodecConfigImpl):
    """This configuration is empty, since the CRC-32C codec takes no
    parameters."""


@register_subclass(name="crc32c")
@autofrozen
class CRC32CCodec(BytesToBytesCodec):
    """Appends a CRC-32C checksum of the preceding bytes, for integrity.

    The checksum is verified and stripped on decode, which catches
    corrupted or truncated chunk data. This codec neither compresses
    nor transforms the chunk data itself.

    Attributes
    ----------
    name : str
        Always ``"crc32c"``.
    configuration : CRC32CConfig
        The codec's parameters. Always empty.
    """

    name: tx.Literal["crc32c"]
    configuration: CRC32CConfig


@autofrozen
class GzipConfig(CodecConfigImpl):
    """Holds the gzip codec's parameters: the compression level.

    Attributes
    ----------
    level : int
        The compression level, from 0 to 9.
    """

    level: codecs.GzipCompressionLevel = 5


@register_subclass(name="gzip")
@autofrozen
class GzipCodec(CompressorCodec):
    """Applies DEFLATE compression (gzip) at a configurable compression
    level.

    Attributes
    ----------
    name : str
        Always ``"gzip"``.
    configuration : GzipConfig
        The codec's parameters.
    """

    name: tx.Literal["gzip"]
    configuration: GzipConfig


@autofrozen
class ShardingConfig(CodecConfigImpl):
    """Holds the sharding codec's parameters: the inner chunking and its
    two codec chains.

    Attributes
    ----------
    chunk_shape : tuple of int
        The shape of the sub-chunks packed into each shard.
    codecs : tuple of Codec
        The codec pipeline that encodes each sub-chunk's data.
    index_codecs : tuple of Codec
        The codec pipeline that encodes the index locating each
        sub-chunk within the shard.
    index_location : str
        Where the index is stored within the shard: ``"start"`` or
        ``"end"``.
    """

    chunk_shape: tz.Shape
    codecs: tx.Tuple[Codec, ...]
    index_codecs: tx.Tuple[Codec, ...]
    index_location: tx.Literal["start", "end"] = "end"


@register_subclass(name="sharding_indexed")
@autofrozen
class ShardingCodec(ArrayToArrayCodec):
    """Packs many sub-chunks into one storage object, indexed for lookup.

    Splits a chunk into smaller sub-chunks, encodes each with its own
    codec chain, and stores the sub-chunks together in a single
    shard alongside an index that maps each sub-chunk to its offset
    and length. Sharding this way reduces the number of files or
    objects a store holds for arrays with many small chunks.

    Attributes
    ----------
    name : str
        Always ``"sharding_indexed"``.
    configuration : ShardingConfig
        The codec's parameters.
    """

    name: tx.Literal["sharding_indexed"]
    configuration: ShardingConfig


@autofrozen
class TransposeConfig(CodecConfigImpl):
    """Holds the transpose codec's parameters: the permutation of axes
    to apply.

    Attributes
    ----------
    order : tuple of int
        The permutation applied to the array's axes.
    """

    order: tx.Tuple[int, ...]


@register_subclass(name="transpose")
@autofrozen
class TransposeCodec(ArrayToArrayCodec):
    """Permutes an array's axes into `configuration.order` before the
    rest of the pipeline.

    The permutation is reversed on decode, so the array's logical shape
    is unchanged. It controls the memory layout the later codecs see,
    for example to make an axis contiguous for a compressor.

    Attributes
    ----------
    name : str
        Always ``"transpose"``.
    configuration : TransposeConfig
        The codec's parameters.
    """

    name: tx.Literal["transpose"]
    configuration: TransposeConfig
