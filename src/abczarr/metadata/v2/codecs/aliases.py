"""Type aliases for Zarr v2 codec configurations (numcodecs API).

Names the enumerated fields of the v2 built-in and extension codec
configs. Previously imported from the removed ``abczarr.schemas.v2``
TypedDicts; also reused by the v1 extension codecs.
"""

__all__ = [
    "BloscCodecName",
    "BloscShuffle",
    "BloscCompressionLevel",
    "Bz2CompressionLevel",
    "GzipCompressionLevel",
    "LZMAFormat",
    "LZMACheck",
    "LZMACompressionLevel",
    "PCodecMode",
    "PCodecDelta",
    "PCodecPaging",
    "PCodecDeltaOrder",
    "PCodecCompressionLevel",
    "ZFPYMode",
    "ZlibCompressionLevel",
]

import typing_extensions as tx

Compression9Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Compression12Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

BloscCodecName = tx.Literal[
    "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"
]
BloscShuffle = tx.Literal[0, 1, 2, -1]
BloscCompressionLevel = Compression9Levels

Bz2CompressionLevel = Compression9Levels
GzipCompressionLevel = Compression9Levels

LZMAFormat = tx.Literal[0, 1, 2, 3]
LZMACheck = tx.Literal[0, 1, 4, 10, 15, 16]
LZMACompressionLevel = Compression9Levels

PCodecMode = tx.Literal["auto", "classic"]
PCodecDelta = tx.Literal["auto", "none", "try_consecutive", "try_lookback"]
PCodecPaging = tx.Literal["equal_pages_up_to"]
PCodecDeltaOrder = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7]
PCodecCompressionLevel = Compression12Levels

ZFPYMode = tx.Literal[0, 1, 2, 3, 4]

ZlibCompressionLevel = Compression9Levels
