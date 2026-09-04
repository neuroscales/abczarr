"""Type aliases for Zarr v1 codec configurations (numcodecs API).

Names the enumerated fields of the v1 built-in codec options, plus the
v1 codec-name literal. Previously imported from the removed
``abczarr.schemas.v1`` TypedDicts.
"""

__all__ = [
    "BloscCodecName",
    "BloscShuffle",
    "BloscCompressionLevel",
    "GzipCompressionLevel",
    "Codec",
]

import typing_extensions as tx

_9Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

BloscCodecName = tx.Literal[
    "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"
]
BloscShuffle = tx.Literal[0, 1, 2, -1]
BloscCompressionLevel = _9Levels
GzipCompressionLevel = _9Levels

# The v1 array's ``compression`` is a codec name.
Codec = tx.Literal[
    "blosc", "bz2", "gzip", "lz4", "lzma", "pcodec", "zfpy", "zlib", "zstd"
]
