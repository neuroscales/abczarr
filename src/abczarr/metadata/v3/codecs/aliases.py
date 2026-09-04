"""Type aliases for Zarr v3 built-in codec configurations.

These name the enumerated fields of the v3 codec configs (blosc, gzip).
Previously imported from the removed ``abczarr.schemas.v3`` TypedDicts.
"""

__all__ = [
    "BloscCodecName",
    "BloscShuffle",
    "BloscCompressionLevel",
    "GzipCompressionLevel",
]

import typing_extensions as tx

CompressionLevel = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

BloscCodecName = tx.Literal[
    "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"
]
BloscShuffle = tx.Literal["noshuffle", "shuffle", "bitshuffle"]
BloscCompressionLevel = CompressionLevel
GzipCompressionLevel = CompressionLevel
