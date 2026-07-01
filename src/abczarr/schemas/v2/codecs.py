"""
Zarr v2 compression codecs follow the numcodecs API exactly.
"""
__all__ = [
    "Codec",
    "Blosc",
    "Bz2",
    "Gzip",
    "LZMA",
    "LZ4",
    "PCodec",
    "ZFPY",
    "Zlib",
    "Zstd",
    "ValidCodec",
    "ValidCodecName",
    "VALID_CODEC_NAMES",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# typing
Compression9Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Compression12Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


class Codec(tx.TypedDict, total=False):
    id: str


BloscCodecName = tx.Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
BloscShuffle = tx.Literal[
    0,      # "noshuffle"
    1,      # "shuffle"
    2,      # "bitshuffle"
    -1,     # "autoshuffle"
]
BloscCompressionLevel = Compression9Levels


class Blosc(Codec, total=True):
    id: tx.Literal["blosc"]
    cname: tx.NotRequired[BloscCodecName]
    clevel: tx.NotRequired[BloscCompressionLevel]
    shuffle: tx.NotRequired[BloscShuffle]
    blocksize: tx.NotRequired[int]
    typesize: tx.NotRequired[int]


Bz2CompressionLevel = Compression9Levels


class Bz2(Codec, total=True):
    id: tx.Literal["bz2"]
    level: tx.NotRequired[Bz2CompressionLevel]


GzipCompressionLevel = Compression9Levels


class Gzip(Codec, total=True):
    id: tx.Literal["gzip"]
    level: tx.NotRequired[GzipCompressionLevel]


LZMAFormat = tx.Literal[
    0,      # auto
    1,      # xz
    2,      # alone
    3,      # raw
]
LZMACheck = tx.Literal[
    0,      # none
    1,      # crc32
    4,      # crc64
    10,     # sha256
    15,     # id_max
    16,     # unknown
]
LZMACompressionLevel = Compression9Levels


class LZMA(Codec, total=True):
    id: tx.Literal["lzma"]
    format: tx.NotRequired[LZMAFormat]
    check: tx.NotRequired[LZMACheck]
    preset: tx.NotRequired[LZMACompressionLevel]
    filters: tx.NotRequired[tz.BuiltinSequence[tz.JSONDict]]


class LZ4(Codec, total=True):
    id: tx.Literal["lz4"]
    acceleration: tx.NotRequired[int]


PCodecMode = tx.Literal["auto", "classic"]
PCodecDelta = tx.Literal["auto", "none", "try_consecutive", "try_lookback"]
PCodecPaging = tx.Literal["equal_pages_up_to"]
PCodecDeltaOrder = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7]
PCodecCompressionLevel = Compression12Levels


class PCodec(Codec, total=True):
    id: tx.Literal["pcodec"]
    level: tx.NotRequired[PCodecCompressionLevel]
    mode_spec: tx.NotRequired[PCodecMode]
    delta_spec: tx.NotRequired[PCodecDelta]
    paging_spec: tx.NotRequired[PCodecPaging]
    delta_encoding_order: tx.NotRequired[tx.Optional[PCodecDeltaOrder]]
    equal_pages_up_to: tx.NotRequired[int]


ZFPYMode = tx.Literal[
    0,  # null
    1,  # expert
    2,  # fixed_rate
    3,  # fixed_precision
    4,  # fixed_accuracy
]


class ZFPY(Codec, total=True):

    id: tx.Literal["zfpy"]
    mode: tx.NotRequired[ZFPYMode]
    tolerance: tx.NotRequired[float]
    rate: tx.NotRequired[int]
    precision: tx.NotRequired[int]
    compression_kwargs: tx.NotRequired[tz.JSONDict]


ZlibCompressionLevel = Compression9Levels


class Zlib(Codec, total=True):
    id: tx.Literal["zlib"]
    level: tx.NotRequired[ZlibCompressionLevel]


class Zstd(Codec, total=True):
    id: tx.Literal["zstd"]
    level: tx.NotRequired[int]


ValidCodec = tx.Union[
    Blosc,
    Bz2,
    Gzip,
    LZMA,
    LZ4,
    PCodec,
    ZFPY,
    Zlib,
    Zstd,
]
ValidCodecName = tx.Literal[
    "blosc",
    "bz2",
    "gzip",
    "lzma",
    "lz4",
    "pcodec",
    "zfpy",
    "zlib",
    "zstd",
]
VALID_CODEC_NAMES = ValidCodecName.__args__
