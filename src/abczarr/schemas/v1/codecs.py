"""
Zarr v1 compression codecs follow the numcodecs API exactly.
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
    "BloscOptions",
    "Bz2Options",
    "GzipOptions",
    "LZMAOptions",
    "LZ4Options",
    "PCodecOptions",
    "ZFPYOptions",
    "ZlibOptions",
    "ZstdOptions",
    "ValidCodecOptions",
    "ValidCodecName",
    "VALID_CODECS",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# typing
_7Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7]
_9Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_12Levels = tx.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


class CodecOptions(tx.TypedDict):
    ...


BloscCodecName = tx.Literal["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
BloscCompressionLevel = _9Levels
BloscShuffle = tx.Literal[
    0,      # "noshuffle"
    1,      # "shuffle"
    2,      # "bitshuffle"
    -1,     # "autoshuffle"
]


class BloscOptions(CodecOptions):
    cname: tx.NotRequired[BloscCodecName]
    clevel: tx.NotRequired[BloscCompressionLevel]
    shuffle: tx.NotRequired[BloscShuffle]
    blocksize: tx.NotRequired[int]
    typesize: tx.NotRequired[int]


BzCompressionLevel = _9Levels


class Bz2Options(CodecOptions):
    level: tx.NotRequired[BzCompressionLevel]


GzipCompressionLevel = _9Levels


class GzipOptions(CodecOptions):
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
LZMAPreset = _9Levels


class LZMAOptions(CodecOptions):
    format: tx.NotRequired[LZMAFormat]
    check: tx.NotRequired[LZMACheck]
    preset: tx.NotRequired[LZMAPreset]
    filters: tx.NotRequired[tz.BuiltinSequence[tz.JSONDict]]


class LZ4Options(CodecOptions):
    acceleration: tx.NotRequired[int]


PCodecMode = tx.Literal["auto", "classic"]
PCodecDelta = tx.Literal["auto", "none", "try_consecutive", "try_lookback"]
PCodecPaging = tx.Literal["equal_pages_up_to"]
PCodecDeltaOrder = _7Levels
PCodecLevel = _12Levels


class PCodecOptions(CodecOptions):
    level: tx.NotRequired[PCodecLevel]
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

class ZFPYOptions(CodecOptions):
    mode: tx.NotRequired[ZFPYMode]
    tolerance: tx.NotRequired[float]
    rate: tx.NotRequired[int]
    precision: tx.NotRequired[int]
    compression_kwargs: tx.NotRequired[tz.JSONDict]


ZlibCompressionLevel = _9Levels


class ZlibOptions(CodecOptions):
    level: tx.NotRequired[ZlibCompressionLevel]


class ZstdOptions(CodecOptions):
    level: tx.NotRequired[int]


Blosc = tx.Union[BloscCodecName, BloscOptions]
Bz2 = tx.Union[BzCompressionLevel, Bz2Options]
Gzip = tx.Union[GzipCompressionLevel, GzipOptions]
LZMA = tx.Union[LZMAFormat, LZMAOptions]
LZ4 = tx.Union[int, LZ4Options]
PCodec = tx.Union[PCodecLevel, PCodecOptions]
ZFPY = tx.Union[ZFPYMode, ZFPYOptions]
Zlib = tx.Union[ZlibCompressionLevel, ZlibOptions]
Zstd = tx.Union[int, ZstdOptions]

ValidCodecOptions = tx.Union[
    Blosc,
    Bz2,
    Gzip,
    LZMA,
    LZ4,
    PCodec,
    ZFPY,
    Zlib,
    Zstd
]

Codec = ValidCodecName = tx.Literal[
    "blosc",
    "bz2",
    "gzip",
    "lz4",
    "lzma",
    "pcodec",
    "zfpy",
    "zlib",
    "zstd",
]

VALID_CODECS = Codec.__args__
