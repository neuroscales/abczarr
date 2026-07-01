"""
Zarr v2 filters follow the numcodecs API exactly.
"""
__all__ = [
    "Filter",
    "Delta",
    "FixedScaleOffset",
    "Quantize",
    "Bitround",
    "PackBits",
    "Categorize",
    "AsType",
    "Shuffle",
    "ValidFilter",
    "ValidFilterName",
    "VALID_FILTER_NAMES",]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz


class Filter(tx.TypedDict):
    id: str


class Delta(Filter):
    id: tx.Literal["delta"]
    dtype: tz.DataTypeV2
    astype: tx.NotRequired[tz.DataTypeV2]


class FixedScaleOffset(Filter):
    id: tx.Literal["fixedscaleoffset"]
    offset: float
    scale: float
    dtype: tz.DataTypeV2
    astype: tx.NotRequired[tz.DataTypeV2]


class Quantize(Filter):
    id: tx.Literal["quantize"]
    digits: int
    dtype: tz.DataTypeV2
    astype: tx.NotRequired[tz.DataTypeV2]


class Bitround(Filter):
    id: tx.Literal["bitround"]
    keepbits: int


class PackBits(Filter):
    id: tx.Literal["packbits"]


class Categorize(Filter):
    id: tx.Literal["categorize"]
    labels: tz.BuiltinSequence[str]
    dtype: tz.DataTypeV2
    astype: tx.NotRequired[tz.DataTypeV2]


class AsType(Filter):
    id: tx.Literal["astype"]
    encode_dtype: tz.DataTypeV2
    decode_dtype: tx.NotRequired[tz.DataTypeV2]


class Shuffle(Filter):
    id: tx.Literal["shuffle"]
    elementsize: tx.NotRequired[int]


ValidFilter = tx.Union[
    Delta,
    FixedScaleOffset,
    Quantize,
    Bitround,
    PackBits,
    Categorize,
    AsType,
    Shuffle,
]
ValidFilterName = tx.Literal[
    "delta",
    "fixedscaleoffset",
    "quantize",
    "bitround",
    "packbits",
    "categorize",
    "astype",
    "shuffle",
]
VALID_FILTER_NAMES = ValidFilterName.__args__
