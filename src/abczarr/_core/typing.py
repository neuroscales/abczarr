"""Shared type aliases used throughout abczarr: numeric and JSON value
types, shapes and chunk specifications, and the enumerated literals for
Zarr's own vocabulary (access modes, compressor names, node types, and
so on).
"""

__all__ = [
    "T",
    "OneOrIter",
    "OneOrSeq",
    "BuiltinSequence",
    "BuiltinNumber",
    "BuiltinReal",
    "BuiltinIntegral",
    "BuiltinScalar",
    "BuiltinPositiveNumber",
    "BuiltinNegativeNumber",
    "BuiltinNonPositiveNumber",
    "BuiltinNonNegativeNumber",
    "BuiltinPositiveIntegral",
    "BuiltinNegativeIntegral",
    "BuiltinNonPositiveIntegral",
    "BuiltinNonNegativeIntegral",
    "BytesLike",
    "StringLike",
    "PathLike",
    "Number",
    "Integral",
    "Real",
    "PositiveNumber",
    "NegativeNumber",
    "NonPositiveNumber",
    "NonNegativeNumber",
    "PositiveIntegral",
    "NegativeIntegral",
    "NonPositiveIntegral",
    "NonNegativeIntegral",
    "JsonNumber",
    "JsonNumberLike",
    "JsonScalar",
    "Json",
    "JsonDict",
    "FrozenJson",
    "FrozenJsonDict",
    "MutableJson",
    "MutableJsonDict",
    "Shape",
    "ShapeIsh",
    "ShapeLike",
    "ChunksLike",
    "ChunksIsh",
    "Chunks",
    "ChunkCoords",
    "LogLevel",
    "AccessMode",
    "KnownDriver",
    "ZarrVersion",
    "CompressorTypeV1",
    "CompressorTypeV2",
    "CompressorTypeV3",
    "CompressorType",
    "NodeType",
    "MemoryOrder",
    "DimensionSeparator",
    "KnownPyramidMode",
    "SpatialAxisName",
    "TimeAxisName",
    "ChannelAxisName",
    "OMEAxisName",
    "AnyAxisName",
    "AnyAxisNames",
    "Attributes",
    "FrozenAttributes",
    "AnyDriver",
    "AnyZarrVersion",
    "AnyCompressorType",
    "CompressorOptions",
    "PyramidFunction",
    "PyramidMode",
    "DataTypeV2",
    "DataTypeV3",
]

# stdlib
import collections.abc
import json
import numbers
import os

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# internals
from .attrs import (
    Converter,
    ToNegative,
    ToNonNegative,
    ToNonPositive,
    ToPositive,
    register_converter,
)
from .dtypes import DataTypeV2, DataTypeV3  # noqa: F401
from .frozendict import FrozenDict
from .typevars.inv import T  # noqa: F401  (re-exported for back-compat)

# General types
OneOrIter = tx.Union[T, tx.Iterable[T]]
OneOrSeq = tx.Union[T, tx.Sequence[T]]
BuiltinSequence = tx.Union[tx.Tuple[T, ...], tx.List[T]]
"""A built-in sequence of one element type: a tuple or a list."""

# Values
_BuiltinIntegralNumber = int
_BuiltinRealNumber = tx.Union[int, float]
_BuiltinNumber = tx.Union[_BuiltinRealNumber, complex]
_BuiltinScalar = tx.Union[_BuiltinNumber, str]
BuiltinNumber = _BuiltinNumber
BuiltinReal = _BuiltinRealNumber
BuiltinIntegral = int
BuiltinScalar = _BuiltinScalar

BuiltinPositiveNumber = tx.Annotated[
    BuiltinReal, ToPositive(compose=True)]
BuiltinNegativeNumber = tx.Annotated[
    BuiltinReal, ToNegative(compose=True)]
BuiltinNonPositiveNumber = tx.Annotated[
    BuiltinReal, ToNonPositive(compose=True)]
BuiltinNonNegativeNumber = tx.Annotated[
    BuiltinReal, ToNonNegative(compose=True)]

BuiltinPositiveIntegral = tx.Annotated[
    BuiltinIntegral, ToPositive(compose=True)]
BuiltinNegativeIntegral = tx.Annotated[
    BuiltinIntegral, ToNegative(compose=True)]
BuiltinNonPositiveIntegral = tx.Annotated[
    BuiltinIntegral, ToNonPositive(compose=True)]
BuiltinNonNegativeIntegral = tx.Annotated[
    BuiltinIntegral, ToNonNegative(compose=True)]
"""A built-in integer constrained to be zero or greater."""

_BytesLike = tx.Union[bytes, bytearray, memoryview]
_StringLike = tx.Union[str, _BytesLike]
_PathLike = tx.Union[str, os.PathLike]
BytesLike = _BytesLike
StringLike = _StringLike
PathLike = _PathLike

_Integral = tx.Union[numbers.Integral, np.integer, np.bool_]
_Real = tx.Union[numbers.Real, np.floating, np.integer, np.bool_]
_Number = tx.Union[numbers.Number, np.number, np.bool_]
Number = _Number
Integral = _Integral
Real = _Real

PositiveNumber = tx.Annotated[Real, ToPositive(compose=True)]
NegativeNumber = tx.Annotated[Real, ToNegative(compose=True)]
NonPositiveNumber = tx.Annotated[Real, ToNonPositive(compose=True)]
NonNegativeNumber = tx.Annotated[Real, ToNonNegative(compose=True)]

PositiveIntegral = tx.Annotated[
    Integral, ToPositive(compose=True)]
NegativeIntegral = tx.Annotated[
    Integral, ToNegative(compose=True)]
NonPositiveIntegral = tx.Annotated[
    Integral, ToNonPositive(compose=True)]
NonNegativeIntegral = tx.Annotated[
    Integral, ToNonNegative(compose=True)]

# JSON
# `JsonNumber`, `JsonNumberLike` and `JsonScalar` each carry a converter
# in their `Annotated` metadata. That converter replaces the one that
# would otherwise be built from the `Union`. A union converter tries its
# members in a fixed order. With `int` first, a numpy float is truncated
# and a numpy bool becomes an `int`. With `float` first, a numpy integer
# becomes a float. `ToJsonNumber` chooses `int` or `float` from the value
# itself, so an integral value becomes an `int` and any other real value
# a `float`. A builtin bool passes through unchanged, as it did with the
# plain union. `ToJsonScalar` keeps strings and booleans as they are and
# hands every other number to `ToJsonNumber`. `ToFrozenJson` applies
# `ToJsonScalar` to each leaf of a frozen JSON value.


class ToJsonNumber(Converter[tx.Union[float, int], tx.Any]):
    """Converts a number to the builtin `int` or `float` that matches it.

    A builtin `int`, `float` or `bool` is returned unchanged. A numpy
    bool becomes a `bool`. Any other integral number, such as
    `numpy.int64`, becomes an `int`. Any other real number, such as
    `numpy.float32`, becomes a `float`. A string is parsed as an `int`
    when it spells an integer, and as a `float` otherwise. A value that
    cannot be read as a real number raises a conversion error.
    """

    def __call__(self, value: tx.Any) -> tx.Union[float, int]:
        if type(value) in (int, float, bool):
            return value
        if isinstance(value, np.bool_):
            return bool(value)
        try:
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return float(value)
            if isinstance(value, numbers.Integral):
                return int(value)
            if isinstance(value, numbers.Real):
                return float(value)
            if isinstance(value, (bytes, bytearray, complex)):
                raise TypeError(type(value).__name__)
            return float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise self.value_error(
                value, f"Cannot convert {value!r} to a JSON number."
            ) from exc


JsonNumber = tx.Annotated[tx.Union[float, int], ToJsonNumber()]
"""A JSON number, stored as a builtin `int` or `float`."""
JsonNumberLike = tx.Annotated[tx.Union[float, int, bool], ToJsonNumber()]
"""A JSON number or boolean, stored as a builtin `int`, `float` or
`bool`."""


class ToJsonScalar(Converter[tx.Union[int, float, bool, str, None], tx.Any]):
    """Converts a scalar to the builtin type that JSON uses for it.

    `None` is returned unchanged. A string is returned as a builtin
    `str`, so a `numpy.str_` becomes a `str`. A string is never parsed
    as a number. A builtin bool or a numpy bool becomes a `bool`. Any
    other number is converted by `ToJsonNumber`. An integral number
    becomes an `int`, and any other real number becomes a `float`. A
    value of any other type raises a conversion error.
    """

    def __call__(self, value: tx.Any) -> tx.Union[int, float, bool, str, None]:
        if value is None:
            return None
        if isinstance(value, str):
            return str(value)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (numbers.Number, np.number)):
            return ToJsonNumber()(value)
        raise self.value_error(
            value, f"Cannot convert {value!r} to a JSON scalar."
        )


JsonScalar = tx.Annotated[
    tx.Union[int, float, bool, str, None], ToJsonScalar()
]
"""A JSON scalar: a number, a boolean, a string, or null."""
Json = tx.Union[JsonScalar, tx.Mapping[str, "Json"], BuiltinSequence["Json"]]
"""Any JSON value. A scalar, a mapping of string keys to JSON values, or
a sequence of JSON values."""
JsonDict = tx.Mapping[str, Json]
"""A JSON object: a mapping of string keys to JSON values."""

# The frozen JSON model. Its mapping and sequence are the `immutable`
# `FrozenDict` and `tuple`, matching the immutable nature of the frozen
# attrs classes that hold it -- an extra item, or an `attributes` value, is
# deep-frozen so the whole object stays genuinely immutable (and, as a
# bonus, hashable). `ToFrozenJson`, carried in `FrozenJson`'s `Annotated`
# metadata, rebuilds a plain `dict`/`list` into a `FrozenDict`/`tuple`
# recursively.
_FrozenJson = tx.Union[
    JsonScalar, FrozenDict[str, "FrozenJson"], tx.Tuple["FrozenJson", ...]
]


class ToFrozenJson(Converter[_FrozenJson, _FrozenJson]):
    """Rebuilds a JSON-like value into its deeply frozen form.

    A field typed as `FrozenJson` is converted by this class. A mapping
    becomes a `FrozenDict` with string keys, and a list or tuple becomes
    a `tuple`. Every nested value is converted in the same way. A scalar
    is converted by `ToJsonScalar`, so a numpy scalar becomes the
    builtin `int`, `float`, `bool` or `str` that matches it. A value
    that is not JSON raises a conversion error. The result is immutable
    and hashable at every level.
    """

    def __call__(self, value: _FrozenJson) -> _FrozenJson:
        if isinstance(value, collections.abc.Mapping):
            return FrozenDict(
                (str(key), self(item)) for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return tuple(self(item) for item in value)
        return _TO_JSON_SCALAR(value)


_TO_JSON_SCALAR = ToJsonScalar()

FrozenJson = tx.Annotated[_FrozenJson, ToFrozenJson()]
FrozenJsonDict = FrozenDict[str, FrozenJson]

MutableJson = tx.Union[
    JsonScalar, tx.MutableMapping[str, "Json"], tx.List["Json"]
]
MutableJsonDict = tx.MutableMapping[str, MutableJson]

# Shapes
Shape = tx.Tuple[BuiltinNonNegativeIntegral, ...]
"""An array or chunk shape: a tuple of non-negative integers, one per
dimension."""
ShapeIsh = tx.Sequence[BuiltinNonNegativeIntegral]
ShapeLike = tx.Iterable[NonNegativeIntegral]
ChunksLike = tx.Union[ShapeLike, tx.Iterable[tx.Iterable[NonNegativeIntegral]]]
ChunksIsh = tx.Union[ShapeIsh, tx.Sequence[tx.Sequence[NonNegativeIntegral]]]
Chunks = tx.Union[Shape, tx.Tuple[Shape, ...]]
ChunkCoords = Shape

# Enums
LogLevel = tx.Literal["debug", "info", "warning", "error", "critical"]
AccessMode = tx.Literal["r", "r+", "a", "w", "w-"]
KnownDriver = tx.Literal["zarr-python", "tensorstore"]
ZarrVersion = tx.Literal[1, 2, 3]
CompressorTypeV1 = tx.Literal[
    "blosc", "gzip", "bz2", "lzma", "lz4", "pcodec", "zfpy", "zlib", "zstd",
    "none"
]
CompressorTypeV2 = CompressorTypeV1
CompressorTypeV3 = tx.Literal["blosc", "gzip", "zstd", "none"]
CompressorType = tx.Union[CompressorTypeV1, CompressorTypeV2, CompressorTypeV3]
NodeType = tx.Literal["group", "array"]
MemoryOrder = tx.Literal["C", "F"]
DimensionSeparator = tx.Literal[".", "/"]
KnownPyramidMode = tx.Literal["mean", "median"]

SpatialAxisName = tx.Literal["x", "y", "z"]
TimeAxisName = tx.Literal["t"]
ChannelAxisName = tx.Literal["c"]
OMEAxisName = tx.Union[SpatialAxisName, TimeAxisName, ChannelAxisName]
AnyAxisName = tx.Union[str, None]
AnyAxisNames = tx.Optional[tx.Sequence[AnyAxisName]]

# Internal types
Attributes = tx.MutableMapping[str, Json]
FrozenAttributes = tx.Mapping[str, Json]
AnyDriver = tx.Union[KnownDriver, str]
AnyZarrVersion = tx.Union[ZarrVersion, int]
AnyCompressorType = tx.Union[CompressorType, str]
CompressorOptions = tx.Mapping[str, tx.Union[float, str]]
PyramidFunction = tx.Callable[[npt.ArrayLike], npt.ArrayLike]
PyramidMode = tx.Union[KnownPyramidMode, PyramidFunction]


@register_converter(Json)
class ToJson(Converter[Json, Json]):
    """Normalizes a value to plain, JSON-compatible built-in types.

    Round-trips the value through `json.dumps` and `json.loads`, so a
    value that is already JSON-compatible, or close to it (a tuple in
    place of a list, for example), comes back as the plain `dict`,
    `list`, `str`, `int`, `float`, `bool`, or `None` JSON itself uses.
    A value that `json.dumps` cannot serialize raises `TypeError`.
    """

    def __call__(self, value: Json) -> Json:
        return json.loads(json.dumps(value))

