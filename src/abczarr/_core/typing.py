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
    "OMEVersion",
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
    "AnyOMEVersion",
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
    ToAnnotated,
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

# Json
_JsonNumber = tx.Union[int, float]
_JsonNumberLike = tx.Union[int, float, bool]
_JsonScalar = tx.Union[int, float, bool, str, None]
_Json = tx.Union[_JsonScalar, tx.Mapping[str, "Json"], BuiltinSequence["Json"]]
JsonNumber = _JsonNumber
JsonNumberLike = _JsonNumberLike
JsonScalar = _JsonScalar
Json = _Json
JsonDict = tx.Mapping[str, Json]

class _Freeze:
    """Marker: deep-freeze this Json value into `FrozenDict`/`tuple`.

    Carried as `Annotated` metadata on `FrozenJson` so the freezing is tied
    to *that* type -- `ToFrozenJson` is registered for this marker -- rather
    than to `FrozenDict`, which is more general than a frozen-Json value.
    """


# The frozen Json model. Its mapping and sequence are the *immutable*
# `FrozenDict` and `tuple`, matching the immutable nature of the frozen
# attrs classes that hold it -- an extra item, or an `attributes` value, is
# deep-frozen so the whole object stays genuinely immutable (and, as a
# bonus, hashable). The `_Freeze` marker makes `ToFrozenJson` (below) rebuild
# a plain `dict`/`list` into a `FrozenDict`/`tuple` recursively.
_FrozenJson = tx.Union[
    _JsonScalar, FrozenDict[str, "FrozenJson"], tx.Tuple["FrozenJson", ...]
]
FrozenJson = tx.Annotated[_FrozenJson, _Freeze()]
FrozenJsonDict = FrozenDict[str, FrozenJson]

_MutableJson = tx.Union[
    _JsonScalar, tx.MutableMapping[str, "Json"], tx.List["Json"]
]
MutableJson = _MutableJson
MutableJsonDict = tx.MutableMapping[str, MutableJson]

# Shapes
Shape = tx.Tuple[BuiltinNonNegativeIntegral, ...]
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
OMEVersion = tx.Literal["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
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
AnyOMEVersion = tx.Union[OMEVersion, str]
AnyCompressorType = tx.Union[CompressorType, str]
CompressorOptions = tx.Mapping[str, tx.Union[float, str]]
PyramidFunction = tx.Callable[[npt.ArrayLike], npt.ArrayLike]
PyramidMode = tx.Union[KnownPyramidMode, PyramidFunction]


@register_converter(Json)
class ToJson(Converter[Json, Json]):
    """
    A converter for Json-compatible types.
    """

    def __call__(self, value: Json) -> Json:
        return json.loads(json.dumps(value))


def _freeze_json(value: tx.Any) -> tx.Any:
    """Recursively freeze a Json value into its immutable form.

    A mapping becomes a `FrozenDict` and a list or tuple a `tuple`, each of
    frozen values; a scalar is returned unchanged. The structure is rebuilt
    directly -- nothing is serialized -- so a frozen-Json field holds a
    deeply immutable value, mirroring the immutable frozen class that carries
    it (and hashable as a consequence).
    """
    if isinstance(value, collections.abc.Mapping):
        return FrozenDict(
            (str(key), _freeze_json(item)) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


@ToAnnotated.register_metadata(_Freeze)
class ToFrozenJson(Converter[_FrozenJson, _FrozenJson]):
    """Deep-freeze a Json value into `FrozenDict`/`tuple` (see `_freeze_json`).

    Registered for the `_Freeze` marker carried by `FrozenJson`, so it applies
    to a frozen-Json value specifically. It converts the whole value in one
    recursive pass rather than relying on the frozen-Json union's branches,
    which cannot tell a `dict` from a `list` without corrupting one of them.
    """

    def __call__(self, value: _FrozenJson) -> _FrozenJson:
        return _freeze_json(value)

