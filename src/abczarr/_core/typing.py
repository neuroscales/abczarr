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
    "JSONNumber",
    "JSONNumberLike",
    "JSONScalar",
    "JSON",
    "JSONDict",
    "FrozenJSON",
    "FrozenJSONDict",
    "MutableJSON",
    "MutableJSONDict",
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

# General types
T = tx.TypeVar("T")
OneOrIter = tx.Union[T, tx.Iterable[T]]
OneOrSeq = tx.Union[T, tx.Sequence[T]]
BuiltinSequence = tx.Union[tx.Tuple[T, ...], tx.List[T]]

# Values
_BuiltinIntegralNumber = int
_BuiltinRealNumber = tx.Union[int, float]
_BuiltinNumber = tx.Union[_BuiltinRealNumber, complex]
_BuiltinScalar = tx.Union[_BuiltinNumber, str]
BuiltinNumber = tx.TypeVar(
    "BuiltinNumber", bound=_BuiltinNumber, default=_BuiltinNumber)
BuiltinReal = tx.TypeVar(
    "BuiltinReal", bound=_BuiltinRealNumber, default=_BuiltinRealNumber)
BuiltinIntegral = tx.TypeVar(
    "BuiltinIntegral", bound=int, default=int)
BuiltinScalar = tx.TypeVar(
    "BuiltinScalar", bound=_BuiltinScalar, default=_BuiltinScalar)

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
BytesLike = tx.TypeVar("BytesLike", bound=_BytesLike, default=_BytesLike)
StringLike = tx.TypeVar("StringLike", bound=_StringLike, default=_StringLike)
PathLike = tx.TypeVar("PathLike", bound=_PathLike, default=_PathLike)

_Integral = tx.Union[numbers.Integral, np.integer, np.bool_]
_Real = tx.Union[numbers.Real, np.floating, np.integer, np.bool_]
_Number = tx.Union[numbers.Number, np.number, np.bool_]
Number = tx.TypeVar("Number", bound=_Number, default=_Number)
Integral = tx.TypeVar("Integral", bound=_Integral, default=_Integral)
Real = tx.TypeVar("Real", bound=_Real, default=_Real)

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
_JSONNumber = tx.Union[int, float]
_JSONNumberLike = tx.Union[int, float, bool]
_JSONScalar = tx.Union[int, float, bool, str, None]
_JSON = tx.Union[_JSONScalar, tx.Mapping[str, "JSON"], BuiltinSequence["JSON"]]
JSONNumber = tx.TypeVar("JSONNumber", bound=_JSONNumber, default=_JSONNumber)
JSONNumberLike = tx.TypeVar(
    "JSONNumberLike", bound=_JSONNumberLike, default=_JSONNumberLike
)
JSONScalar = tx.TypeVar("JSONScalar", bound=_JSONScalar, default=_JSONScalar)
JSON = tx.TypeVar("JSON", bound=_JSON, default=_JSON)
JSONDict = tx.Mapping[str, JSON]

_FrozenJSON = tx.Union[
    _JSONScalar, FrozenDict[str, "JSON"], tx.Tuple["JSON", ...]
]
FrozenJSON = tx.TypeVar("FrozenJSON", bound=_FrozenJSON, default=_FrozenJSON)
FrozenJSONDict = FrozenDict[str, FrozenJSON]

_MutableJSON = tx.Union[
    _JSONScalar, tx.MutableMapping[str, "JSON"], tx.List["JSON"]
]
MutableJSON = tx.TypeVar(
    "MutableJSON", bound=_MutableJSON, default=_MutableJSON
)
MutableJSONDict = tx.MutableMapping[str, MutableJSON]

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
KnownDriver = tx.Literal["zarr-python", "tensorstore", "zarrita"]
ZarrVersion = tx.Literal[1, 2, 3]
OMEVersion = tx.Literal["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
CompressorTypeV1 = tx.Literal[
    "blosc", "gzip", "bz2", "lzma", "lz4", "pcodec", "zfpy", "zlib", "zstd",
    "none"
]
CompressorTypeV2 = CompressorTypeV1
CompressorTypeV3 = tx.Literal["blosc", "gzip", "none"]
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
Attributes = tx.MutableMapping[str, JSON]
FrozenAttributes = tx.Mapping[str, JSON]
AnyDriver = tx.Union[KnownDriver, str]
AnyZarrVersion = tx.Union[ZarrVersion, int]
AnyOMEVersion = tx.Union[OMEVersion, str]
AnyCompressorType = tx.Union[CompressorType, str]
CompressorOptions = tx.Mapping[str, tx.Union[float, str]]
PyramidFunction = tx.Callable[[npt.ArrayLike], npt.ArrayLike]
PyramidMode = tx.Union[KnownPyramidMode, PyramidFunction]


@register_converter(JSON)
class ToJson(Converter[JSON, JSON]):
    """
    A converter for JSON-compatible types.
    """

    def __call__(self, value: JSON) -> JSON:
        return json.loads(json.dumps(value))
