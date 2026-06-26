# stdlib
import os
from numbers import Number

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# constants
FILE_MODES = ("r", "r+", "a", "w", "w-")
LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
COMPRESSORS_V2 = ("blosc", "zlib", "bz2", "zstd", "none")
COMPRESSORS_V3 = ("blosc", "gzip", "zstd", "none")
ZARR_VERSIONS = (2, 3)
OME_VERSIONS = ("0.4", "0.5", "0.6")
DRIVERS = ("zarr-python", "tensorstore", "zarrita")

# General types
T = tx.TypeVar("T")
OneOrMore = tx.Union[T, tx.Sequence[T]]
JSON = tx.Any
Attributes = tx.MutableMapping[str, JSON]
FrozenAttributes = tx.Mapping[str, JSON]
Shape = tx.Tuple[int, ...]
ShapeLike = tx.Sequence[int]
PathLike = tx.Union[str, os.PathLike]
Value = tx.Union[Number, bool, np.number]
LogLevel = tx.Literal["debug", "info", "warning", "error", "critical"]
FileMode = tx.Literal["r", "r+", "a", "w", "w-"]

# Zarr-specific types
KnownDriver = tx.Literal["zarr-python", "tensorstore", "zarrita"]
AnyDriver = tx.Union[KnownDriver, str]
ZarrVersion = tx.Literal[2, 3]
AnyZarrVersion = tx.Union[ZarrVersion, int]
OMEVersion = tx.Literal["0.4", "0.5", "0.6"]
AnyOMEVersion = tx.Union[OMEVersion, str]
ZarrNodeType = tx.Literal["group", "array"]
CompressorType = tx.Literal["blosc", "zlib", "bz2", "zstd", "none"]
AnyCompressorType = tx.Union[CompressorType, str]
CompressorOptions = tx.Mapping[str, tx.Union[float, str]]
DimensionSeparator = tx.Literal[".", "/"]
ArrayOrder = tx.Literal["C", "F"]
PyramidMode = tx.Union[
    tx.Literal["mean", "median"],
    tx.Callable[[npt.ArrayLike], Value]
]
SpatialAxisName = tx.Literal["x", "y", "z"]
TimeAxisName = tx.Literal["t"]
ChannelAxisName = tx.Literal["c"]
AxisName = tx.Union[SpatialAxisName, TimeAxisName, ChannelAxisName]
