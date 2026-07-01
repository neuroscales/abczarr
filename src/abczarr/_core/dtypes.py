__all__ = [
    "DataTypeV2",
    "DataTypeV3",
    "asdtype",
    "to_zarr2",
    "to_zarr3",
]

import re
from collections import abc

import numpy as np
import numpy.typing as npt
import typing_extensions as tx


class RegexMatch(str):
    def __class_getitem__(cls, pattern: tx.Union[str, re.Pattern]) -> type:
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        return tx.Annotated[str, pattern]


T = tx.TypeVar("T")
BuiltinSequence = tx.Union[tx.Tuple[T, ...], tx.List[T]]

RawDataTypeV3 = RegexMatch[r"r\d+"]
BuiltinDataTypeV3 = tx.Literal[
    "bool",         # Boolean
    "int8",         # Integer in [-2^7, 2^7-1]
    "int16",        # Integer in [-2^15, 2^15-1]
    "int32",        # Integer in [-2^31, 2^31-1]
    "int64",        # Integer in [-2^63, 2^63-1]
    "uint8",        # Integer in [0, 2^8-1]
    "uint16",       # Integer in [0, 2^16-1]
    "uint32",       # Integer in [0, 2^32-1]
    "uint64",       # Integer in [0, 2^64-1]
    "float16",      # IEEE 754 half-precision floating point: sign bit, 5 bits exponent, 10 bits mantissa  (optionally supported)
    "float32",      # IEEE 754 single-precision floating point: sign bit, 8 bits exponent, 23 bits mantissa
    "float64",      # IEEE 754 double-precision floating point: sign bit, 11 bits exponent, 52 bits mantissa
    "complex64",    # real and complex components are each IEEE 754 single-precision floating point
    "complex128"    # real and complex components are each IEEE 754 double-precision floating point
]
DataTypeV3 = tx.Union[BuiltinDataTypeV3, RawDataTypeV3]

RegexDataTypeV2 = (
    r"^(?:"
    r"\|b1"                 # bool
    r"|[<>|][iu][1248]"     # int
    r"|[<>][f][248]"        # float
    r"|[<>][c][816]"        # complex
    r"|[<>|][mM][1248]"     # time
    r"(?:\[(?:h|m|s|ms|us|μs|ns|ps|fs|as|Y|M|W|D|nat|naT|nAt|nAT|Nat|NaT|NAt|NAT)\])?" # time unit
    r"|[<>|][SUV]\d+"       # array
    r")$"
)

ScalarDataTypeV2 = RegexMatch[
    r"^(?:"
    r"\|b1"                 # bool
    r"|[<>|][iu][1248]"     # int
    r"|[<>][f][248]"        # float
    r"|[<>][c][816]"        # complex
    r"|[<>|][mM][1248]"     # time
    r"(?:\[(?:h|m|s|ms|us|μs|ns|ps|fs|as|Y|M|W|D|nat|naT|nAt|nAT|Nat|NaT|NAt|NAT)\])?" # time unit
    r"|[<>|][SUV]\d+"       # array
    r")$"
]

DataTypeV2 = tx.Union[
    ScalarDataTypeV2,
    tx.Tuple[str, "DataTypeV2"],
    tx.Tuple[str, "DataTypeV2", int],
    tx.Tuple[str, "DataTypeV2", BuiltinSequence[int]],
    BuiltinSequence["DataTypeV2"],
]


def asdtype(
    dtype: tx.Union[npt.DTypeLike, tx.Mapping],
    type: tx.Optional[tx.Type[np.generic]] = None,
    kind: tx.Optional[tx.Union[str, re.Pattern]] = None
) -> np.dtype:
    """
    Convert a string or numpy dtype to a numpy dtype.

    Parameters
    ----------
    dtype : dtype-like
        The dtype to convert.
    type : type[np.generic], optional
        Check that the resulting dtype is a subclass of this type.
    kind : str or re.Pattern, optional
        Check that the resulting dtype has this kind.

    Returns
    -------
    np.dtype
        The converted numpy dtype.
    """
    # Our DType metadata -> dict
    if hasattr(dtype, "to_dict"):
        dtype = dtype.to_dict()

    # Dictionaries are Zarr v3 data type extensions
    if isinstance(dtype, abc.Mapping):
        name = dtype["name"]
        if not getattr(dtype, "configuration", None):
            dtype = name

        # Structured data type
        if name in ("struct", "structured"):
            fields = dtype["configuration"]["fields"]
            dtype = [
                (field["name"], asdtype(field["data_type"]))
                for field in fields
            ]

        # Time data type
        if name in ("numpy.datetime64", "numpy.timedelta64"):
            unit = dtype["configuration"]["unit"]
            scale = dtype["configuration"]["scale_factor"]
            time_type = name.split(".")[-1]
            dtype = f"{time_type}[{scale}{unit}]"

    dtype = np.dtype(dtype)

    if type is not None:
        if not issubclass(dtype.type, type):
            raise TypeError(f"Expected dtype of type {type}, got {dtype}")

    if isinstance(kind, re.Pattern):
        if not kind.match(dtype.kind):
            raise TypeError(f"Expected dtype of kind {kind}, got {dtype}")
    elif kind is not None:
        if dtype.kind != kind:
            raise TypeError(f"Expected dtype of kind {kind}, got {dtype}")

    return dtype


def to_zarr2(dtype: tx.Union[npt.DTypeLike, tx.Mapping]) -> DataTypeV2:
    """
    Convert a numpy dtype to a Zarr v2 data type.

    Parameters
    ----------
    dtype : dtype-like
        The dtype to convert.

    Returns
    -------
    DataTypeV2
        The converted Zarr v2 data type.
    """
    dtype = asdtype(dtype)

    dtype = dtype.descr
    if len(dtype) == 1 and dtype[0][0] == "":
        dtype = dtype[0][1]
    return dtype


def to_zarr3(dtype: tx.Union[npt.DTypeLike, tx.Mapping]) -> DataTypeV3:
    """
    Convert a numpy dtype to a Zarr v3 data type.

    Parameters
    ----------
    dtype : dtype-like
        The dtype to convert.

    Returns
    -------
    DataTypeV3
        The converted Zarr v3 data type.
    """
    dtype = asdtype(dtype)

    descr = dtype.descr
    if len(descr) == 1 and descr[0][0] == "":
        descr = descr[0][1]

    if isinstance(descr, list):
        fields = [
            {"name": name, "data_type": to_zarr3(subdtype)}
            for name, subdtype in descr
        ]
        return {"name": "struct", "configuration": {"fields": fields}}

    prefix_datetime64 = ("<M8", "|M8", ">M8")
    if descr.upper().startswith(prefix_datetime64):
        if dtype.startswith(prefix_datetime64):
            name = "numpy.datetime64"
        else:
            name = "numpy.timedelta64"
        if "[" in descr:
            unit = descr.split("[")[-1].split("]")[0]
            scale, unit = re.match(r"(\d+)(\w+)", unit).groups()
            scale = int(scale or 1)
            unit = unit or "generic"
        else:
            scale, unit = 1, "generic"
        return {
            "name": name,
            "configuration": {"unit": unit, "scale_factor": scale}
        }

    return dtype.name
