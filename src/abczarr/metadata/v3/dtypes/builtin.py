__all__ = ["DTypeBuiltin", "Raw"]

# stdlib
import re

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autofrozen, RegexConverter

# locals
from ...base import register_subclass
from .base import DType, _make_dtype_classes

# constants
RE_RAW = re.compile(r"r\d+")
DTYPES_BUILTIN = (
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
    "complex128",   # real and complex components are each IEEE 754 double-precision floating point
)


@autofrozen
class DTypeBuiltin(DType):
    ...


@register_subclass(name=RE_RAW)
@autofrozen
class Raw(DTypeBuiltin):
    """
    Raw data type, with a specified number of bits per element.
    """
    name: tx.Annotated[str, RegexConverter(RE_RAW)] = "r8"


__all__ += _make_dtype_classes(
    globals(),
    DTYPES_BUILTIN,
    base=DTypeBuiltin
)
