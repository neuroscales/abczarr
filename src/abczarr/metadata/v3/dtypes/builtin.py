"""The core Zarr v3 data types every implementation must support.

Includes the boolean, integer, floating-point and complex types, and
the raw fixed-bit-width type.
"""

__all__ = ["DTypeBuiltin", "Raw"]

# stdlib
import re

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto import ToRegexMatch, autofrozen

# locals
from ...base import register_subclass
from .base import DTypeImpl, _make_dtype_classes

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
    "float16",      # IEEE 754 half-precision floating point
    "float32",      # IEEE 754 single-precision floating point
    "float64",      # IEEE 754 double-precision floating point
    "complex64",    # real and complex components are each IEEE 754 single
    "complex128",   # real and complex components are each IEEE 754 double
)


@autofrozen
class DTypeBuiltin(DTypeImpl):
    """This class is the base for a core Zarr v3 data type, one every
    implementation must support."""


@register_subclass(name=RE_RAW)
@autofrozen
class Raw(DTypeBuiltin):
    """An opaque sequence of bits, with a fixed number of bits per
    element.

    Attributes
    ----------
    name : str
        The bit width, spelled as ``"r"`` followed by the number of
        bits, such as ``"r8"`` for one byte per element.
    """

    name: tx.Annotated[str, ToRegexMatch(RE_RAW)] = "r8"


__all__ += _make_dtype_classes(
    globals(),
    DTYPES_BUILTIN,
    base=DTypeBuiltin
)
