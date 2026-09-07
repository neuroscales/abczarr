"""Extension data types for Zarr v3, beyond the required core set.

Includes structured records, extended-precision and reduced-precision
floating-point types, numpy's datetime and timedelta types, and
fixed-length string and byte types.
"""

__all__ = [
    "DTypeExtra",
    "StructField",
    "StructConfig",
    "Struct",
    "NumpyTimeConfig",
    "FixedLengthConfig",
]

# stdlib
import re

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto import autofrozen

# locals
from ...base import register_subclass
from .base import DType, DTypeConfigImpl, DTypeImpl, _make_dtype_classes

# constants
DTYPES_EXTENSIONS = (
    "bytes",
    "bfloat16",
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "complex_bfloat16",
    "complex_float16",
    "complex_float32",
    "complex_float64",
    "complex_float4_e2m1fn",
    "complex_float6_e2m3fn",
    "complex_float6_e3m2fn",
    "complex_float8_e3m4",
    "complex_float8_e4m3",
    "complex_float8_e4m3b11fnuz",
    "complex_float8_e4m3fnuz",
    "complex_float8_e5m2",
    "complex_float8_e5m2fnuz",
    "complex_float8_e8m0fnu",
    "numpy.datetime64",
    "numpy.timedelta64",
    "string",
    "struct", "structured",
)


@autofrozen
class DTypeExtra(DTypeImpl):
    """This class is the base for a Zarr v3 extension data type, one
    registered outside the core spec."""


@autofrozen
class StructField(DTypeConfigImpl):
    """Represents one named field of a struct data type: its name and
    its data type.
    """

    name: str
    """The field's name."""
    data_type: DType
    """The field's data type."""


@autofrozen
class StructConfig(DTypeConfigImpl):
    """Holds the struct data type's parameters: the ordered list of
    fields.
    """

    fields: tx.Tuple[StructField, ...]
    """The struct's fields, in record order."""


@register_subclass(name=re.compile(r"(?:struct|structured)"))
@autofrozen
class Struct(DTypeExtra):
    """Represents a structured data type: a fixed, ordered sequence of
    named fields.

    Each element is a record combining every field's value, mirroring
    numpy's structured dtype.
    """

    name: tx.Literal["struct", "structured"]
    """Either ``"struct"`` or ``"structured"``."""
    configuration: StructConfig
    """The struct's fields."""


@autofrozen
class NumpyTimeConfig(DTypeConfigImpl):
    """Holds the datetime/timedelta parameters: the time unit and its
    scale factor.

    A value counts `scale_factor` multiples of `unit`. A
    `scale_factor` of 10 with `unit` ``"s"`` counts in tens of
    seconds, matching numpy's datetime64/timedelta64 resolution.
    """

    unit: tx.Literal[
        "Y", "M", "W", "D", "h", "m", "s",
        "ms", "us", "μs", "ns", "ps", "fs", "as", "generic",
    ]
    """The base time unit, such as ``"s"`` or ``"ns"``."""
    scale_factor: int
    """The number of `unit`s one increment counts."""


@register_subclass(name="numpy.datetime64")
@autofrozen
class NumpyDatetime64(DTypeExtra):
    """Represents a point in time, at the resolution `configuration`
    names.

    Mirrors numpy's `datetime64`: an integer count of the configured
    time unit since the Unix epoch.
    """

    name: tx.Literal["numpy.datetime64"]
    """Always ``"numpy.datetime64"``."""
    configuration: NumpyTimeConfig
    """The time unit and scale factor."""


@register_subclass(name="numpy.timedelta64")
@autofrozen
class NumpyTimedelta64(DTypeExtra):
    """Represents a duration, at the resolution `configuration` names.

    Mirrors numpy's `timedelta64`: an integer count of the configured
    time unit.
    """

    name: tx.Literal["numpy.timedelta64"]
    """Always ``"numpy.timedelta64"``."""
    configuration: NumpyTimeConfig
    """The time unit and scale factor."""


# Fixed-length string data types. These carry a ``length_bytes`` and mirror
# the numpy fixed-length Unicode (``<U``) and byte (``S``) dtypes:
# ``fixed_length_utf32`` <-> numpy ``<U{n}`` (``length_bytes == n * 4``) and
# ``null_terminated_bytes`` <-> numpy ``S{n}`` (``length_bytes == n``). These
# extension types are marked "unstable / not finalized" upstream; the shape
# here matches zarr-python's representation. See
# https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
@autofrozen
class FixedLengthConfig(DTypeConfigImpl):
    """Holds the fixed-length data type's parameters: the element's
    size in bytes.
    """

    length_bytes: int
    """The size, in bytes, of one element."""


@register_subclass(name="fixed_length_utf32")
@autofrozen
class FixedLengthUtf32(DTypeExtra):
    """Represents a fixed-length UTF-32 string, mirroring numpy's
    `<U{n}` dtype.

    `configuration.length_bytes` is four times the character count.
    """

    name: tx.Literal["fixed_length_utf32"]
    """Always ``"fixed_length_utf32"``."""
    configuration: FixedLengthConfig
    """The element's size in bytes."""


@register_subclass(name="null_terminated_bytes")
@autofrozen
class NullTerminatedBytes(DTypeExtra):
    """Represents a fixed-length byte string, mirroring numpy's
    `S{n}` dtype.
    """

    name: tx.Literal["null_terminated_bytes"]
    """Always ``"null_terminated_bytes"``."""
    configuration: FixedLengthConfig
    """The element's size in bytes."""


__all__ += _make_dtype_classes(
    globals(),
    DTYPES_EXTENSIONS,
    ignore=("struct", "structured", "numpy.datetime64", "numpy.timedelta64"),
    base=DTypeExtra
)
