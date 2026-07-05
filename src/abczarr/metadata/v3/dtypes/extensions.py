__all__ = ["DTypeExtra", "Struct"]

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
    ...


@autofrozen
class StructField(DTypeConfigImpl):
    name: str
    data_type: DType


@autofrozen
class StructConfig(DTypeConfigImpl):
    fields: tx.Tuple[StructField, ...]


@register_subclass(name=re.compile(r"(?:struct|structured)"))
@autofrozen
class Struct(DTypeExtra):
    name: tx.Literal["struct", "structured"]
    configuration: StructConfig


@autofrozen
class NumpyTimeConfig(DTypeConfigImpl):
    unit: tx.Literal[
        "Y", "M", "W", "D", "h", "m", "s",
        "ms", "us", "μs", "ns", "ps", "fs", "as", "generic",
    ]
    scale_factor: int


@register_subclass(name="numpy.datetime64")
@autofrozen
class NumpyDatetime64(DTypeExtra):
    name: tx.Literal["numpy.datetime64"]
    configuration: NumpyTimeConfig


@register_subclass(name="numpy.timedelta64")
@autofrozen
class NumpyTimedelta64(DTypeExtra):
    name: tx.Literal["numpy.timedelta64"]
    configuration: NumpyTimeConfig


__all__ += _make_dtype_classes(
    globals(),
    DTYPES_EXTENSIONS,
    ignore=("struct", "structured", "numpy.datetime64", "numpy.timedelta64"),
    base=DTypeExtra
)
