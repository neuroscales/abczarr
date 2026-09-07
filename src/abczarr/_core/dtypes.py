"""Conversion between numpy dtypes and the Zarr v2 and v3 data type
representations.

`asdtype` reads either representation, and any value numpy itself accepts,
into a numpy dtype. `to_zarr2` and `to_zarr3` go the other way, each
producing the JSON-serializable form its format version stores.
"""

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

from abczarr.errors import UnsupportedConversion

# Zarr v3 extension float dtypes that have no numpy scalar on their own but
# gain one once ``ml_dtypes`` is imported: importing that package registers
# these names with numpy, after which ``np.dtype("bfloat16")`` (and the
# float8/float6/float4 variants) resolves. The names match ml_dtypes' numpy
# registrations exactly. Complex extension floats (``complex_float32``, ...)
# are absent because ml_dtypes provides no scalar for them, so they keep the
# plain error regardless of whether ml_dtypes is installed. See
# https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
_ML_DTYPES_EXTENSION_NAMES = frozenset({
    "bfloat16",
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
})


class RegexMatch(str):
    """A string type hint, subscripted with the pattern a valid value must
    match.

    ``RegexMatch[r"r\\d+"]`` is `Annotated[str, pattern]`, where *pattern*
    is a compiled `re.Pattern`. A validator resolved from that hint checks
    a value against the pattern.
    """

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
    "float16",      # IEEE 754 half-precision floating point
    "float32",      # IEEE 754 single-precision floating point
    "float64",      # IEEE 754 double-precision floating point
    "complex64",    # real and complex components are each IEEE 754 single
    "complex128"    # real and complex components are each IEEE 754 double
]
DataTypeV3 = tx.Union[BuiltinDataTypeV3, RawDataTypeV3]

RegexDataTypeV2 = (
    r"^(?:"
    r"\|b1"                 # bool
    r"|[<>|][iu][1248]"     # int
    r"|[<>][f][248]"        # float
    r"|[<>][c][816]"        # complex
    r"|[<>|][mM][1248]"     # time
    r"(?:\[(?:h|m|s|ms|us|μs|ns|ps|fs|as|Y|M|W|D|nat|naT|nAt|nAT|Nat|NaT|NAt|NAT)\])?"  # noqa: E501 (time unit)
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
    r"(?:\[(?:h|m|s|ms|us|μs|ns|ps|fs|as|Y|M|W|D|nat|naT|nAt|nAT|Nat|NaT|NAt|NAT)\])?"  # noqa: E501 (time unit)
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
    """Resolve a numpy dtype from a Zarr v2 or v3 data type, or from
    anything numpy itself accepts.

    A Zarr v3 extension mapping (``{"name": ..., "configuration": ...}``)
    is unpacked first: a structured type builds each field recursively, a
    time type rebuilds numpy's ``datetime64``/``timedelta64`` spelling, and
    a fixed-length string type rebuilds numpy's ``U``/``S`` spelling. The
    variable-length ``"string"`` and ``"bytes"`` names resolve to numpy's
    ``object`` dtype, since numpy has no dedicated scalar for either.
    Anything else is handed to `numpy.dtype` directly, so an ordinary numpy
    dtype, dtype string, or `DType` object all resolve the same way.

    A handful of Zarr v3 extension floats (``bfloat16``, the float8
    variants) have no numpy scalar until ``ml_dtypes`` is imported. When
    the requested name is one of those, ``ml_dtypes`` is imported and
    resolution is retried before giving up.

    Parameters
    ----------
    dtype : dtype-like
        The dtype to resolve, in any of the forms above.
    type : type[np.generic], optional
        Require the resolved dtype's scalar type to be a subclass of this
        type.
    kind : str or re.Pattern, optional
        Require the resolved dtype's `numpy.dtype.kind` to equal this
        string, or to match this pattern.

    Returns
    -------
    np.dtype
        The resolved numpy dtype.

    Raises
    ------
    [UnsupportedConversion][abczarr.errors.UnsupportedConversion]
        When *dtype* cannot be resolved to a numpy dtype.
    TypeError
        When the resolved dtype fails the *type* or *kind* check.
    """
    # Our DType metadata -> dict
    if hasattr(dtype, "to_json"):
        dtype = dtype.to_json()

    # Dictionaries are Zarr v3 data type extensions
    if isinstance(dtype, abc.Mapping):
        name = dtype["name"]
        configuration = dtype.get("configuration")
        if not configuration:
            dtype = name

        # Structured data type
        if name in ("struct", "structured"):
            fields = configuration["fields"]
            dtype = [
                (field["name"], asdtype(field["data_type"]))
                for field in fields
            ]

        # Time data type
        if name in ("numpy.datetime64", "numpy.timedelta64"):
            unit = configuration["unit"]
            scale = configuration["scale_factor"]
            time_type = name.split(".")[-1]
            dtype = f"{time_type}[{scale}{unit}]"

        # Fixed-length string data types. ``length_bytes`` is the numpy
        # ``itemsize``: UTF-32 stores 4 bytes per code unit ("<U" length),
        # while bytes store one byte per character ("S" length). See
        # https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
        if name == "fixed_length_utf32":
            length = configuration["length_bytes"]
            dtype = "<U" + str(length // 4)
        if name == "null_terminated_bytes":
            length = configuration["length_bytes"]
            dtype = "S" + str(length)

    # Variable-length string and bytes. Zarr v3 spells these ``string``
    # (variable-length UTF-8) and ``bytes`` (variable-length bytes) -- a bare
    # name with no configuration, so a ``DType`` serializes to the plain
    # string here. numpy has no fixed-width scalar for either, so the
    # conventional Zarr representation -- the one zarr-python uses -- is numpy
    # ``object`` (``|O``) carrying a vlen codec (``vlen-utf8`` for ``string``,
    # ``vlen-bytes`` for ``bytes``). The array-metadata conversion surfaces
    # that codec as a v2 filter so the "is-a-string / is-a-bytes" tag numpy
    # ``object`` drops is restored. See
    # https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
    if isinstance(dtype, str) and dtype in ("string", "bytes"):
        dtype = "object"

    try:
        dtype = np.dtype(dtype)
    except TypeError as exc:
        # ``ml_dtypes`` provides a numpy scalar for a handful of exotic v3
        # extension floats, but only once it is imported (importing it
        # registers the names with numpy). If the requested name is one of
        # those, import ``ml_dtypes`` lazily and resolve again -- this keeps
        # resolution transparent for anyone who installed the optional extra
        # without also importing the package. If ``ml_dtypes`` is missing (or
        # too old to supply this name), point the error at the extra.
        name = str(dtype)
        if name in _ML_DTYPES_EXTENSION_NAMES:
            hint = "install abczarr[ml-dtypes] to enable this dtype"
            try:
                import ml_dtypes  # noqa: F401 (registers dtypes on import)

                dtype = np.dtype(name)
            except (ImportError, TypeError):
                raise UnsupportedConversion(name, 2, hint) from exc
        else:
            raise UnsupportedConversion(name, 2) from exc

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
    """Render a numpy dtype (or anything `asdtype` resolves) as a Zarr v2
    data type.

    A plain scalar dtype renders as its numpy string form (``"<f8"``). A
    structured dtype with one unnamed field unwraps to that field's own
    data type. A structured dtype with named fields renders as the
    nested ``(name, data_type[, shape])`` structure Zarr v2 uses for a
    structured array.

    Parameters
    ----------
    dtype : dtype-like
        The dtype to render.

    Returns
    -------
    DataTypeV2
        The Zarr v2 data type.
    """
    dtype = asdtype(dtype)

    dtype = dtype.descr
    if len(dtype) == 1 and dtype[0][0] == "":
        dtype = dtype[0][1]
    return dtype


def to_zarr3(dtype: tx.Union[npt.DTypeLike, tx.Mapping]) -> DataTypeV3:
    """Render a numpy dtype (or anything `asdtype` resolves) as a Zarr v3
    data type.

    A dtype with a builtin Zarr v3 name (``"int32"``, ``"float64"``, ...)
    renders as that name. A structured dtype renders as a ``"struct"``
    extension type, with each field's own data type rendered recursively.
    A ``datetime64``/``timedelta64`` dtype renders as the matching
    ``"numpy.datetime64"``/``"numpy.timedelta64"`` extension type, carrying
    its unit and scale factor. A fixed-length Unicode or byte dtype (``U``
    or ``S``) renders as the ``"fixed_length_utf32"`` or
    ``"null_terminated_bytes"`` extension type, carrying its storage size
    in bytes.

    Parameters
    ----------
    dtype : dtype-like
        The dtype to render.

    Returns
    -------
    DataTypeV3
        The Zarr v3 data type.
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
        if descr.startswith(prefix_datetime64):
            name = "numpy.datetime64"
        else:
            name = "numpy.timedelta64"
        if "[" in descr:
            unit = descr.split("[")[-1].split("]")[0]
            scale, unit = re.match(r"(\d*)(\w+)", unit).groups()
            scale = int(scale or 1)
            unit = unit or "generic"
        else:
            scale, unit = 1, "generic"
        return {
            "name": name,
            "configuration": {"unit": unit, "scale_factor": scale}
        }

    # Fixed-length Unicode ('U') and byte ('S') numpy dtypes map to Zarr v3
    # extension data types carrying the storage size in ``length_bytes``.
    # These extension types are marked "unstable / not finalized" upstream;
    # the shape here matches zarr-python's representation. See
    # https://github.com/zarr-developers/zarr-extensions/tree/main/data-types
    if dtype.kind == "U":
        # numpy stores each UTF-32 code unit in 4 bytes, so ``itemsize`` is
        # already the byte length ("<U5" -> length_bytes 20).
        return {
            "name": "fixed_length_utf32",
            "configuration": {"length_bytes": dtype.itemsize},
        }
    if dtype.kind == "S":
        # numpy stores one byte per character ("S3" -> length_bytes 3).
        return {
            "name": "null_terminated_bytes",
            "configuration": {"length_bytes": dtype.itemsize},
        }

    return dtype.name
