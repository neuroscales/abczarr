__all__ = ["DType", "ScalarDType", "StructDType"]

# stdlib
from functools import wraps

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core.auto._typing import DTYPE_LIKE
from abczarr._core.auto.converters import Converter, register_converter
from abczarr._core.dtypes import asdtype, to_zarr3
from abczarr._core.dtypes import to_zarr2 as dtype_to_zarr2


class DType:
    """A v2 data type is either a numpy dtype string or a structured
    field list.

    Constructing ``DType(value)`` returns a
    [`ScalarDType`][abczarr.metadata.v2.dtypes.ScalarDType] for a plain
    dtype (e.g. ``"<f8"``) or a
    [`StructDType`][abczarr.metadata.v2.dtypes.StructDType] for a
    structured one, inferred from *value*.
    """

    def __new__(cls, value: tx.Any) -> tx.Self:
        if cls is DType:
            value = dtype_to_zarr2(asdtype(value))
            if isinstance(value, str):
                return ScalarDType(value)
            elif isinstance(value, list):
                return StructDType(value)
            raise TypeError(f"Cannot convert {value!r} to DType")
        return super().__new__(cls, value)

    def to_version(self, version: int) -> tx.Any:
        if version in (1, 2):
            return self
        elif version == 3:
            # local import: v2 and v3 dtypes reference each other for
            # cross-version conversion, so a module-level import is a cycle
            from abczarr.metadata.v3.dtypes import DType as DTypeV3
            # A v3 data type is keyed by its zarr name ("float64"), not by a
            # numpy object, so build it from the canonical v3 name/spec.
            spec = to_zarr3(self.numpy)
            if isinstance(spec, str):
                return DTypeV3(spec)
            return DTypeV3(**spec)
        else:
            raise ValueError(f"Unsupported version: {version}")

    @property
    def numpy(self) -> np.dtype:
        """
        Return the corresponding numpy dtype.
        """
        return asdtype(self)


class ScalarDType(str, DType):
    """A v2 scalar data type is a numpy dtype string, for example
    ``"<f8"``.

    A `ScalarDType` encodes the byte order, kind and item size the way
    ``.zarray``'s ``dtype`` field does.
    """

    def __new__(cls, value: str) -> tx.Self:
        value = dtype_to_zarr2(asdtype(value))
        if not isinstance(value, str):
            raise TypeError(f"Cannot convert {value!r} to ScalarDType")
        return str.__new__(cls, value)


def _immutable(self: "StructDType", *args, **kwargs) -> None:
    raise TypeError(f"{self.__class__.__name__} is immutable")


class StructDType(list, DType):
    """A v2 structured data type is an ordered list of ``(name, dtype)``
    fields.

    A `StructDType` mirrors numpy's structured dtype list form and is
    immutable. The list-mutation methods raise instead of changing the
    value after construction.
    """

    def __new__(cls, value: tx.Iterable[tx.Tuple[str, str]]) -> tx.Self:
        items = []
        for item in value:
            name, dtype = item
            name = str(name)
            dtype = DType(dtype)
            items.append((name, dtype))
        return list.__new__(cls, items)

    # Make the list immutable
    __setitem__ = wraps(list.__setitem__)(_immutable)
    __delitem__ = wraps(list.__delitem__)(_immutable)
    __iadd__ = wraps(list.__iadd__)(_immutable)
    __imul__ = wraps(list.__imul__)(_immutable)
    append = wraps(list.append)(_immutable)
    extend = wraps(list.extend)(_immutable)
    pop = wraps(list.pop)(_immutable)
    clear = wraps(list.clear)(_immutable)
    insert = wraps(list.insert)(_immutable)
    remove = wraps(list.remove)(_immutable)
    reverse = wraps(list.reverse)(_immutable)


@register_converter(DType)
class DTypeConverter(Converter[DType, DTYPE_LIKE]):
    """Converts a dtype-like value (a numpy dtype, string, or field list) to
    a [`DType`][abczarr.metadata.v2.dtypes.DType]."""

    DEFAULT = DType
    FALLBACK = DType

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        return DTYPE_LIKE

    def __call__(self, value: DTYPE_LIKE) -> DType:
        if isinstance(value, self.origin):
            return value
        return self.fallback(value)
