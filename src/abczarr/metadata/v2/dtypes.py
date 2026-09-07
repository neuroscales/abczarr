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
    """A Zarr v2 data type: a numpy dtype string, or a structured
    field list.

    Constructing `DType(value)` returns a
    [ScalarDType][abczarr.metadata.v2.dtypes.ScalarDType] for a plain
    dtype, such as ``"<f8"``, or a
    [StructDType][abczarr.metadata.v2.dtypes.StructDType] for a
    structured one, whichever `value` describes.

    Parameters
    ----------
    value : dtype-like
        A numpy dtype, a dtype string, or a structured field list, to
        convert to the matching `DType` subclass.
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
        """Convert this data type to another Zarr version.

        Parameters
        ----------
        version : int
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        DType
            This data type unchanged for version 1 or 2, since v1
            and v2 share the same dtype model, or the equivalent v3
            data type for version 3.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3.
        """
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
        """The equivalent `numpy.dtype`."""
        return asdtype(self)


class ScalarDType(str, DType):
    """A Zarr v2 scalar data type: a numpy dtype string, for example
    ``"<f8"``.

    A `ScalarDType` encodes the byte order, kind and item size the way
    `.zarray`'s `dtype` field does.

    Parameters
    ----------
    value : str
        A numpy dtype string in Zarr v2's encoding, such as ``"<f8"``.
    """

    def __new__(cls, value: str) -> tx.Self:
        value = dtype_to_zarr2(asdtype(value))
        if not isinstance(value, str):
            raise TypeError(f"Cannot convert {value!r} to ScalarDType")
        return str.__new__(cls, value)


def _immutable(self: "StructDType", *args, **kwargs) -> None:
    """Refuse to mutate `self`, since a `StructDType` is immutable."""
    raise TypeError(f"{self.__class__.__name__} is immutable")


class StructDType(list, DType):
    """A Zarr v2 structured data type: an ordered list of
    ``(name, dtype)`` fields.

    A `StructDType` mirrors numpy's structured dtype list form and is
    immutable. Every list-mutation method raises `TypeError` instead
    of changing the value after construction.

    Parameters
    ----------
    value : iterable of tuple
        The ``(name, dtype)`` pairs the structured type is built
        from, in field order.
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
    """Converts a dtype-like value to a
    [DType][abczarr.metadata.v2.dtypes.DType].

    Accepts a numpy dtype, a dtype string, or a structured field
    list.
    """

    DEFAULT = DType
    FALLBACK = DType

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        return DTYPE_LIKE

    def __call__(self, value: DTYPE_LIKE) -> DType:
        if isinstance(value, self.origin):
            return value
        return self.fallback(value)
