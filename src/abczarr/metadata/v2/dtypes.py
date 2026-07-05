__all__ = ["DType", "ScalarDType", "StructDType"]

# stdlib
from functools import wraps

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core.auto._typing import DTYPE_LIKE
from abczarr._core.auto.converters import Converter, register_converter
from abczarr._core.dtypes import asdtype
from abczarr._core.dtypes import to_zarr2 as dtype_to_zarr2


class DType:

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
            from abczarr.metadata.v3.dtypes import DType as DTypeV3
            return DTypeV3(self.numpy)
        else:
            raise ValueError(f"Unsupported version: {version}")

    @property
    def numpy(self) -> np.dtype:
        """
        Return the corresponding numpy dtype.
        """
        return asdtype(self)


class ScalarDType(str, DType):

    def __new__(cls, value: str) -> tx.Self:
        value = dtype_to_zarr2(asdtype(value))
        if not isinstance(value, str):
            raise TypeError(f"Cannot convert {value!r} to ScalarDType")
        return str.__new__(cls, value)


def _immutable(self: "StructDType", *args, **kwargs) -> None:
    raise TypeError(f"{self.__class__.__name__} is immutable")


class StructDType(list, DType):

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

    DEFAULT = DType
    FALLBACK = DType

    @classmethod
    def like(cls, value: DTYPE_LIKE) -> bool:
        return DTYPE_LIKE

    def __call__(self, value: DTYPE_LIKE) -> DType:
        if isinstance(value, self.origin):
            return value
        return self.fallback(value)
