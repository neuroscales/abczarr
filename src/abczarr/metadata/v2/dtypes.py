__all__ = ["DType", "ScalarDType", "StructDType"]

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core.attrs import Converter, register_converter, DTYPELIKE
from abczarr._core.dtypes import asdtype, to_zarr2 as dtype_to_zarr2


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


class StructDType(list, DType):

    def __new__(cls, value: tx.Iterable[tx.Tuple[str, str]]) -> tx.Self:
        items = []
        for item in value:
            name, dtype = item
            name = str(name)
            dtype = DType(dtype)
            items.append((name, dtype))
        return list.__new__(cls, items)


@register_converter(DType)
class DTypeConverter(Converter[DType, DTYPELIKE]):

    DEFAULT = DType
    FALLBACK = DType

    @classmethod
    def like(cls, value: DTYPELIKE) -> bool:
        return DTYPELIKE

    def __call__(self, value: DTYPELIKE) -> DType:
        if isinstance(value, self.origin):
            return value
        return self.fallback(value)
