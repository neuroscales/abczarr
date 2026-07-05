__all__ = [
    "BitroundFilter",
    "PackBitsFilter",
    "ScaleOffsetFilter",
    "AsTypeFilter",
    "DeltaFilter",
    "QuantizeFilter",
    "CategorizeFilter",
    "Shuffle",
]

# dependencies
import numpy as np
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr._core.dtypes import to_zarr3 as dtype_to_zarr3
from abczarr._core.metadata import register_subclass

# locals
from .base import FilterImpl

# -
# Filters that have a compatible v3 Codec
# -


@register_subclass(id="bitround")
@autofrozen
class BitroundFilter(FilterImpl):
    id: tx.Literal["bitround"]
    keepbits: int = 1

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import BitroundCodec
            return BitroundCodec.from_dict({
                "name": self.name,
                "configuration": {
                    "keepbits": self.keepbits
                }
            })
        raise ValueError(f"Unsupported Zarr version: {version}")


@register_subclass(id="packbits")
@autofrozen
class PackBitsFilter(FilterImpl):
    id: tx.Literal["packbits"]

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import PackBitsCodec
            return PackBitsCodec.from_dict({
                "name": self.name,
                "configuration": {
                    "padding_encoding": "first_byte",
                    "first_bit": None,
                    "last_bit": None
                }
            })
        raise ValueError(f"Unsupported Zarr version: {version}")


@register_subclass(id="fixedscaleoffset")
@autofrozen
class ScaleOffsetFilter(FilterImpl):
    id: tx.Literal["fixedscaleoffset"]
    offset: float
    scale: float
    dtype: np.dtype
    astype: tx.Optional[np.dtype]

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import ScaleOffsetCodec
            return ScaleOffsetCodec.from_dict({
                "name": self.name,
                "configuration": {
                    "offset": self.offset,
                    "scale": self.scale,
                }
            })
        raise ValueError(f"Unsupported Zarr version: {version}")


@register_subclass(id="astype")
@autofrozen
class AsTypeFilter(FilterImpl):
    id: tx.Literal["astype"]
    encode_dtype: np.dtype
    decode_dtype: tx.Optional[np.dtype]

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import CastValueCodec
            return CastValueCodec.from_dict({
                "name": self.name,
                "configuration": {
                    "data_type": dtype_to_zarr3(self.encode_dtype),
                    "rounding": "towards-zero",
                    "out_of_range": "wrap",
                }
            })
        raise ValueError(f"Unsupported Zarr version: {version}")


@register_subclass(id="categorize")
@autofrozen
class CategorizeFilter(FilterImpl):
    id: tx.Literal["categorize"]
    labels: tx.Tuple[str, ...]
    dtype: np.dtype
    astype: tx.Optional[np.dtype]
    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import CastValueCodec
            return CastValueCodec.from_dict({
                "name": self.name,
                "configuration": {
                    "data_type": dtype_to_zarr3(self.dtype),
                    "rounding": "towards-zero",
                    "out_of_range": "wrap",
                    "scalar_map": {
                        "encode": list(map(reversed, enumerate(self.labels))),
                        "decode": list(enumerate(self.labels)),
                    }
                }
            })
        raise ValueError(f"Unsupported Zarr version: {version}")


# -
# Filters that do not have a compatible v3 Codec
# -


@register_subclass(id="delta")
@autofrozen
class DeltaFilter(FilterImpl):
    id: tx.Literal["delta"]
    dtype: np.dtype
    astype: tx.Optional[np.dtype]


@register_subclass(id="quantize")
@autofrozen
class QuantizeFilter(FilterImpl):
    id: tx.Literal["quantize"]
    digits: int
    dtype: np.dtype
    astype: tx.Optional[np.dtype]


@register_subclass(id="shuffle")
@autofrozen
class Shuffle(FilterImpl):
    id: tx.Literal["shuffle"]
    elementsize: tx.Optional[int]
