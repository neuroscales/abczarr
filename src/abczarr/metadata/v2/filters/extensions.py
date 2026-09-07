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
    """Rounds a float's mantissa to ``keepbits`` bits, zeroing the rest.

    A lossy transform that improves the compressibility of floating-point
    data by discarding low-order precision the data does not need.
    """

    id: tx.Literal["bitround"]
    keepbits: int = 1

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import BitroundCodec
            return BitroundCodec.from_json({
                "name": self.id,
                "configuration": {
                    "keepbits": self.keepbits
                }
            })
        raise ValueError(f"Unsupported Zarr version: {version}")


@register_subclass(id="packbits")
@autofrozen
class PackBitsFilter(FilterImpl):
    """Packs a boolean array down to one bit per element for storage."""

    id: tx.Literal["packbits"]

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import PackBitsCodec
            return PackBitsCodec.from_json({
                "name": self.id,
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
    """Quantizes values as ``(value - offset) * scale``, stored as ``astype``.

    Recovers an approximation of the original value on decode by
    reversing the scale and offset; a lossy transform useful for storing
    a bounded floating-point range in fewer bits.
    """

    id: tx.Literal["fixedscaleoffset"]
    offset: float
    scale: float
    dtype: np.dtype
    astype: tx.Optional[np.dtype]

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            # self.id ("fixedscaleoffset") is the numcodecs id, not the v3
            # codec name; ScaleOffsetCodec supplies its own single name.
            from abczarr.metadata.v3 import ScaleOffsetCodec
            return ScaleOffsetCodec.from_json({
                "configuration": {
                    "offset": self.offset,
                    "scale": self.scale,
                }
            })
        raise ValueError(f"Unsupported Zarr version: {version}")


@register_subclass(id="astype")
@autofrozen
class AsTypeFilter(FilterImpl):
    """Casts an array to ``encode_dtype`` for storage, and back on decode.

    ``decode_dtype``, when given, is the dtype values are cast back to on
    read; otherwise the array's own dtype is used.
    """

    id: tx.Literal["astype"]
    encode_dtype: np.dtype
    decode_dtype: tx.Optional[np.dtype]

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            # self.id ("astype") is the numcodecs id, not the v3 codec name;
            # CastValueCodec supplies its own single name.
            from abczarr.metadata.v3 import CastValueCodec
            return CastValueCodec.from_json({
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
    """Encodes each value in ``labels`` as its index, and back on decode.

    Turns a categorical array into a compact integer array of category
    indices.
    """

    id: tx.Literal["categorize"]
    labels: tx.Tuple[str, ...]
    dtype: np.dtype
    astype: tx.Optional[np.dtype]
    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            # self.id ("categorize") is the numcodecs id, not the v3 codec
            # name; CastValueCodec supplies its own single name.
            from abczarr.metadata.v3 import CastValueCodec
            return CastValueCodec.from_json({
                "configuration": {
                    "data_type": dtype_to_zarr3(self.dtype),
                    "rounding": "towards-zero",
                    "out_of_range": "wrap",
                    "scalar_map": {
                        # Materialize each pair: ``reversed`` returns a
                        # single-use iterator, so a list of them yields its
                        # values only on the first read and nothing after.
                        "encode": [
                            [label, index]
                            for index, label in enumerate(self.labels)
                        ],
                        "decode": [
                            [index, label]
                            for index, label in enumerate(self.labels)
                        ],
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
    """Stores each value as its difference from the previous one.

    Improves compressibility of arrays whose values change gradually
    along the last axis, such as a monotonic coordinate.
    """

    id: tx.Literal["delta"]
    dtype: np.dtype
    astype: tx.Optional[np.dtype]


@register_subclass(id="quantize")
@autofrozen
class QuantizeFilter(FilterImpl):
    """Rounds floating-point values to ``digits`` decimal digits.

    A lossy transform that improves compressibility by discarding
    precision the data does not need.
    """

    id: tx.Literal["quantize"]
    digits: int
    dtype: np.dtype
    astype: tx.Optional[np.dtype]


@register_subclass(id="shuffle")
@autofrozen
class Shuffle(FilterImpl):
    """Reorders each element's bytes to group same-significance bytes together.

    Groups the Nth byte of every element (``elementsize`` bytes wide) so
    that similar bytes sit next to each other, which a downstream
    compressor typically compresses better than the original interleaving.
    """

    id: tx.Literal["shuffle"]
    elementsize: tx.Optional[int]
