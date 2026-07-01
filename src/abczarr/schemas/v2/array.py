__all__ = ["Array"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# locals
from .codecs import ValidCodec
from .filters import ValidFilter


class Array(tx.TypedDict, extra_items=tz.JSON):

    # --- Required ----
    zarr_format: tx.Literal[2]
    shape:  tz.BuiltinSequence[int]
    chunks: tz.BuiltinSequence[int]
    dtype: tz.DataTypeV2
    compressor: tx.Optional[ValidCodec]
    fill_value: tx.Optional[tz.BuiltinNumber]
    order: tz.MemoryOrder
    filters: tz.BuiltinSequence[ValidFilter]

    # --- Optional ----
    dimension_separator: tx.NotRequired[tz.DimensionSeparator]
