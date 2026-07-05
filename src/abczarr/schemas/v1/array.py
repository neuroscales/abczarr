__all__ = ["Array"]

# stdlib
import typing_extensions as tx

# core
from abczarr._core import typing as tz

# locals
from .codecs import Codec, ValidCodecOptions


class Array(tx.TypedDict, extra_items=tz.JSON):

    # --- Required ----
    zarr_format: tx.Literal[1]
    shape:  tz.BuiltinSequence[int]
    chunks: tz.BuiltinSequence[int]
    dtype: tz.DataTypeV2
    compression: Codec
    compression_opts: ValidCodecOptions
    fill_value: tx.Optional[tz.BuiltinNumber]
    order: tz.MemoryOrder
