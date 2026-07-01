__all__ = [
    "ArrayMetadata",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autofrozen, eq_safenan, field
from abczarr.schemas.v1 import Codec
from abczarr.metadata.base import register_subclass

# locals
from .dtypes import DType
from .codecs import CodecOptions
from .base import ArrayMetadataV1


# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


@register_subclass(zarr_format=1, node_type="array")
@autofrozen(kw_only=True, extra_items=tz.JSON)
class ArrayMetadata(ArrayMetadataV1):

    # --- Required ----
    shape: tz.Shape
    chunks: tz.Shape
    dtype: DType
    compression: tx.Optional[Codec]
    compression_opts: tx.Optional[CodecOptions]
    fill_value: tx.Optional[tz.BuiltinNumber] = field(eq=eq_safenan)
    order: tz.MemoryOrder
