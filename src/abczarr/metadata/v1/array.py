__all__ = [
    "ArrayMetadata",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, eq_safenan, field
from abczarr.metadata.base import register_subclass
from abczarr.schemas.v1 import Codec

from .base import ArrayMetadataV1
from .codecs import CodecOptions

# locals
from .dtypes import DType

# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


@register_subclass(zarr_format=1, node_type="array")
@autofrozen(kw_only=True, extra_items=tz.FrozenJSON)
class ArrayMetadata(ArrayMetadataV1):

    # --- Required ----
    shape: tz.Shape
    chunks: tz.Shape
    dtype: DType
    compression: tx.Optional[Codec]
    compression_opts: tx.Optional[CodecOptions]
    fill_value: tx.Optional[tz.BuiltinNumber] = field(eq=eq_safenan)
    order: tz.MemoryOrder
