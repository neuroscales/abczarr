__all__ = [
    "ArrayMetadata",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, eq_safenan, field
from abczarr.metadata import base
from abczarr.metadata.base import ConversionPolicy, register_subclass
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

    # --- Conversion ---

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: ConversionPolicy = "lossy",
    ) -> base.ArrayMetadata:
        if version == 1:
            return self
        if version == 2:
            return self._to_v2(policy)
        if version == 3:
            # route through v2 -- v1 and v2 share the numcodecs model
            return self._to_v2(policy).to_version(3, policy)
        raise ValueError(f"Unsupported version: {version}")

    def _to_v2(self, policy: ConversionPolicy = "lossy") -> base.ArrayMetadata:
        from abczarr.metadata import v2

        # v1 splits the codec into a name + options; v2 keeps them together.
        compressor = None
        if self.compression:
            options = dict(self.compression_opts or {})
            compressor = {"id": self.compression, **options}

        return v2.ArrayMetadata(
            shape=self.shape,
            chunks=self.chunks,
            dtype=self.dtype,
            compressor=compressor,
            fill_value=self.fill_value,
            order=self.order,
            filters=(),
            dimension_separator=".",
            attributes=self.attributes,
        )
