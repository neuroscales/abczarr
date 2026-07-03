__all__ = [
    "ArrayMetadata",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, eq_safenan, field
from abczarr._core.metadata import register_subclass
from abczarr.metadata import base

# locals
from .dtypes import DType
from .codecs import Codec
from .filters import Filter
from .base import ArrayMetadataV2


# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


@register_subclass(zarr_format=2, node_type="array")
@autofrozen(kw_only=True)
class ArrayMetadata(ArrayMetadataV2):

    # --- Required ----
    shape: tz.Shape
    chunks: tz.Shape
    dtype: DType
    compressor: tx.Optional[Codec]
    fill_value: tx.Optional[tz.BuiltinNumber] = field(eq=eq_safenan)
    order: tz.MemoryOrder
    filters: tx.Tuple[Filter, ...]

    # --- Optional ----
    dimension_separator: tx.Optional[tz.DimensionSeparator]

    # --- Conversion ---

    def to_version(self, version: tz.ZarrVersion) -> base.ArrayMetadata:
        if version == 1:
            return self._to_v1()
        if version == 2:
            return self
        if version == 3:
            return self._to_v3()
        else:
            raise ValueError(f"Unsupported version: {version}")

    def _to_v1(self) -> base.ArrayMetadata:
        from abczarr.metadata import v1

        compressor = compressor_opt = None
        if self.compressor:
            self.compressor: Codec
            compressor = self.compressor.id
            compressor_opt = self.compressor.to_version(1)

        return v1.ArrayMetadata(
            shape=self.shape,
            chunks=self.chunks,
            dtype=self.dtype.to_version(1),
            compression=compressor,
            compression_opts=compressor_opt,
            fill_value=self.fill_value,
            filters=self.filters,
        )

    def _to_v3(self) -> base.ArrayMetadata:
        from abczarr.metadata import v3

        separator = self.dimension_separator or "."
        chunk_grid = v3.RegularChunkGrid(configuration=self.chunks)
        chunk_key_encoding = v3.V2ChunkKeyEncoding(separator)
        codecs = [c.to_version(3) for c in self.filters]
        if self.compressor:
            codecs.append(self.compressor.to_version(3))

        return v3.ArrayMetadata(
            shape=self.shape,
            data_type=self.dtype.to_version(3),
            chunk_grid=chunk_grid,
            chunk_key_encoding=chunk_key_encoding,
            fill_value=self.fill_value,
            codecs=codecs,
        )
