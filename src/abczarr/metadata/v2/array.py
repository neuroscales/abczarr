__all__ = [
    "ArrayMetadata",
]

# stdlib
import sys

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, eq_safenan, field
from abczarr._core.metadata import register_subclass
from abczarr.metadata import base
from abczarr.metadata.base import ConversionPolicy

from .base import ArrayMetadataV2
from .codecs import Codec

# locals
from .dtypes import DType
from .filters import Filter

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

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: ConversionPolicy = "lossy",
    ) -> base.ArrayMetadata:
        if version == 1:
            return self._to_v1(policy)
        if version == 2:
            return self
        if version == 3:
            return self._to_v3(policy)
        else:
            raise ValueError(f"Unsupported version: {version}")

    def _to_v1(self, policy: ConversionPolicy = "lossy") -> base.ArrayMetadata:
        from abczarr.metadata import v1

        # v1 has no filters -- only a single compressor.
        if self.filters:
            base.report_loss(policy, "filters", 1)

        # v1 splits the numcodecs codec into a name and an options dict.
        compression = compression_opts = None
        if self.compressor:
            options = dict(self.compressor.to_dict())
            compression = options.pop("id")
            compression_opts = options or None

        return v1.ArrayMetadata(
            shape=self.shape,
            chunks=self.chunks,
            dtype=self.dtype.to_version(1),
            compression=compression,
            compression_opts=compression_opts,
            fill_value=self.fill_value,
            order=self.order,
            attributes=self.attributes,
        )

    def _to_v3(self, policy: ConversionPolicy = "lossy") -> base.ArrayMetadata:
        from abczarr.metadata import v3

        # A source stashed by an "annotate" down-conversion is the exact
        # original: restore it and drop the marker.
        source = self.attributes.get(base.SOURCE_ATTR)
        if source is not None:
            return v3.ArrayMetadata.from_dict(source)

        # v3 has no C/F memory-order field; only C (row-major) round-trips.
        if self.order != "C":
            base.report_loss(policy, "order", 3)

        separator = self.dimension_separator or "."
        chunk_grid = v3.RegularChunkGrid(configuration=self.chunks)
        chunk_key_encoding = v3.V2ChunkKeyEncoding(configuration=separator)

        # v3 codec pipeline: array->array filters, then the array->bytes
        # serializer that carries the endianness v2 keeps in the dtype, then
        # the bytes->bytes compressor.
        codecs = [c.to_version(3) for c in self.filters]
        codecs.append(_bytes_codec(v3, self.dtype.numpy))
        if self.compressor:
            codecs.append(self.compressor.to_version(3))

        return v3.ArrayMetadata(
            shape=self.shape,
            data_type=self.dtype.to_version(3),
            chunk_grid=chunk_grid,
            chunk_key_encoding=chunk_key_encoding,
            fill_value=self.fill_value,
            codecs=codecs,
            attributes=self.attributes,
        )


def _bytes_codec(v3: tx.Any, dtype: tx.Any) -> tx.Any:
    """The v3 array-to-bytes codec carrying *dtype*'s byte order."""
    endian = {
        "<": "little",
        ">": "big",
        "=": sys.byteorder,
    }.get(dtype.byteorder)  # "|" (byte-order-agnostic) -> None
    return v3.BytesCodec(configuration=endian)
