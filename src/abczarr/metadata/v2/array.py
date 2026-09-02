"""Zarr v2 array metadata.

Zarr v2 describes an array with a single numcodecs compressor, an
ordered list of numcodecs filters applied before it, and a
byte-order-bearing dtype. Converting to v1 (see
[ArrayMetadata.to_version][abczarr.metadata.v2.array.ArrayMetadata.to_version])
keeps only the compressor, since v1 has no filters; converting to v3
maps the filters and compressor onto v3's codec pipeline and folds
the dtype's byte order into an explicit serializer codec.
"""

__all__ = [
    "ArrayMetadata",
]

# stdlib
import sys

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import (
    autofrozen,
    eq_safenan,
    field,
    get_converter,
)
from abczarr._core.features import feature_key
from abczarr._core.metadata import register_subclass
from abczarr.metadata import base
from abczarr.metadata.base import ConversionPolicy

from .base import ArrayMetadataV2
from .codecs import Codec

# locals
from .dtypes import DType
from .filters import Filter

# In Zarr v2, ``filters`` is a list or ``null``, where null means no filters;
# accept null as an empty tuple so metadata straight from a store loads.
_to_filter_tuple = get_converter(tx.Tuple[Filter, ...])


def _filters_converter(value: tx.Any) -> tx.Tuple[Filter, ...]:
    return _to_filter_tuple(() if value is None else value)


# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


@register_subclass(zarr_format=2, node_type="array")
@autofrozen(kw_only=True)
class ArrayMetadata(ArrayMetadataV2):
    """A Zarr v2 array's metadata: shape, dtype, chunking and codecs.

    Corresponds to the contents of `.zarray`. `filters` run, in
    order, before `compressor` when encoding a chunk, and in reverse
    order after it when decoding.

    !!! example
        ```pycon
        >>> from abczarr.metadata import v2
        >>> meta = v2.ArrayMetadata.from_dict({
        ...     "zarr_format": 2,
        ...     "shape": [10, 10],
        ...     "chunks": [5, 5],
        ...     "dtype": "<f8",
        ...     "compressor": {"id": "zstd", "level": 3},
        ...     "fill_value": 0,
        ...     "order": "C",
        ...     "filters": [],
        ...     "attributes": {},
        ... })
        >>> [codec.name for codec in meta.to_version(3).codecs]
        ['bytes', 'zstd']

        ```
    """

    # --- Required ----
    shape: tz.Shape
    chunks: tz.Shape
    dtype: DType
    compressor: tx.Optional[Codec]
    fill_value: tx.Optional[tz.BuiltinNumber] = field(eq=eq_safenan)
    order: tz.MemoryOrder
    filters: tx.Tuple[Filter, ...] = field(converter=_filters_converter)

    # --- Optional ----
    dimension_separator: tx.Optional[tz.DimensionSeparator]

    # --- Serialization ---

    def to_dict(self) -> tz.JSONDict:
        """Serialize to `.zarray`, writing `filters` as null when there are
        none.

        The model normalizes a missing `filters` to an empty tuple, but the
        Zarr v2 spec wants `null` for no filters, not an empty list.
        """
        data = super().to_dict()
        if not data.get("filters"):
            data["filters"] = None
        return data

    # --- Conversion ---

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: ConversionPolicy = "lossy",
    ) -> base.ArrayMetadata:
        """Convert this array's metadata to another Zarr version.

        Converting to v1 keeps only the compressor: v1 has no
        filters, so any `filters` are subject to *policy*. Converting
        to v3 maps each filter and the compressor onto v3's codec
        pipeline and, when `order` is not ``"C"``, applies *policy*
        as well, since v3 has no memory-order field.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.
        policy : ConversionPolicy
            How to treat a field the target version can't hold.

        Returns
        -------
        ArrayMetadata
            Equivalent metadata for *version*. Converting to 2
            returns this object unchanged.

        Raises
        ------
        ValueError
            If *version* is not 1, 2 or 3.
        UnsupportedConversion
            If *policy* is ``"strict"`` and a field cannot be
            represented in *version*.
        """
        if version == 1:
            return self._to_v1(policy)
        if version == 2:
            return self
        if version == 3:
            return self._to_v3(policy)
        else:
            raise ValueError(f"Unsupported version: {version}")

    def required_features(self) -> tx.FrozenSet[str]:
        """The features a driver needs to read or write this array.

        One key per named filter and, if set, the compressor -- e.g.
        ``{"v2:filter:delta", "v2:codec:zstd"}``.
        """
        feats = set()  # type: tx.Set[str]
        if self.compressor is not None:
            name = getattr(self.compressor, "id", None)
            if name:
                feats.add(feature_key("v2", "codec", name))
        for filt in self.filters:
            name = getattr(filt, "id", None)
            if name:
                feats.add(feature_key("v2", "filter", name))
        return frozenset(feats)

    def _to_v1(self, policy: ConversionPolicy = "lossy") -> base.ArrayMetadata:
        from abczarr.metadata import v1

        # v1 has no filters -- only a single compressor.
        if self.filters:
            base._report_loss(policy, "filters", 1)

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

        # v3 has no C/F memory-order field; only C (row-major) round-trips.
        if self.order != "C":
            base._report_loss(policy, "order", 3)

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
