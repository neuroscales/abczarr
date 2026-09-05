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
        >>> meta = v2.ArrayMetadata.from_json({
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

    def to_json(self) -> tz.JsonDict:
        """Serialize to `.zarray`, writing `filters` as null when there are
        none.

        The model normalizes a missing `filters` to an empty tuple, but the
        Zarr v2 spec wants `null` for no filters, not an empty list.
        """
        data = super().to_json()
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
            options = dict(self.compressor.to_json())
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

        # A variable-length string/bytes array is numpy object (|O) in v2 with
        # a vlen filter (vlen-utf8 / vlen-bytes); in v3 it is the "string" /
        # "bytes" data type whose vlen codec is itself the array->bytes
        # serializer -- so no separate bytes serializer is added, matching
        # zarr-python. The vlen filter above already converted to that
        # serializer codec.
        vlen_dtype = _vlen_data_type(self.filters)
        if vlen_dtype is not None:
            data_type = v3.DType.from_json(vlen_dtype)
        else:
            data_type = self.dtype.to_version(3)
            codecs.append(_bytes_codec(v3, self.dtype.numpy))

        if self.compressor:
            codecs.append(self.compressor.to_version(3))

        return v3.ArrayMetadata(
            shape=self.shape,
            data_type=data_type,
            chunk_grid=chunk_grid,
            chunk_key_encoding=chunk_key_encoding,
            fill_value=self.fill_value,
            codecs=codecs,
            attributes=self.attributes,
        )


# The v2 vlen filter ids and the v3 data type each restores. A vlen filter
# on an |O array is what tags it as variable-length string ("vlen-utf8") or
# bytes ("vlen-bytes"), which numpy object alone cannot express.
_VLEN_FILTER_TO_V3_DTYPE = {"vlen-utf8": "string", "vlen-bytes": "bytes"}


def _vlen_data_type(filters: tx.Iterable[tx.Any]) -> tx.Optional[str]:
    """The v3 data type a vlen filter restores, or ``None``.

    Parameters
    ----------
    filters : iterable of Filter
        The v2 filters of an array.

    Returns
    -------
    str or None
        ``"string"`` for a ``vlen-utf8`` filter, ``"bytes"`` for a
        ``vlen-bytes`` filter, or ``None`` when no vlen filter is present.
    """
    for filt in filters:
        name = _VLEN_FILTER_TO_V3_DTYPE.get(getattr(filt, "id", None))
        if name is not None:
            return name
    return None


def _bytes_codec(v3: tx.Any, dtype: tx.Any) -> tx.Any:
    """The v3 array-to-bytes codec carrying *dtype*'s byte order."""
    endian = {
        "<": "little",
        ">": "big",
        "=": sys.byteorder,
    }.get(dtype.byteorder)  # "|" (byte-order-agnostic) -> None
    return v3.BytesCodec(configuration=endian)
