"""Zarr v1 array metadata.

Zarr v1 has no groups, so an array is the only node this format
defines. Its metadata names a single numcodecs compressor by name and
carries its options separately, the model v2 also uses -- converting
to v2 or v3 (see
[ArrayMetadata.to_version][abczarr.metadata.v1.array.ArrayMetadata.to_version])
routes through v2's richer representation of the same compressor.
"""

__all__ = [
    "ArrayMetadata",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen, eq_safenan, field
from abczarr._core.features import feature_key
from abczarr._errors import UnsupportedConversion
from abczarr.metadata import base
from abczarr.metadata.base import ConversionPolicy, register_subclass

from .base import ArrayMetadataV1
from .codecs import CodecOptions
from .codecs.aliases import Codec

# locals
from .dtypes import DType

# The single numcodecs option a scalar ``compression_opts`` fills, by codec.
# A scalar is codec-specific in Zarr v1: an integer compression level, or --
# for blosc -- a string compressor name. These tables are kept here rather
# than derived from numcodecs, so this metadata layer needs no numcodecs
# dependency; a codec absent from the matching table has no scalar form.
_SCALAR_LEVEL_KEY = {
    "zlib": "level",
    "gzip": "level",
    "bz2": "level",
    "zstd": "level",
    "lzma": "preset",
    "lz4": "acceleration",
}
_SCALAR_NAME_KEY = {
    "blosc": "cname",
}

# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


@register_subclass(zarr_format=1, node_type="array")
@autofrozen(kw_only=True, extra_items=tz.FrozenJson)
class ArrayMetadata(ArrayMetadataV1):
    """A Zarr v1 array's metadata: shape, dtype, chunking and codec.

    Corresponds to the contents of `.zarray`. The compressor is
    named by `compression` (a numcodecs id) with its options in
    `compression_opts`; a `None` compression means the array is
    stored uncompressed.

    !!! example
        ```pycon
        >>> from abczarr.metadata import v1
        >>> meta = v1.ArrayMetadata.from_json({
        ...     "zarr_format": 1,
        ...     "shape": [10],
        ...     "chunks": [5],
        ...     "dtype": "<f8",
        ...     "compression": "zlib",
        ...     "compression_opts": {"level": 1},
        ...     "fill_value": 0,
        ...     "order": "C",
        ...     "attributes": {},
        ... })
        >>> meta.to_version(2).compressor
        ZlibCodec(id='zlib', level=1)

        ```
    """

    # --- Required ----
    shape: tz.Shape
    chunks: tz.Shape
    dtype: DType
    compression: tx.Optional[Codec]
    # numcodecs carries a codec's options as an object, but the authored v1
    # ``array.schema`` also allows the scalar forms (an integer level, or a
    # string) that some codecs use, so accept those alongside the object.
    compression_opts: tx.Optional[tx.Union[CodecOptions, int, str]]
    fill_value: tx.Optional[tz.BuiltinNumber] = field(eq=eq_safenan)
    order: tz.MemoryOrder

    # --- Conversion ---

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: ConversionPolicy = "lossy",
    ) -> base.ArrayMetadata:
        """Convert this array's metadata to another Zarr version.

        A v1 array converts to v2 or v3 without loss: v2's compressor
        model can represent everything v1's compression/
        compression_opts pair can, and v3's codec pipeline in turn
        can represent v2's. *policy* is accepted for a consistent
        signature across versions but is never invoked, since nothing
        is dropped.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.
        policy : ConversionPolicy
            How to treat a field the target can't hold. Unused here.

        Returns
        -------
        ArrayMetadata
            Equivalent metadata for *version*. Converting to 1
            returns this object unchanged.

        Raises
        ------
        ValueError
            If *version* is not 1, 2 or 3.
        """
        if version == 1:
            return self
        if version == 2:
            return self._to_v2(policy)
        if version == 3:
            # route through v2 -- v1 and v2 share the numcodecs model
            return self._to_v2(policy).to_version(3, policy)
        raise ValueError(f"Unsupported version: {version}")

    def required_features(self) -> tx.FrozenSet[str]:
        """The features a driver needs to read or write this array.

        A single-element set naming the compressor, e.g.
        ``{"v1:codec:zlib"}``, or an empty set when the array is
        stored uncompressed.
        """
        if not self.compression:
            return frozenset()
        # v1 carries the compressor as its numcodecs name (a string)
        name = getattr(self.compression, "id", None) or self.compression
        return frozenset({feature_key("v1", "codec", str(name))})

    def _to_v2(self, policy: ConversionPolicy = "lossy") -> base.ArrayMetadata:
        from abczarr.metadata import v2

        # v1 splits the codec into a name + options; v2 keeps them together.
        compressor = None
        if self.compression:
            compressor = {"id": self.compression, **self._compressor_options()}

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

    def _compressor_options(self) -> tx.Dict[str, tx.Any]:
        """The compressor's numcodecs options, to merge onto its ``id``.

        v1 keeps a codec's options in ``compression_opts``. That is usually
        an object, used as is. The spec (and the authored ``array.schema``)
        also allow a scalar -- an integer level, or a string -- for codecs
        that take one; a scalar is not a numcodecs config on its own, so it
        is placed under the option that codec fills it into (``zlib`` ``1``
        -> ``{"level": 1}``; ``lz4`` -> ``acceleration``; ``blosc`` ``"lz4"``
        -> ``cname``). A codec with no scalar form raises
        [UnsupportedConversion][abczarr._errors.UnsupportedConversion]. No
        compression means no options.
        """
        opts = self.compression_opts
        if opts is None:
            return {}
        name = str(self.compression)
        if isinstance(opts, str):
            key = _SCALAR_NAME_KEY.get(name)
        elif isinstance(opts, int):
            key = _SCALAR_LEVEL_KEY.get(name)
        else:
            # an object (a CodecOptions, or a plain mapping): dict-able as is.
            return dict(opts)
        if key is None:
            raise UnsupportedConversion(
                f"scalar compression_opts for {self.compression!r}", 2
            )
        return {key: opts}
