__all__ = [
    "ChunkGrid",
    "RegularChunkGrid",
    "RectilinearChunkGrid",
    "ChunkKeyEncoding",
    "DefaultChunkKeyEncoding",
    "V2ChunkKeyEncoding",
    "ArrayMetadata",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autofrozen, update, field, eq_safenan

# metadata
from abczarr.metadata import base
from abczarr.metadata.base import register_subclass

# locals
from .base import ArrayMetadataV3
from .extensions import MustUnderstandExtension, ExtraField, TypedConfig
from .codecs.base import Codec
from .dtypes.base import DType


# ----------------------------------------------------------------------
#   CHUNK GRID
# ----------------------------------------------------------------------


@autofrozen
class ChunkGrid(MustUnderstandExtension):
    ...


@autofrozen
class RegularChunkGridConfig(TypedConfig):
    chunk_shape: tz.Shape


@register_subclass(name="regular")
@autofrozen
class RegularChunkGrid(ChunkGrid):
    name: tx.Literal["regular"]
    configuration: RegularChunkGridConfig


@autofrozen
class RectilinearChunkGridConfig(TypedConfig):
    kind: tx.Literal["inline"]
    chunk_shapes: tz.Shape


@register_subclass(name="rectilinear")
@autofrozen
class RectilinearChunkGrid(ChunkGrid):
    name: tx.Literal["rectilinear"]
    configuration: RectilinearChunkGridConfig


# ----------------------------------------------------------------------
#   CHUNK KEY ENCODING
# ----------------------------------------------------------------------


@autofrozen(extra_items=tz.JSON)
class ChunkKeyEncodingConfig(TypedConfig):
    ...


@autofrozen(extra_items=False)
class CommonChunkKeyEncodingConfig(ChunkKeyEncodingConfig):
    separator: tz.DimensionSeparator = "/"


@autofrozen
class ChunkKeyEncoding(MustUnderstandExtension):
    name: str
    configuration: ChunkKeyEncodingConfig

    def __new___(cls, name: str, *a, **k) -> tx.Self:
        if cls is ChunkKeyEncoding:
            if name == "default":
                return super().__new__(DefaultChunkKeyEncoding)
            elif name == "v2":
                return super().__new__(V2ChunkKeyEncoding)
        return super().__new__(cls)


@autofrozen(field_transformer=update(separator={"default": "/"}))
class DefaultChunkKeyEncodingConfig(CommonChunkKeyEncodingConfig):
    ...


@register_subclass(name="default")
@autofrozen
class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    name: tx.Literal["default"]
    configuration: DefaultChunkKeyEncodingConfig


@autofrozen(field_transformer=update(separator={"default": "."}))
class V2ChunkKeyEncodingConfig(CommonChunkKeyEncodingConfig):
    ...


@register_subclass(name="v2")
@autofrozen
class V2ChunkKeyEncoding(ChunkKeyEncoding):
    name: tx.Literal["v2"]
    configuration: V2ChunkKeyEncodingConfig


# ----------------------------------------------------------------------
#   ARRAY
# ----------------------------------------------------------------------


_AxisNames = tx.Tuple[tx.Optional[str], ...]


@register_subclass(zarr_format=3, node_type="array")
@autofrozen(kw_only=True, extra_items=ExtraField)
class ArrayMetadata(ArrayMetadataV3):

    # --- Required ----
    shape: tz.Shape
    data_type: DType
    chunk_grid: ChunkGrid
    chunk_key_encoding: ChunkKeyEncoding
    fill_value: tx.Optional[tz.BuiltinNumber] = field(eq=eq_safenan)
    codecs: tx.Tuple[Codec, ...]

    # --- Optional ----
    attributes: tz.JSONDict
    dimension_names: tx.Optional[_AxisNames]
    storage_transformers: tx.Tuple[tz.JSONDict, ...]

    # --- Conversion ---

    def to_version(self, version: tz.ZarrVersion) -> base.ArrayMetadata:
        if version == 1:
            return self._to_v1()
        if version == 2:
            return self._to_v2()
        if version == 3:
            return self
        else:
            raise ValueError(f"Unsupported version: {version}")

    def _to_v1(self) -> base.ArrayMetadata:
        from abczarr.metadata import v1

        if self.chunk_grid.name != "regular":
            raise ValueError("Only regular chunk grids are supported in Zarr v1")
        chunk_grid = tx.cast(RegularChunkGrid, self.chunk_grid)
        chunk_shape = chunk_grid.configuration.chunk_shape

        compressor = None
        filters = [c.to_version(1) for c in self.codecs]
        if filters and isinstance(filters[-1], v1.Codec):
            compressor = filters[-1]
            filters = filters[:-1]
        if filters:
            raise ValueError("Zarr v1 does not support filters")

        return v1.ArrayMetadata(
            shape=self.shape,
            chunks=chunk_shape,
            dtype=self.data_type.to_version(1),
            compression=compressor,
            compression_opts=None,
            fill_value=self.fill_value,
        )

    def _to_v2(self) -> base.ArrayMetadata:
        from abczarr.metadata import v2

        if self.chunk_grid.name != "regular":
            raise ValueError("Only regular chunk grids are supported in Zarr v2")
        chunk_grid = tx.cast(RegularChunkGrid, self.chunk_grid)
        chunk_shape = chunk_grid.configuration.chunk_shape

        separator = getattr(
            self.chunk_key_encoding.configuration, "separator", "."
        )

        compressor = None
        filters = [c.to_version(2) for c in self.codecs]
        if filters:
            if isinstance(filters[-1], v2.Codec):
                compressor = filters[-1]
                filters = filters[:-1]

        return v2.ArrayMetadata(
            shape=self.shape,
            chunks=chunk_shape,
            dtype=self.data_type.to_version(2),
            compressor=compressor,
            fill_value=self.fill_value,
            filters=filters,
            dimension_separator=separator
        )
