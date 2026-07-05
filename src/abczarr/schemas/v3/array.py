__all__ = [
    "ChunkGrid",
    "RegularChunkGrid",
    "RectilinearChunkGrid",
    "ChunkKeyEncoding",
    "DefaultChunkKeyEncoding",
    "V2ChunkKeyEncoding",
    "Array",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from .codecs import ValidCodec
from .extensions import Config, Extension, ExtensionWithConfig, ExtraField

# typing
Optional = RequirementForTypedDict.Optional


class ChunkGridConfig(Config):
    ...


class ChunkGrid(Extension):
    name: tx.Literal["regular", "rectilinear"]
    configuration: Optional[ChunkGridConfig]


class RegularChunkGridConfig(Config):
    chunk_shape: tz.BuiltinSequence[int]


class RegularChunkGrid(ExtensionWithConfig):
    name: tx.Literal["regular"]
    configuration: RegularChunkGridConfig


class RectilinearChunkGridConfig(Config):
    kind: tx.Literal["inline"]
    chunk_shapes: tz.BuiltinSequence[int]


class RectilinearChunkGrid(ExtensionWithConfig):
    name: tx.Literal["rectilinear"]
    configuration: RectilinearChunkGridConfig


ValidChunkGrid = tx.Union[RegularChunkGrid, RectilinearChunkGrid]


class ChunkKeyEncodingConfig(Config):
    ...


class CommonChunkKeyEncodingConfig(ChunkKeyEncodingConfig):
    separator: tx.NotRequired[tz.DimensionSeparator]


class ChunkKeyEncoding(Extension):
    name: tx.Literal["default", "v2"]
    configuration: Optional[ChunkKeyEncodingConfig]


class DefaultChunkKeyEncoding(ChunkKeyEncoding):
    name: tx.Literal["default"]
    configuration: Optional[CommonChunkKeyEncodingConfig]


class V2ChunkKeyEncoding(ChunkKeyEncoding):
    name: tx.Literal["v2"]
    configuration: Optional[CommonChunkKeyEncodingConfig]


ValidChunkKeyEncoding = tx.Union[DefaultChunkKeyEncoding, V2ChunkKeyEncoding]


class StorageTransformer(Extension):
    # No storage transformer specified so far, it seems.
    ...


class Array(tx.TypedDict, extra_items=ExtraField):

    # --- Required ----
    zarr_format: tx.Literal[3]
    node_type: tx.Literal["array"]
    shape:  tz.BuiltinSequence[int]
    data_type: tz.DataTypeV3
    chunk_grid: ValidChunkGrid
    chunk_key_encoding: ValidChunkKeyEncoding
    fill_value: tx.Optional[tz.BuiltinNumber]
    codecs: tz.BuiltinSequence[ValidCodec]

    # --- Optional ----
    attributes: Optional[tz.JSONDict]
    storage_transformers: Optional[tz.BuiltinSequence[StorageTransformer]]
    dimension_names: Optional[tz.BuiltinSequence[tx.Optional[str]]]
