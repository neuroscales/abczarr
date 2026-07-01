__all__ = [
    "Metadata",
    "NodeMetadataV1",
    "ArrayMetadataV1",
    "NodeMetadata",
]

from ..base import (
    Metadata,
    NodeMetadataV1,
    ArrayMetadataV1,
    register_subclass,
)


@register_subclass(zarr_format=1)
class NodeMetadata(NodeMetadataV1):
    ...
