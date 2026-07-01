__all__ = [
    "Metadata",
    "NodeMetadataV2",
    "ArrayMetadataV2",
    "GroupMetadataV2",
    "NodeMetadata",
    "GroupMetadata",
]

from ..base import (
    Metadata,
    NodeMetadataV2,
    ArrayMetadataV2,
    GroupMetadataV2,
    register_subclass,
)


@register_subclass(zarr_format=2)
class NodeMetadata(NodeMetadataV2):
    ...


@register_subclass(zarr_format=2, node_type="group")
class GroupMetadata(GroupMetadataV2):
    ...
