__all__ = [
    "Metadata",
    "NodeMetadataV3",
    "ArrayMetadataV3",
    "GroupMetadataV3",
    "NodeMetadata",
    "GroupMetadata",
]

from ..base import (
    Metadata,
    NodeMetadataV3 ,
    ArrayMetadataV3,
    GroupMetadataV3,
    register_subclass,
)


@register_subclass(zarr_format=3)
class NodeMetadata(NodeMetadataV3):
    ...


@register_subclass(zarr_format=3, node_type="group")
class GroupMetadata(GroupMetadataV3):
    ...
