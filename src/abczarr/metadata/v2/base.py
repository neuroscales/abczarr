__all__ = [
    "NodeMetadata",
    "GroupMetadata",
    "NodeMetadataV2",
    "ArrayMetadataV2",
    "GroupMetadataV2",
]

from abczarr._core.metadata import register_subclass

from ..base import (
    NodeMetadataV2,
    ArrayMetadataV2,
    GroupMetadataV2,
)


@register_subclass(zarr_format=2)
class NodeMetadata(NodeMetadataV2):
    ...


@register_subclass(zarr_format=2, node_type="group")
class GroupMetadata(GroupMetadataV2):
    ...
