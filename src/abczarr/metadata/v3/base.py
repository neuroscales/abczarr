__all__ = [
    "NodeMetadata",
    "GroupMetadata",
    "NodeMetadataV3",
    "ArrayMetadataV3",
    "GroupMetadataV3",
]

from abczarr._core.metadata import register_subclass
from abczarr.metadata.base import (
    NodeMetadataV3 ,
    ArrayMetadataV3,
    GroupMetadataV3,
)


@register_subclass(zarr_format=3)
class NodeMetadata(NodeMetadataV3):
    ...


@register_subclass(zarr_format=3, node_type="group")
class GroupMetadata(GroupMetadataV3):
    ...
