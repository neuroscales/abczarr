__all__ = [
    "NodeMetadata",
    "GroupMetadata",
    "NodeMetadataV3",
    "ArrayMetadataV3",
    "GroupMetadataV3",
]

from abczarr._core.metadata import register_subclass
from abczarr.metadata.base import (
    ArrayMetadataV3,
    GroupMetadataV3,
    NodeMetadataV3,
)


@register_subclass(zarr_format=3)
class NodeMetadata(NodeMetadataV3):
    """A Zarr v3 node's metadata, reached from `abczarr.metadata.v3`.

    Identical to
    [NodeMetadataV3][abczarr.metadata.base.NodeMetadataV3]. See
    [GroupMetadata][abczarr.metadata.v3.base.GroupMetadata] or
    [ArrayMetadata][abczarr.metadata.v3.array.ArrayMetadata] for the
    concrete field sets.
    """


@register_subclass(zarr_format=3, node_type="group")
class GroupMetadata(GroupMetadataV3):
    """A Zarr v3 group's metadata, reached from `abczarr.metadata.v3`.

    Identical to
    [GroupMetadataV3][abczarr.metadata.base.GroupMetadataV3]: user
    attributes and a format version, and nothing else.
    """
