__all__ = [
    "NodeMetadata",
    "GroupMetadata",
    "NodeMetadataV2",
    "ArrayMetadataV2",
    "GroupMetadataV2",
]

from abczarr._core.metadata import register_subclass

from ..base import (
    ArrayMetadataV2,
    GroupMetadataV2,
    NodeMetadataV2,
)


@register_subclass(zarr_format=2)
class NodeMetadata(NodeMetadataV2):
    """A Zarr v2 node's metadata, reached from `abczarr.metadata.v2`.

    Identical to
    [NodeMetadataV2][abczarr.metadata.base.NodeMetadataV2]. See
    [GroupMetadata][abczarr.metadata.v2.base.GroupMetadata] or
    [ArrayMetadata][abczarr.metadata.v2.array.ArrayMetadata] for the
    concrete field sets.
    """


@register_subclass(zarr_format=2, node_type="group")
class GroupMetadata(GroupMetadataV2):
    """A Zarr v2 group's metadata, reached from `abczarr.metadata.v2`.

    Identical to
    [GroupMetadataV2][abczarr.metadata.base.GroupMetadataV2]: user
    attributes and a format version, and nothing else.
    """
