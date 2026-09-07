__all__ = [
    "NodeMetadata",
    "NodeMetadataV1",
    "ArrayMetadataV1",
]

from abczarr._core.metadata import register_subclass
from abczarr.metadata.base import ArrayMetadataV1, NodeMetadataV1


@register_subclass(zarr_format=1)
class NodeMetadata(NodeMetadataV1):
    """A Zarr v1 node's metadata, reached from `abczarr.metadata.v1`.

    Identical to
    [NodeMetadataV1][abczarr.metadata.base.NodeMetadataV1]. Zarr v1
    has no groups, so every node is an array. See
    [ArrayMetadata][abczarr.metadata.v1.array.ArrayMetadata] for the
    concrete field set.
    """
