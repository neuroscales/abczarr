__all__ = [
    "NodeMetadata",
    "NodeMetadataV1",
    "ArrayMetadataV1",
]

from abczarr._core.metadata import register_subclass
from abczarr.metadata.base import ArrayMetadataV1, NodeMetadataV1


@register_subclass(zarr_format=1)
class NodeMetadata(NodeMetadataV1):
    ...
