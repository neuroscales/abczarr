"""Build abczarr metadata from a backend node's metadata dict.

A driver reads a node's metadata as the plain JSON dict a Zarr store holds
(``zarr.json`` / ``.zarray``) and turns it into the typed, version-specific
abczarr metadata. Shared by the backend drivers so they agree on the mapping.
"""

__all__ = [
    "metadata_from_dict",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz


def metadata_from_dict(data: tz.JSONDict) -> tx.Any:
    """Build the abczarr metadata for a node from its metadata dict.

    The Zarr format version and node type in *data* choose the class.
    """
    from abczarr.metadata import base, v1, v2, v3

    zarr_format = data.get("zarr_format", 3)
    node_type = data.get("node_type") or (
        "array" if "shape" in data else "group"
    )
    if node_type == "array":
        array_cls = {
            1: v1.ArrayMetadata,
            2: v2.ArrayMetadata,
            3: v3.ArrayMetadata,
        }[zarr_format]
        return array_cls.from_dict(data)
    group_cls = {
        2: base.GroupMetadataV2,
        3: base.GroupMetadataV3,
    }.get(zarr_format, base.GroupMetadata)
    return group_cls.from_dict(data)
