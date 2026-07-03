"""
Metadata handling for Zarr.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""
__all__ = [
    "Metadata",
    "FlexibleMetadata",
    "NodeMetadata",
    "GroupMetadata",
    "ArrayMetadata",
    "NodeMetadataV1",
    "ArrayMetadataV1",
    "NodeMetadataV2",
    "GroupMetadataV2",
    "ArrayMetadataV2",
    "NodeMetadataV3",
    "GroupMetadataV3",
    "ArrayMetadataV3",
]

# stdlib
import json
import os
import tempfile

# dependencies
import typing_extensions as tx

# locals
from abczarr._core import typing as tz
from abczarr._core import constants
from abczarr._core.attrs import autofrozen, evolve
from abczarr._core.metadata import (
    Metadata, FlexibleMetadata, register_subclass
)


# ======================================================================
#
#                                BASE
#
# ======================================================================


@autofrozen
class NodeMetadata(Metadata):

    attributes: tz.JSONDict
    zarr_format: tz.ZarrVersion = 3
    node_type: tz.NodeType = "group"

    # Convenience updaters (immutably return new metadata)
    def update_attributes(self, attributes: tz.JSONDict) -> tx.Self:
        """Return a new Metadata with updated attributes."""
        return evolve(self, attributes=dict(attributes))

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""
        zarr_json = root / constants.Z3_JSON
        if zarr_json.exists():
            return NodeMetadataV3.from_file(root)
        zgroup = root / constants.Z2GROUP_JSON
        zarrays = root / constants.Z2ARRAY_JSON
        if zgroup.exists() or zarrays.exists():
            return NodeMetadataV2.from_files(root)
        zmeta = root / constants.Z1META_JSON
        if zmeta.exists():
            return NodeMetadataV1.from_files(root)
        raise FileNotFoundError(
            f"No metadata found in {root}.Expected one of: "
            f"{constants.Z3_JSON}, "
            f"{constants.Z2GROUP_JSON}, "
            f"{constants.Z2ARRAY_JSON}, "
            f"{constants.Z1META_JSON}"
        )


@register_subclass(node_type="group")
@autofrozen
class GroupMetadata(NodeMetadata):

    node_type: tx.Literal["group"] = "group"


@register_subclass(node_type="array")
@autofrozen
class ArrayMetadata(NodeMetadata):

    node_type: tx.Literal["array"] = "array"


# ======================================================================
#
#                                   V1
#
# ======================================================================


@register_subclass(zarr_format=1)
@autofrozen
class NodeMetadataV1(NodeMetadata):

    zarr_format: tx.Literal[1] = 1

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""
        attrs = {}
        zattrs = root / constants.Z1ATTRS_JSON
        if zattrs.exists():
            with zattrs.open("r", encoding="utf-8") as f:
                attrs = json.load(f)

        meta = {}
        zmeta = root / constants.Z1META_JSON
        if zmeta.exists():
            with zmeta.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        meta.setdefault("zarr_format", 1)

        if cls is NodeMetadataV1:
            # There are no groups in Zarr v1
            cls = getattr(ArrayMetadataV1, "_IMPL", ArrayMetadataV1)

        return cls.from_dict({**meta, "attributes": attrs})

    @classmethod
    def from_dict(cls, data: tz.JSONDict) -> tx.Self:
        if cls is NodeMetadataV1:
            # There are no groups in Zarr v1
            cls = getattr(ArrayMetadataV1, "_IMPL", ArrayMetadataV1)
        return super().from_dict(cls, data)


@register_subclass(zarr_format=1, node_type="array")
@autofrozen
class ArrayMetadataV1(NodeMetadataV1, ArrayMetadata):
    ...


# ======================================================================
#
#                                   V2
#
# ======================================================================


@register_subclass(zarr_format=2)
@autofrozen
class NodeMetadataV2(NodeMetadata):

    zarr_format: tx.Literal[2] = 2

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""

        # --- Detect node type ---

        if cls is NodeMetadataV2:

            if (root / constants.Z2ARRAY_JSON).exists():
                return ArrayMetadataV2.from_files(root)

            if (root / constants.Z2GROUP_JSON).exists():
                return GroupMetadataV2.from_files(root)

            raise FileNotFoundError(
                f"No Zarr v2 metadata found in {root}. Expected one of: "
                f"{constants.Z2ARRAY_JSON}, {constants.Z2GROUP_JSON}"
            )

        # --- We know our node type ---

        if issubclass(cls, ArrayMetadataV2):
            META_JSON = constants.Z2ARRAY_JSON
        elif issubclass(cls, GroupMetadataV2):
            META_JSON = constants.Z2GROUP_JSON
        else:
            raise ValueError(
                f"Cannot determine metadata type for {cls.__name__}"
            )

        meta = {}
        zgroup = root / META_JSON
        if zgroup.exists():
            with zgroup.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.setdefault("zarr_format", 2)

        attrs = {}
        zattrs = root / constants.Z2ATTRS_JSON
        if zattrs.exists():
            with zattrs.open("r", encoding="utf-8") as f:
                attrs = json.load(f)

        return cls.from_dict({**meta, "attributes": attrs})

    @classmethod
    def to_file(self, root: os.PathLike) -> None:
        """Write this metadata to the specified root directory."""
        new_meta = self.to_dict()
        new_attrs = new_meta.pop("attributes", {})

        META_JSON = {
            "array": constants.Z2ARRAY_JSON,
            "group": constants.Z2GROUP_JSON,
        }[self.node_type]

        meta = {}
        mpath = root / META_JSON
        if mpath.exists():
            with mpath.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.update(new_meta)

        apath = root / constants.Z2ATTRS_JSON
        if apath.exists():
            with apath.open("r", encoding="utf-8") as f:
                attrs = json.load(f)
        else:
            attrs = {}
        attrs.update(new_attrs)

        _atomic_write(mpath, meta)
        _atomic_write(apath, attrs)


@register_subclass(zarr_format=2, node_type="group")
@autofrozen
class GroupMetadataV2(NodeMetadataV2, GroupMetadata):
    ...


@register_subclass(zarr_format=2, node_type="array")
@autofrozen
class ArrayMetadataV2(NodeMetadataV2, ArrayMetadata):
    ...


# ======================================================================
#
#                                   V3
#
# ======================================================================


@register_subclass(zarr_format=3)
@autofrozen
class NodeMetadataV3(NodeMetadata):
    """Metadata for Zarr v3, including attributes and format version."""

    zarr_format: tx.Literal[3] = 3

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""
        zarr_json = root / constants.Z3_JSON
        if zarr_json.exists():
            with zarr_json.open("r", encoding="utf-8") as f:
                d = json.load(f)
        return cls.from_dict(d)

    def to_file(self, root: os.PathLike) -> None:
        """Write this metadata to the specified root directory."""
        path = root / constants.Z3_JSON
        data = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        data.update(self.to_dict())
        _atomic_write(path, data)


@register_subclass(zarr_format=3, node_type="group")
@autofrozen
class GroupMetadataV3(NodeMetadataV3, GroupMetadata):
    ...


@register_subclass(zarr_format=3, node_type="array")
@autofrozen
class ArrayMetadataV3(NodeMetadataV3, ArrayMetadata):
    ...


# ======================================================================
#
#                                 UTILS
#
# ======================================================================


def _atomic_write(path: os.PathLike, data: tz.JSONDict) -> None:
    """Write JSON data to path atomically."""
    PathType = type(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".meta_tmp_", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        PathType(tmp).replace(path)
    finally:
        try:
            if PathType(tmp).exists():
                PathType(tmp).unlink()
        except Exception:
            pass
