"""
The version-independent metadata model.

Every node in a Zarr hierarchy -- a group or an array -- is described
by a small JSON document: `zarr.json` in Zarr v3, `.zarray`/`.zgroup`
plus `.zattrs` in v2, `.zarray`/`.zattrs` in v1. This module defines
the typed classes that document holds, one hierarchy per format
version, and the shared vocabulary
([ArrayMetadata][abczarr.metadata.base.ArrayMetadata],
[GroupMetadata][abczarr.metadata.base.GroupMetadata]) that lets code
work with a node's metadata without caring which version produced it.

`ArrayMetadata.to_version` converts a node's metadata to another
format version. Not every
version can represent everything another one can; how such a
conversion treats a field it cannot carry over is set by a
`ConversionPolicy`.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""
__all__ = [
    "ConversionPolicy",
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
import warnings

# dependencies
import typing_extensions as tx

from abczarr._core import constants

# locals
from abczarr._core import typing as tz
from abczarr._core.auto import autofrozen, evolve
from abczarr._core.metadata import (
    FlexibleMetadata,
    Metadata,
    register_subclass,
)
from abczarr.errors import UnsupportedConversion

# ======================================================================
#
#                          CONVERSION POLICY
#
# ======================================================================

#: How a conversion treats a field the target version cannot hold.
#:
#: * ``"lossy"`` (the default) -- drop the field silently.
#: * ``"warn"`` -- drop the field, but emit one warning naming it.
#: * ``"strict"`` -- raise
#:   [UnsupportedConversion][abczarr.errors.UnsupportedConversion]
#:   instead of dropping anything.
ConversionPolicy = tx.Literal["lossy", "warn", "strict"]


def _report_loss(
    policy: ConversionPolicy, field: str, version: tz.ZarrVersion
) -> None:
    """Apply a conversion policy to a field the target can't hold.

    Called by a version's `to_version` implementation for each field
    it cannot carry over to *version*. Does nothing under
    ``"lossy"``, emits a warning under ``"warn"``, and raises
    [UnsupportedConversion][abczarr.errors.UnsupportedConversion]
    under ``"strict"``.

    Parameters
    ----------
    policy : ConversionPolicy
        How to treat the loss.
    field : str
        The name of the field that cannot be represented.
    version : ZarrVersion
        The Zarr format version being converted to.

    Raises
    ------
    UnsupportedConversion
        If *policy* is ``"strict"``.
    """
    if policy == "lossy":
        return
    if policy == "warn":
        warnings.warn(
            f"dropping {field!r}: not representable in Zarr v{version}",
            stacklevel=3,
        )
        return
    if policy == "strict":
        raise UnsupportedConversion(field, version)
    raise ValueError(f"unknown conversion policy: {policy!r}")

# ======================================================================
#
#                                BASE
#
# ======================================================================


@autofrozen
class NodeMetadata(Metadata):
    """The metadata common to every node in a Zarr hierarchy.

    A node is either a group or an array; both carry user attributes
    and a format version. Use
    [GroupMetadata][abczarr.metadata.base.GroupMetadata] or
    [ArrayMetadata][abczarr.metadata.base.ArrayMetadata] -- or one of
    their per-version subclasses -- rather than this class directly.
    """

    attributes: tz.JsonDict
    zarr_format: tz.ZarrVersion = 3
    node_type: tz.NodeType = "group"

    # Convenience updaters (immutably return new metadata)
    def update_attributes(self, attributes: tz.JsonDict) -> tx.Self:
        """Return a copy of this metadata with new attributes.

        The rest of the metadata -- shape, dtype, chunking and so on
        -- is unchanged.
        """
        return evolve(self, attributes=dict(attributes))

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a node's metadata from its directory.

        Detects the Zarr format version by which metadata file is
        present under *root* -- `zarr.json` (v3), `.zarray` or
        `.zgroup` (v2), or `.zarray` (v1) -- and returns metadata of
        the matching version.

        Raises
        ------
        FileNotFoundError
            If *root* holds no recognized metadata file.
        """
        zarr_json = root / constants.Z3_JSON
        if zarr_json.exists():
            return NodeMetadataV3.from_file(root)
        zgroup = root / constants.Z2GROUP_JSON
        zarrays = root / constants.Z2ARRAY_JSON
        if zgroup.exists() or zarrays.exists():
            return NodeMetadataV2.from_file(root)
        zmeta = root / constants.Z1META_JSON
        if zmeta.exists():
            return NodeMetadataV1.from_file(root)
        raise FileNotFoundError(
            f"No metadata found in {root}.Expected one of: "
            f"{constants.Z3_JSON}, "
            f"{constants.Z2GROUP_JSON}, "
            f"{constants.Z2ARRAY_JSON}, "
            f"{constants.Z1META_JSON}"
        )


def _node_type_at(root: os.PathLike) -> tx.Optional[tz.NodeType]:
    """Report whether *root* holds a Zarr array, a group, or neither.

    Reads only enough to answer that -- a v3 `zarr.json`'s `node_type`
    field, or which of `.zarray` and `.zgroup` a v2 node has -- never the
    rest of the metadata. That keeps it cheap enough to call on every
    child while listing a group.

    Parameters
    ----------
    root : PathLike
        The directory to inspect.

    Returns
    -------
    str or None
        ``"array"`` or ``"group"`` if *root* holds Zarr metadata of that
        kind, otherwise `None`.
    """
    detected = _node_at(root)
    return detected[0] if detected else None


def _node_at(
    root: os.PathLike,
) -> tx.Optional[tx.Tuple[tz.NodeType, tz.ZarrVersion]]:
    """The kind and Zarr version of the node stored at *root*.

    Returns a ``(node_type, version)`` pair -- ``"array"`` or ``"group"``
    with 1, 2 or 3 -- for a directory that holds Zarr metadata, or `None`
    for one that does not. Reads only enough to answer that: a v3
    `zarr.json`'s `node_type`, or which of `.zarray` and `.zgroup` a v2 node
    has. A v3 `zarr.json` without a valid `node_type` is treated as no node,
    since the format requires one, rather than guessed at from its other
    fields.

    Parameters
    ----------
    root : PathLike
        The directory to inspect.
    """
    zarr_json = root / constants.Z3_JSON
    if zarr_json.exists():
        try:
            with zarr_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        node_type = data.get("node_type")
        if node_type in ("array", "group"):
            return node_type, 3
        return None
    if (root / constants.Z2ARRAY_JSON).exists():
        return "array", 2
    if (root / constants.Z2GROUP_JSON).exists():
        return "group", 2
    if (root / constants.Z1META_JSON).exists():
        return "array", 1
    return None


@register_subclass(node_type="group")
@autofrozen
class GroupMetadata(NodeMetadata):
    """A group's metadata: user attributes and a format version.

    A group holds no data of its own, so beyond
    [NodeMetadata][abczarr.metadata.base.NodeMetadata] it adds
    nothing but its `node_type`. Members and their metadata are
    reached through the store, not through this object.
    """

    node_type: tx.Literal["group"] = "group"

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: ConversionPolicy = "lossy",
    ) -> "GroupMetadata":
        """Convert this group's metadata to another Zarr version.

        A group carries only user attributes and a format version, so a
        conversion between v2 and v3 re-stamps the format and carries the
        attributes across unchanged -- nothing is lost, and *policy* is
        never invoked. Zarr v1 has no group concept, so converting a group
        to v1 has no representation and raises regardless of *policy*.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.
        policy : ConversionPolicy
            How to treat a field the target version can't hold. Accepted
            for a signature that matches
            [ArrayMetadata.to_version][abczarr.metadata.base.ArrayMetadata],
            but a group has no such field between v2 and v3.

        Returns
        -------
        GroupMetadata
            Equivalent metadata for *version*. Converting to the group's
            own version returns this object unchanged.

        Raises
        ------
        ValueError
            If *version* is not 1, 2 or 3.
        UnsupportedConversion
            If *version* is 1, which has no group concept.
        """
        if version == self.zarr_format:
            return self
        if version == 1:
            # Zarr v1 predates groups entirely -- a group has no v1 form to
            # carry attributes into, so the conversion cannot proceed under
            # any policy. This is a documented limitation, not a
            # policy-governed loss, so it raises a named error regardless of
            # *policy*.
            raise UnsupportedConversion("group", 1)
        if version in (2, 3):
            target = {2: GroupMetadataV2, 3: GroupMetadataV3}[version]
            return target(attributes=self.attributes)
        raise ValueError(f"Unsupported version: {version}")


@register_subclass(node_type="array")
@autofrozen
class ArrayMetadata(NodeMetadata):
    """An array's metadata: shape, data type, chunking and codecs.

    The exact fields depend on the Zarr format version -- see
    [ArrayMetadataV1][abczarr.metadata.base.ArrayMetadataV1],
    [ArrayMetadataV2][abczarr.metadata.base.ArrayMetadataV2] and
    [ArrayMetadataV3][abczarr.metadata.base.ArrayMetadataV3]. What
    they share is `to_version`, which converts between versions, and
    [required_features][abczarr.metadata.base.ArrayMetadata.required_features],
    which reports what a driver needs to support to read or write
    the array.
    """

    node_type: tx.Literal["array"] = "array"

    def required_features(self) -> tx.FrozenSet[str]:
        """The features a driver needs to read or write this array.

        Each feature is a namespaced key built from the array's
        codecs, chunk grid, chunk-key encoding and data type -- for
        example ``"v3:codec:zstd"`` or ``"v2:filter:delta"``. A
        driver compares this set against what it supports to decide
        whether it can open the array, so an unsupported codec is
        named up front rather than failing partway through a read.

        Every concrete array metadata class overrides this with its
        own version-specific keys; the base implementation returns an
        empty set.
        """
        return frozenset()


# ======================================================================
#
#                                   V1
#
# ======================================================================


@register_subclass(zarr_format=1)
@autofrozen
class NodeMetadataV1(NodeMetadata):
    """A Zarr v1 node's metadata.

    Zarr v1 has no groups, so every node is an array; use
    [ArrayMetadataV1][abczarr.metadata.base.ArrayMetadataV1], or
    build one through this class -- see
    [from_file][abczarr.metadata.base.NodeMetadataV1.from_file] and
    [from_json][abczarr._core.metadata.Metadata.from_json].
    """

    zarr_format: tx.Literal[1] = 1

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a v1 array's metadata from its directory.

        Reads `.zarray` for the array's metadata and `.zattrs` for
        its user attributes, if present.
        """
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

        return cls.from_json({**meta, "attributes": attrs})

    @classmethod
    def from_json(cls, data: tz.JsonDict) -> tx.Self:
        """Build v1 metadata from a plain JSON-compatible dict.

        *data* is the merged content of `.zarray` and `.zattrs`
        (under the key `"attributes"`), the same shape
        [to_json][abczarr._core.metadata.Metadata.to_json] produces.
        Called on this class directly, it returns
        [ArrayMetadataV1][abczarr.metadata.base.ArrayMetadataV1]
        metadata, since Zarr v1 has no groups.
        """
        if cls is NodeMetadataV1:
            # There are no groups in Zarr v1
            cls = getattr(ArrayMetadataV1, "_IMPL", ArrayMetadataV1)
        # Dispatch through the base implementation bound to the resolved
        # class (calling super().from_json(cls, data) would pass cls as the
        # data argument).
        return Metadata.from_json.__func__(cls, data)


@register_subclass(zarr_format=1, node_type="array")
@autofrozen
class ArrayMetadataV1(NodeMetadataV1, ArrayMetadata):
    """The array-specific fields shared by every Zarr v1 array.

    See [ArrayMetadata][abczarr.metadata.v1.array.ArrayMetadata] for
    the concrete class with shape, chunking, data type and codec
    fields, and its `to_version` for conversion to v2 and v3.
    """


# ======================================================================
#
#                                   V2
#
# ======================================================================


@register_subclass(zarr_format=2)
@autofrozen
class NodeMetadataV2(NodeMetadata):
    """A Zarr v2 node's metadata.

    Use [GroupMetadataV2][abczarr.metadata.base.GroupMetadataV2] or
    [ArrayMetadataV2][abczarr.metadata.base.ArrayMetadataV2] for the
    concrete field sets; called on this class,
    [from_file][abczarr.metadata.base.NodeMetadataV2.from_file]
    works out which one applies.
    """

    zarr_format: tx.Literal[2] = 2

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a v2 node's metadata from its directory.

        Reads `.zarray` or `.zgroup` for the node's metadata and
        `.zattrs` for its user attributes. Called on this class, the
        node type is detected from which of `.zarray` and `.zgroup`
        is present; called on a group or array subclass, that file is
        read directly.

        Raises
        ------
        FileNotFoundError
            If *root* holds neither `.zarray` nor `.zgroup`.
        """

        # --- Detect node type ---

        if cls is NodeMetadataV2:

            if (root / constants.Z2ARRAY_JSON).exists():
                return ArrayMetadataV2.from_file(root)

            if (root / constants.Z2GROUP_JSON).exists():
                return GroupMetadataV2.from_file(root)

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

        return cls.from_json({**meta, "attributes": attrs})

    def to_file(self, root: os.PathLike) -> None:
        """Write this metadata to its directory.

        Writes the array/group fields to `.zarray`/`.zgroup` and the
        user attributes to `.zattrs`, merging into whatever is
        already there rather than overwriting the whole file.
        """
        new_meta = self.to_json()
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
    """A Zarr v2 group's metadata: user attributes only."""


@register_subclass(zarr_format=2, node_type="array")
@autofrozen
class ArrayMetadataV2(NodeMetadataV2, ArrayMetadata):
    """The array-specific fields shared by every Zarr v2 array.

    See [ArrayMetadata][abczarr.metadata.v2.array.ArrayMetadata] for
    the concrete class with shape, chunking, data type, compressor
    and filter fields, and its `to_version` for conversion to v1 and
    v3.
    """


# ======================================================================
#
#                                   V3
#
# ======================================================================


@register_subclass(zarr_format=3)
@autofrozen
class NodeMetadataV3(NodeMetadata):
    """A Zarr v3 node's metadata.

    Use [GroupMetadataV3][abczarr.metadata.base.GroupMetadataV3] or
    [ArrayMetadataV3][abczarr.metadata.base.ArrayMetadataV3] for the
    concrete field sets; a v3 node's type is recorded in its
    `zarr.json`, so `node_type` need not be known in advance to load
    it.
    """

    zarr_format: tx.Literal[3] = 3

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a v3 node's metadata from its `zarr.json`."""
        zarr_json = root / constants.Z3_JSON
        if zarr_json.exists():
            with zarr_json.open("r", encoding="utf-8") as f:
                d = json.load(f)
        return cls.from_json(d)

    def to_file(self, root: os.PathLike) -> None:
        """Write this metadata to its `zarr.json`.

        Merges into whatever is already at the path rather than
        overwriting the whole file.
        """
        path = root / constants.Z3_JSON
        data = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        data.update(self.to_json())
        _atomic_write(path, data)


@register_subclass(zarr_format=3, node_type="group")
@autofrozen
class GroupMetadataV3(NodeMetadataV3, GroupMetadata):
    """A Zarr v3 group's metadata: user attributes only."""


@register_subclass(zarr_format=3, node_type="array")
@autofrozen
class ArrayMetadataV3(NodeMetadataV3, ArrayMetadata):
    """The array-specific fields shared by every Zarr v3 array.

    See [ArrayMetadata][abczarr.metadata.v3.array.ArrayMetadata] for
    the concrete class with shape, data type, chunk grid, chunk-key
    encoding and codec-pipeline fields, and its `to_version` for
    conversion to v1 and v2.
    """


# ======================================================================
#
#                                 UTILS
#
# ======================================================================


def _atomic_write(path: os.PathLike, data: tz.JsonDict) -> None:
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
