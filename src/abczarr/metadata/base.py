"""
The version-independent metadata model.

Every node in a Zarr hierarchy, a group or an array, is described by
a small JSON document: `zarr.json` in Zarr v3, `.zarray` or
`.zgroup` plus `.zattrs` in v2, and `.zarray` plus `.zattrs` in v1.
This module defines the typed classes that document holds, one set
per format version, and the shared vocabulary that lets code work
with a node's metadata without caring which version produced it:
[ArrayMetadata][abczarr.metadata.base.ArrayMetadata] and
[GroupMetadata][abczarr.metadata.base.GroupMetadata].

[ArrayMetadata.to_version][abczarr.metadata.base.ArrayMetadata]
converts a node's metadata to another format version. Not every
version can represent everything another version can.
`ConversionPolicy` controls how such a conversion treats a field it
cannot carry over.

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

#: Protocols whose paths live on a local filesystem, and for which the
#: temp-file-and-replace atomic write is available. An empty protocol
#: is a plain local path. ``file`` and ``local`` are its aliases. A
#: remote store goes through the direct-write branch of
#: ``_atomic_write`` instead.
_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})

# ======================================================================
#
#                          CONVERSION POLICY
#
# ======================================================================

#: How a conversion treats a field the target version cannot hold.
#:
#: * ``"lossy"``, the default, drops the field without comment.
#: * ``"warn"`` drops the field and emits one warning naming it.
#: * ``"strict"`` raises
#:   [UnsupportedConversion][abczarr.errors.UnsupportedConversion]
#:   instead of dropping anything.
ConversionPolicy = tx.Literal["lossy", "warn", "strict"]


def _report_loss(
    policy: ConversionPolicy, field: str, version: tz.ZarrVersion
) -> None:
    """Apply a conversion policy to a field the target cannot hold.

    A version's `to_version` implementation calls this once for each
    field it cannot carry over to `version`. Under ``"lossy"`` the
    call does nothing. Under ``"warn"`` it emits a warning naming the
    field. Under ``"strict"`` it raises
    [UnsupportedConversion][abczarr.errors.UnsupportedConversion].

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
        If `policy` is ``"strict"``.
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

    A node is either a group or an array. Both kinds carry user
    attributes and a format version.
    [GroupMetadata][abczarr.metadata.base.GroupMetadata],
    [ArrayMetadata][abczarr.metadata.base.ArrayMetadata], and their
    per-version subclasses are the classes to construct. This class
    holds only what they share.
    """

    attributes: tz.JsonDict
    """The node's user-defined attributes, stored as arbitrary JSON."""
    zarr_format: tz.ZarrVersion = 3
    """The Zarr format version the metadata is written in: 1, 2 or 3."""
    node_type: tz.NodeType = "group"
    """Either ``"group"`` or ``"array"``."""

    # Convenience updaters (immutably return new metadata)
    def update_attributes(self, attributes: tz.JsonDict) -> tx.Self:
        """Return a copy of this metadata with new attributes.

        Every other field, such as shape, dtype, and chunking, is
        unchanged.

        Parameters
        ----------
        attributes : dict
            The user attributes the returned metadata carries,
            replacing the current ones in full.

        Returns
        -------
        NodeMetadata
            A new metadata object with `attributes` replaced.
        """
        return evolve(self, attributes=dict(attributes))

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a node's metadata from its directory.

        The Zarr format version is detected from which metadata file
        is present under `root`: `zarr.json` for v3, `.zarray` or
        `.zgroup` for v2, or `.zarray` for v1. The metadata is
        returned as an instance of the matching version's class.

        Parameters
        ----------
        root : PathLike
            The directory holding the node's metadata files.

        Returns
        -------
        NodeMetadata
            The loaded metadata, as an instance of the class
            matching the detected format version.

        Raises
        ------
        FileNotFoundError
            If `root` holds no recognized metadata file.
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
    """Report whether `root` holds a Zarr array, a group, or neither.

    Reads only enough of the metadata to answer that question, a v3
    `zarr.json`'s `node_type` field or which of `.zarray` and
    `.zgroup` a v2 node has, never the rest of the metadata. That
    keeps the check cheap enough to call on every child while listing
    a group.

    Parameters
    ----------
    root : PathLike
        The directory to inspect.

    Returns
    -------
    str or None
        ``"array"`` or ``"group"`` if `root` holds Zarr metadata of
        that kind, otherwise `None`.
    """
    detected = _node_at(root)
    return detected[0] if detected else None


def _node_at(
    root: os.PathLike,
) -> tx.Optional[tx.Tuple[tz.NodeType, tz.ZarrVersion]]:
    """Identify the kind and Zarr version of the node stored at `root`.

    Reads only enough of the metadata to answer the question: a v3
    `zarr.json`'s `node_type`, or which of `.zarray` and `.zgroup` a
    v2 node has. The format requires `node_type` on a v3
    `zarr.json`, so a document missing a valid value counts as no
    node. The value is never guessed at from the document's other
    contents.

    Parameters
    ----------
    root : PathLike
        The directory to inspect.

    Returns
    -------
    tuple or None
        A ``(node_type, version)`` pair, ``"array"`` or ``"group"``
        paired with 1, 2 or 3, for a directory that holds Zarr
        metadata, or `None` for one that does not.
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

    A group holds no data of its own, so beyond what
    [NodeMetadata][abczarr.metadata.base.NodeMetadata] already
    defines, this class adds nothing but its `node_type`. A member
    and its metadata are reached through the store, not through this
    object.
    """

    node_type: tx.Literal["group"] = "group"
    """Always ``"group"``."""

    def to_version(
        self,
        version: tz.ZarrVersion,
        policy: ConversionPolicy = "lossy",
    ) -> "GroupMetadata":
        """Convert this group's metadata to another Zarr version.

        A group carries only user attributes and a format version.
        Converting between v2 and v3 re-stamps the format and
        carries the attributes across unchanged, so nothing is lost
        and `policy` is never invoked. Zarr v1 has no group concept,
        so converting a group to v1 raises regardless of `policy`.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.
        policy : ConversionPolicy
            How to treat a field the target version cannot hold.
            This parameter exists to match the signature of
            [ArrayMetadata.to_version][abczarr.metadata.base.ArrayMetadata].
            No field of a group's metadata is affected by it between
            v2 and v3.

        Returns
        -------
        GroupMetadata
            Equivalent metadata for `version`. Converting to the group's
            own version returns this object unchanged.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3.
        UnsupportedConversion
            If `version` is 1, which has no group concept.
        """
        if version == self.zarr_format:
            return self
        if version == 1:
            # Zarr v1 predates groups entirely, so a group has no v1 form to
            # carry attributes into and the conversion cannot proceed under
            # any policy. This is a documented limitation, not a
            # policy-governed loss, so it raises a named error regardless of
            # `policy`.
            raise UnsupportedConversion("group", 1)
        if version in (2, 3):
            target = {2: GroupMetadataV2, 3: GroupMetadataV3}[version]
            return target(attributes=self.attributes)
        raise ValueError(f"Unsupported version: {version}")


@register_subclass(node_type="array")
@autofrozen
class ArrayMetadata(NodeMetadata):
    """An array's metadata: shape, data type, chunking and codecs.

    The exact fields depend on the Zarr format version. See
    [ArrayMetadataV1][abczarr.metadata.base.ArrayMetadataV1],
    [ArrayMetadataV2][abczarr.metadata.base.ArrayMetadataV2] and
    [ArrayMetadataV3][abczarr.metadata.base.ArrayMetadataV3] for the
    concrete field sets. Every version shares `to_version`, which
    converts the metadata between versions, and
    [required_features][abczarr.metadata.base.ArrayMetadata.required_features],
    which reports what a driver needs to support in order to read or
    write the array.
    """

    node_type: tx.Literal["array"] = "array"
    """Always ``"array"``."""

    def required_features(self) -> tx.FrozenSet[str]:
        """Report the features a driver needs to read or write this array.

        Each feature is a namespaced key built from the array's
        codecs, chunk grid, chunk-key encoding, and data type, such
        as ``"v3:codec:zstd"`` or ``"v2:filter:delta"``. A driver
        compares this set against what it supports to decide whether
        it can open the array. That names an unsupported codec up
        front, before a read ever starts.

        Every concrete array metadata class overrides this method
        with its own version-specific keys.

        Returns
        -------
        frozenset of str
            The feature keys this array requires. The base
            implementation returns an empty set.
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

    Zarr v1 has no groups, so every node is an array. Use
    [ArrayMetadataV1][abczarr.metadata.base.ArrayMetadataV1]
    directly, or build one through this class with
    [from_file][abczarr.metadata.base.NodeMetadataV1.from_file] or
    [from_json][abczarr._core.metadata.Metadata.from_json].
    """

    zarr_format: tx.Literal[1] = 1
    """Always ``1``."""

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a v1 array's metadata from its directory.

        The array's metadata is read from `.zarray`. Its user
        attributes are read from `.zattrs`, when that file is
        present.

        Parameters
        ----------
        root : PathLike
            The directory holding `.zarray` and, optionally,
            `.zattrs`.

        Returns
        -------
        ArrayMetadataV1
            The loaded array metadata. Zarr v1 has no groups, so
            called on this class the result is always an array.
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

        Called on this class directly, the method returns
        [ArrayMetadataV1][abczarr.metadata.base.ArrayMetadataV1]
        metadata, since Zarr v1 has no groups.

        Parameters
        ----------
        data : dict
            The merged content of `.zarray` and `.zattrs`, with the
            attributes under the key `"attributes"`. This is the
            same shape
            [to_json][abczarr._core.metadata.Metadata.to_json]
            produces.

        Returns
        -------
        ArrayMetadataV1
            The metadata built from `data`.
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
    the concrete class that adds the shape, chunking, data type and
    codec fields, and for its `to_version` method, which converts to
    v2 and v3.
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
    concrete field sets. Called on this class directly,
    [from_file][abczarr.metadata.base.NodeMetadataV2.from_file]
    works out which one applies.
    """

    zarr_format: tx.Literal[2] = 2
    """Always ``2``."""

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a v2 node's metadata from its directory.

        The node's metadata is read from `.zarray` or `.zgroup`, and
        its user attributes from `.zattrs`. Called on this class
        directly, the node type is detected from which of `.zarray`
        and `.zgroup` is present. Called on a group or array
        subclass, that subclass's own file is read directly.

        Parameters
        ----------
        root : PathLike
            The directory holding the node's metadata files.

        Returns
        -------
        NodeMetadataV2
            The loaded metadata, as a
            [GroupMetadataV2][abczarr.metadata.base.GroupMetadataV2]
            or an
            [ArrayMetadataV2][abczarr.metadata.base.ArrayMetadataV2].

        Raises
        ------
        FileNotFoundError
            If `root` holds neither `.zarray` nor `.zgroup`.
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

        The array or group fields are written to `.zarray` or
        `.zgroup`, and the user attributes to `.zattrs`. Each file is
        merged with whatever content is already there. Neither file
        is overwritten wholesale.

        Parameters
        ----------
        root : PathLike
            The directory the metadata files are written under.
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
    the concrete class that adds the shape, chunking, data type,
    compressor and filter fields, and for its `to_version` method,
    which converts to v1 and v3.
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
    concrete field sets. A v3 node's type is recorded in its own
    `zarr.json`, so its `node_type` need not be known in advance to
    load it.
    """

    zarr_format: tx.Literal[3] = 3
    """Always ``3``."""

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load a v3 node's metadata from its `zarr.json`.

        Parameters
        ----------
        root : PathLike
            The directory holding `zarr.json`.

        Returns
        -------
        NodeMetadataV3
            The loaded metadata.

        Raises
        ------
        FileNotFoundError
            If `root` holds no `zarr.json`.
        """
        zarr_json = root / constants.Z3_JSON
        if not zarr_json.exists():
            raise FileNotFoundError(
                f"No Zarr v3 metadata found in {root}. Expected: "
                f"{constants.Z3_JSON}"
            )
        with zarr_json.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_json(d)

    def to_file(self, root: os.PathLike) -> None:
        """Write this metadata to its `zarr.json`.

        The write merges into whatever content is already at the
        path. The file is not overwritten wholesale.

        Parameters
        ----------
        root : PathLike
            The directory `zarr.json` is written under.
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
    the concrete class that adds the shape, data type, chunk grid,
    chunk-key encoding and codec-pipeline fields, and for its
    `to_version` method, which converts to v1 and v2.
    """


# ======================================================================
#
#                                 UTILS
#
# ======================================================================


def _atomic_write(path: os.PathLike, data: tz.JsonDict) -> None:
    """Write JSON data to `path` so a reader never sees a partial file.

    A local filesystem gets the classic temp-file-and-rename dance: a
    sibling temporary file is written, ``fsync``ed, then moved onto
    the target with ``os.replace``, which POSIX guarantees is atomic.
    A remote or object store has no such rename, but a single object
    PUT is itself atomic at the object level, so the metadata is
    written directly through the store's own ``write_bytes``, and a
    reader still never observes a half-written `zarr.json`. This
    keeps the path-based group/array fallback working on remote
    backends.

    Which branch runs is read off the path's ``protocol`` attribute.
    ``""``, ``file`` and ``local`` count as local. A plain
    `pathlib.Path` has no such attribute and is treated as local too.

    The final write is atomic on both branches. The read, merge and
    write-back a caller performs around it is not, since it is not a
    compare-and-swap against a remote store. Two concurrent writers
    to the same node can still clobber each other there, the same
    race a local path already has. Transactional remote writes are a
    separate concern from this function.

    Parameters
    ----------
    path : PathLike
        The file to write.
    data : dict
        The JSON-compatible data to serialize and write.
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    protocol = getattr(path, "protocol", "")
    if protocol not in _LOCAL_PROTOCOLS:
        # No local temp file or rename on a remote store; a single object PUT
        # is atomic at the object level, which is the guarantee this needs.
        path.write_bytes(payload.encode("utf-8"))
        return
    PathType = type(path)
    fd, tmp = tempfile.mkstemp(prefix=".meta_tmp_", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        PathType(tmp).replace(path)
    finally:
        try:
            if PathType(tmp).exists():
                PathType(tmp).unlink()
        except Exception:
            pass
