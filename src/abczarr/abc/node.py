"""The base of every Zarr object.

[ZarrNode][abczarr.abc.node.ZarrNode] is the common ancestor of
[ZarrArray][abczarr.abc.array.ZarrArray] and
[ZarrGroup][abczarr.abc.group.ZarrGroup]: metadata, attributes, the
Zarr format version, and the capability query all live here.
"""

__all__ = [
    "ZarrNode",
    "KNOWN_CAPABILITIES",
    "Support",
]

# stdlib
import json
import os
from abc import ABC, abstractmethod

# dependencies
import typing_extensions as tx
from bagof.paths import Path

# core
from abczarr._core import constants
from abczarr._core import typing as tz
from abczarr._core.attributes import NodeAttributes, attribute_writes
from abczarr.abc.store import PathBasedStore
from abczarr.metadata.base import NodeMetadata

# locals -- KNOWN_CAPABILITIES and Support are re-exported for callers that
# reach them through this module (they are listed in __all__).
from .capabilities import (  # noqa: F401
    KNOWN_CAPABILITIES,
    Support,
    SupportsCapabilities,
)


class ZarrNode(SupportsCapabilities, ABC):
    """Base class for any Zarr object: a group or an array.

    Use
    [capability][abczarr.abc.capabilities.SupportsCapabilities.capability]
    or
    [supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
    to check what a node's backend can do.
    """

    def __init__(self, store_path: tz.PathLike) -> None:
        if isinstance(store_path, (str, bytes)):
            store_path = Path(store_path)
        self._store_path = store_path
        # The raw backend object (a zarr.Array, a tensorstore.TensorStore,
        # ...). A concrete driver sets it; it stays None where a node has no
        # single backing object (e.g. a group that is only a path).
        self._native: tx.Any = None
        # The node's metadata, loaded once and kept in memory. A node that
        # reads its metadata from a store or a metadata file caches it here
        # (the I/O is the open); a node backed by a live Zarr object reads
        # from that object instead and leaves this None.
        self._cached_metadata: tx.Optional[NodeMetadata] = None

    @property
    def store_path(self) -> os.PathLike:
        """The path to this node's location in its store."""
        return self._store_path

    @property
    def native(self) -> tx.Any:
        """The underlying backend object, or `None`.

        The escape hatch: anything the uniform surface does not name
        is still reachable through the backend object itself -- a
        `zarr.Array`, a `tensorstore.TensorStore`, and so on.
        """
        return self._native

    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        """This node's Zarr metadata."""
        ...

    @property
    def attrs(self) -> NodeAttributes:
        """This node's user attributes, as a live, write-through mapping.

        Reads are served from this node's metadata, the single source of
        truth. Mutations persist: `node.attrs["k"] = v` adds or replaces `k`
        and `del node.attrs["k"]` removes it, both routed through the node's
        persistence path.

        !!! example
            ```python
            node.attrs["unit"] = "micrometer"
            del node.attrs["unit"]
            ```
        """
        return NodeAttributes(self)

    def update_attributes(self, attributes: tz.JsonDict) -> "ZarrNode":
        """Add or replace several attributes at once, and persist them.

        The *attributes* are merged into this node's existing attributes --
        an existing key is overwritten, the rest are kept -- and the change is
        written through the node's persistence path. Mirrors zarr-python's
        `update_attributes`.

        !!! example
            ```python
            node.update_attributes({"unit": "micrometer", "scale": 0.5})
            ```

        Parameters
        ----------
        attributes : dict
            The attributes to add or replace. Values must be Json-compatible.

        Returns
        -------
        ZarrNode
            This node, with the updated attributes visible on
            [attrs][abczarr.abc.node.ZarrNode.attrs] and
            [metadata][abczarr.abc.node.ZarrNode.metadata].
        """
        merged = dict(self.metadata.attributes)
        merged.update(attributes)
        return self._replace_attributes(merged)

    def _replace_attributes(self, attributes: tz.JsonDict) -> "ZarrNode":
        """Replace this node's attributes wholesale, and persist them."""
        new_metadata = self.metadata.update_attributes(attributes)
        self._write_metadata(new_metadata)
        return self

    def _write_metadata(self, new_metadata: NodeMetadata) -> None:
        """Persist *new_metadata*, then update the cached metadata.

        The default rewrites the node's metadata document through a
        [Store][abczarr.abc.store.Store] over the node's location, so the
        write goes through the store rather than straight to a file. A driver
        that wraps a live Zarr object overrides this to delegate to that
        object, keeping the backend's own caches consistent.
        """
        store = PathBasedStore(str(self._store_path))
        version = new_metadata.zarr_format
        existing = None  # type: tx.Optional[tx.Dict[str, tx.Any]]
        if version >= 3:
            raw = store.get(constants.Z3_JSON)
            existing = json.loads(raw) if raw else new_metadata.to_dict()
        for key, value in attribute_writes(
            version, new_metadata.attributes, existing
        ):
            store.set(key, value)
        self._cache_metadata(new_metadata)

    def _cache_metadata(self, metadata: NodeMetadata) -> None:
        """Record *metadata* as this node's in-memory metadata."""
        self._cached_metadata = metadata

    @property
    @abstractmethod
    def zarr_version(self) -> tz.ZarrVersion:
        """The Zarr format version this node was written with."""
        ...
