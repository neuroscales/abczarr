"""The base of every async node.

[AsyncZarrNode][abczarr.abc.async_node.AsyncZarrNode] is the coroutine twin
of [ZarrNode][abczarr.abc.node.ZarrNode], the common ancestor of the async
array and group. Its I/O is coroutines; the metadata, attributes, version
and capability query mirror the sync node and stay synchronous, because they
never touch the node's data path.

Every async node has a synchronous twin over the same backend handle, reached
with [as_sync][abczarr.abc.async_node.AsyncZarrNode.as_sync]; the sync node's
[as_async][abczarr.abc.array.ZarrArray.as_async] returns the async one. The
non-blocking accessors delegate to the sync twin, so the two colors always
report the same metadata, attributes and backend features.
"""

__all__ = [
    "AsyncZarrNode",
]

# stdlib
import json
import os
from abc import ABC, abstractmethod

# dependencies
import typing_extensions as tx

# core
from abczarr._core import constants
from abczarr._core import typing as tz
from abczarr._core.attributes import NodeAttributes, attribute_writes
from abczarr.metadata.base import NodeMetadata

# locals
from .capabilities import Support, SupportsCapabilities
from .node import ZarrNode
from .store import AsyncPathBasedStore


class AsyncZarrNode(SupportsCapabilities, ABC):
    """The coroutine twin of a [ZarrNode][abczarr.abc.node.ZarrNode].

    Only the I/O differs from the sync node: it is coroutines. The
    non-blocking accessors -- metadata, attributes, version, location and
    the capability query -- delegate to the sync twin, so both colors agree.
    Use [capability][abczarr.abc.capabilities.SupportsCapabilities.capability]
    or [supports][abczarr.abc.capabilities.SupportsCapabilities.supports] to
    check whether the async surface is native to the backend or synthesized.
    """

    #: How the async surface is provided: `Support.NATIVE` when the backend
    #: (or the async store beneath a path group) drives the coroutines, and
    #: `Support.SYNTHESIZED` when abczarr runs the sync ops in a thread pool.
    _async_support = Support.SYNTHESIZED  # type: tx.ClassVar[Support]

    @abstractmethod
    def as_sync(self) -> ZarrNode:
        """The synchronous twin over the same backend handle."""
        ...

    @property
    def store_path(self) -> os.PathLike:
        """The path to this node's location in its store."""
        return self.as_sync().store_path

    @property
    def native(self) -> tx.Any:
        """The underlying backend object, or `None` -- the escape hatch for
        anything the uniform surface does not name."""
        return self.as_sync().native

    @property
    def metadata(self) -> NodeMetadata:
        """This node's Zarr metadata."""
        return self.as_sync().metadata

    @property
    def attrs(self) -> NodeAttributes:
        """This node's user attributes, as a read-cached mapping.

        Reads stay synchronous on the async twin -- they come from the cached
        metadata, so there is nothing to await. Writing a single key cannot be
        awaited (an assignment expression is not a coroutine), so there is no
        per-key async setter; use
        [update_attributes][abczarr.abc.async_node.AsyncZarrNode.update_attributes]
        to persist a change, the same reason the async array writes with
        `setitem` rather than `[]`.
        """
        return self.as_sync().attrs

    async def update_attributes(
        self, attributes: tz.JsonDict
    ) -> "AsyncZarrNode":
        """Add or replace several attributes at once, and persist them.

        The coroutine twin of
        [ZarrNode.update_attributes][abczarr.abc.node.ZarrNode.update_attributes]:
        the *attributes* are merged into this node's existing attributes and
        the change is written through the node's async persistence path.
        Mirrors zarr-python's async `update_attributes`.

        !!! example
            ```python
            await node.update_attributes({"unit": "micrometer"})
            ```

        Parameters
        ----------
        attributes : dict
            The attributes to add or replace. Values must be Json-compatible.

        Returns
        -------
        AsyncZarrNode
            This node, with the updated attributes visible on `attrs` and
            `metadata`.
        """
        sync = self.as_sync()
        merged = dict(sync.metadata.attributes)
        merged.update(attributes)
        new_metadata = sync.metadata.update_attributes(merged)
        await self._awrite_metadata(new_metadata)
        return self

    async def _awrite_metadata(self, new_metadata: NodeMetadata) -> None:
        """Persist *new_metadata*, then update the sync twin's cache.

        The default rewrites the node's metadata document through an
        [AsyncStore][abczarr.abc.store.AsyncStore] over the node's location,
        so the write goes through the store rather than straight to a file. A
        driver that wraps a live Zarr object overrides this to delegate to
        that object's own async `update_attributes`.
        """
        sync = self.as_sync()
        store = AsyncPathBasedStore(str(sync.store_path))
        version = new_metadata.zarr_format
        existing = None  # type: tx.Optional[tx.Dict[str, tx.Any]]
        if version >= 3:
            raw = await store.get(constants.Z3_JSON)
            existing = json.loads(raw) if raw else new_metadata.to_dict()
        for key, value in attribute_writes(
            version, new_metadata.attributes, existing
        ):
            await store.set(key, value)
        sync._cache_metadata(new_metadata)

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        """The Zarr format version this node was written with."""
        return self.as_sync().zarr_version

    def capability(self, name: str) -> Support:
        """How this async node provides the capability *name*.

        The answer for `"async"` is this twin's own -- native or
        synthesized -- and every other name is answered by the sync twin,
        so the two colors report the same backend features.
        """
        if name == "async":
            return self._async_support
        return self.as_sync().capability(name)
