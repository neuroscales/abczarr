"""The asynchronous Zarr node classes: the coroutine twins of the
synchronous surface.

[AsyncZarrNode][abczarr.abc.asynchronous.AsyncZarrNode] is the
coroutine twin of [ZarrNode][abczarr.abc.sync.ZarrNode], the base
class both the async array and the async group inherit from. Its
data I/O is coroutines. Its metadata, attributes, format version, and
capability query stay synchronous, because none of them touch the
node's data path.

Every async node has a synchronous twin over the same backend handle,
reached with
[as_sync][abczarr.abc.asynchronous.AsyncZarrNode.as_sync]. A
synchronous node's own
[as_async][abczarr.abc.sync.ZarrArray.as_async] returns the async
twin in the other direction. The non-blocking accessors delegate to
the synchronous twin, so the two colors always report the same
metadata, attributes and backend features.

[AsyncZarrArray][abczarr.abc.asynchronous.AsyncZarrArray] reads and
writes through methods rather than indexing: `await
array.getitem(index)` and `await array.setitem(index, value)`.
[AsyncZarrGroup][abczarr.abc.asynchronous.AsyncZarrGroup] reaches a
member with `await group.getitem(name)`, iterates member names with
`async for`, and creates children with `await`. It has two concrete
implementations: the native
[AsyncPathGroup][abczarr.abc.asynchronous.AsyncPathGroup], and the
thread-pool fallback
[ThreadedAsyncGroup][abczarr.abc.asynchronous.ThreadedAsyncGroup].
"""

__all__ = [
    "AsyncZarrNode",
    "AsyncZarrArray",
    "ThreadedAsyncArray",
    "AsyncZarrGroup",
    "AsyncPathGroup",
    "ThreadedAsyncGroup",
]

# stdlib
import json
import os
from abc import ABC, abstractmethod

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import constants
from abczarr._core import typing as tz
from abczarr._core.asyncutils import run_sync
from abczarr._core.attributes import NodeAttributes, attribute_writes
from abczarr.api.config import ArrayConfig, ArrayOptions
from abczarr.metadata.base import NodeMetadata
from abczarr.ome.node import (
    merge_ome,
    ome_delete_plan,
    ome_write_plan,
    read_ome,
)

# locals
from .capabilities import Support, SupportsCapabilities
from .store import AsyncPathBasedStore
from .sync import (
    PathGroup,
    ZarrArray,
    ZarrGroup,
    ZarrNode,
    _resolve_array_config,
)

if tx.TYPE_CHECKING:
    from abczarr.ome.base import OME


class AsyncZarrNode(SupportsCapabilities, ABC):
    """The coroutine twin of a [ZarrNode][abczarr.abc.sync.ZarrNode].

    Only data I/O differs from the synchronous node: it is coroutines
    here. The non-blocking accessors, including metadata, attributes,
    format version, location, and the capability query, delegate to
    the synchronous twin, so both colors always agree. Use
    [capability][abczarr.abc.capabilities.SupportsCapabilities.capability]
    or
    [supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
    to check whether the async surface is native to the backend or
    synthesized in a thread pool.
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
        """The underlying backend object, or `None` when the node has
        none.

        This property is the escape hatch for anything the uniform
        node surface does not expose.
        """
        return self.as_sync().native

    @property
    def metadata(self) -> NodeMetadata:
        """This node's Zarr metadata."""
        return self.as_sync().metadata

    @property
    def attrs(self) -> NodeAttributes:
        """This node's user attributes, as a read-only, cached
        mapping.

        Reading this mapping is synchronous. The values come from the
        node's cached metadata, so no I/O is needed and there is
        nothing to await. Assigning a single key cannot be awaited,
        so this mapping has no per-key async setter. Persisting a
        change requires
        [update_attributes][abczarr.abc.asynchronous.AsyncZarrNode.update_attributes],
        which writes the whole set of attributes through the node's
        async persistence path.
        """
        return self.as_sync().attrs

    @property
    def ome(self) -> "tx.Optional[OME]":
        """This node's OME-Zarr metadata as a typed object, read only.

        Reading this property is synchronous, like
        [attrs][abczarr.abc.asynchronous.AsyncZarrNode.attrs]. The
        metadata is parsed from the node's cached attributes, so no
        I/O is needed. The property returns the matching version's
        [OME][abczarr.ome.base.OME] object, or `None` when the node
        carries no OME metadata. Writing cannot be awaited through a
        plain assignment, so this property has no setter. Use
        [set_ome][abczarr.abc.asynchronous.AsyncZarrNode.set_ome],
        [update_ome][abczarr.abc.asynchronous.AsyncZarrNode.update_ome],
        or [del_ome][abczarr.abc.asynchronous.AsyncZarrNode.del_ome]
        to persist a change, the same reason the async node writes
        attributes through `update_attributes` rather than through
        item assignment.
        """

        return read_ome(self)

    async def set_ome(
        self, value: "tx.Union[OME, tz.JsonDict]"
    ) -> "AsyncZarrNode":
        """Store OME metadata on this node, and persist it.

        This method is the coroutine twin of assigning
        [ZarrNode.ome][abczarr.abc.sync.ZarrNode.ome]. It serializes
        *value* into the envelope its version calls for: the
        ``"ome"`` attribute from version 0.5 on, or the bare
        top-level attribute keys before 0.5. The serialized metadata
        is then written through the node's async persistence path.
        Any OME metadata already present is replaced, and every other
        attribute is left untouched.

        Parameters
        ----------
        value : OME or dict
            The metadata to store, typed or a plain mapping carrying a
            ``version``.

        Returns
        -------
        AsyncZarrNode
            This node, with the metadata visible on
            [ome][abczarr.abc.asynchronous.AsyncZarrNode.ome].
        """

        await self._apply_ome_plan(ome_write_plan, value)
        return self

    async def update_ome(
        self, ome: "tx.Union[OME, tz.JsonDict]"
    ) -> "AsyncZarrNode":
        """Merge OME metadata into this node's existing OME metadata,
        and persist the result.

        This method is the coroutine twin of
        [ZarrNode.update_ome][abczarr.abc.sync.ZarrNode.update_ome].
        A top-level key present in *ome* replaces the value already
        stored under that key on the node's current OME metadata. A
        key the node already carries that *ome* does not name is
        kept. When the node has no OME metadata yet and the merged
        result still names no version, the metadata is written with
        the latest released OME version.

        The merge is shallow. A nested structure, such as a
        multiscale or a plate definition, is replaced as a whole
        rather than merged recursively. For a structured edit, read
        [ome][abczarr.abc.asynchronous.AsyncZarrNode.ome], change the
        typed object, and pass the result to
        [set_ome][abczarr.abc.asynchronous.AsyncZarrNode.set_ome].

        Parameters
        ----------
        ome : OME or dict
            The metadata whose top-level keys are merged in.

        Returns
        -------
        AsyncZarrNode
            This node, with the merged metadata visible on
            [ome][abczarr.abc.asynchronous.AsyncZarrNode.ome].
        """

        return await self.set_ome(merge_ome(read_ome(self), ome))

    async def del_ome(self) -> "AsyncZarrNode":
        """Remove this node's OME-Zarr metadata, and persist the
        removal.

        This method is the coroutine twin of `del node.ome` on
        [ZarrNode.ome][abczarr.abc.sync.ZarrNode.ome]. It drops the
        OME attribute keys of whichever envelope the node uses. Every
        other attribute is left untouched. A node that carries no OME
        metadata is left unchanged.

        Returns
        -------
        AsyncZarrNode
            This node.
        """

        await self._apply_ome_plan(ome_delete_plan)
        return self

    async def _apply_ome_plan(
        self, plan: tx.Callable, *args: tx.Any
    ) -> None:
        """Apply an OME attribute *plan* through this node's async write path.

        *plan* takes the current attributes (plus, for a write, the value to
        store) and returns ``(payload, stale)``; this persists the full
        result -- unrelated attributes carried over, stale OME keys dropped
        -- by awaiting the async metadata write. The one async persistence
        path that ``set_ome`` and ``del_ome`` share.
        """
        sync = self.as_sync()
        current = dict(sync.metadata.attributes)
        payload, stale = plan(current, *args)
        new = {k: v for k, v in current.items() if k not in stale}
        new.update(payload)
        await self._awrite_metadata(sync.metadata.update_attributes(new))

    async def update_attributes(
        self, attributes: tz.JsonDict
    ) -> "AsyncZarrNode":
        """Add or replace several attributes at once, and persist the
        change.

        This method is the coroutine twin of
        [ZarrNode.update_attributes][abczarr.abc.sync.ZarrNode.update_attributes].
        The keys in *attributes* are merged into this node's existing
        attributes. The merged result is written through the node's
        async persistence path. The behavior mirrors zarr-python's
        own async `update_attributes`.

        !!! example
            ```python
            await node.update_attributes({"unit": "micrometer"})
            ```

        Parameters
        ----------
        attributes : dict
            The attributes to add or replace. Values must be JSON-compatible.

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
            existing = json.loads(raw) if raw else new_metadata.to_json()
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

        For `"async"`, the answer is this twin's own: native or
        synthesized, depending on the backend. For every other name,
        the answer comes from the synchronous twin, so the two colors
        always report the same backend features.
        """
        if name == "async":
            return self._async_support
        return self.as_sync().capability(name)


class AsyncZarrArray(AsyncZarrNode):
    """The coroutine twin of a [ZarrArray][abczarr.abc.sync.ZarrArray].

    An `AsyncZarrArray` is read and written with `await`, through the
    `getitem` and `setitem` methods rather than indexing.

    !!! example
        ```python
        block = await array.getitem((slice(0, 64), slice(0, 64)))
        await array.setitem((slice(0, 64), slice(0, 64)), block * 2)
        ```
    """

    def __init__(self, sync: ZarrArray) -> None:
        self._sync = sync

    def as_sync(self) -> ZarrArray:
        """The synchronous [ZarrArray][abczarr.abc.sync.ZarrArray] twin."""
        return self._sync

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        return self._sync.ndim

    @property
    def shape(self) -> tz.Shape:
        """The shape of the array."""
        return self._sync.shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the array."""
        return self._sync.dtype

    @property
    def chunks(self) -> tz.Shape:
        """The chunk shape of the array."""
        return self._sync.chunks

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        """The shard shape of the array, or `None` if it is not sharded."""
        return self._sync.shards

    @abstractmethod
    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        """Read data from the array at *index* (a NumPy-style selection)."""
        ...

    @abstractmethod
    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        """Write *value* at *index* (a NumPy-style selection)."""
        ...


class ThreadedAsyncArray(AsyncZarrArray):
    """An [AsyncZarrArray][abczarr.abc.asynchronous.AsyncZarrArray]
    that runs a synchronous array's reads and writes in a bounded
    thread pool.

    This class is the default async array for a backend that has no
    coroutine surface of its own. It reports `"async"` as
    `Support.SYNTHESIZED`.
    """

    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        return await run_sync(self._sync.__getitem__, index)

    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        await run_sync(self._sync.__setitem__, index, value)


class AsyncZarrGroup(AsyncZarrNode):
    """The coroutine twin of a [ZarrGroup][abczarr.abc.sync.ZarrGroup].

    An `AsyncZarrGroup` reaches a member with `await
    group.getitem(name)`, iterates member names with `async for`,
    and creates a subgroup or an array with `await`.

    !!! example
        ```python
        child = await group.getitem("images")
        async for name in group:
            ...
        array = await group.create_array("labels", (8, 8), "uint8")
        ```
    """

    def __init__(self, sync: ZarrGroup) -> None:
        self._sync = sync

    def as_sync(self) -> ZarrGroup:
        """The synchronous [ZarrGroup][abczarr.abc.sync.ZarrGroup] twin."""
        return self._sync

    @abstractmethod
    async def getitem(self, key: str) -> AsyncZarrNode:
        """Open the subgroup or array named *key* as an async node."""
        ...

    @abstractmethod
    def keys(self) -> tx.AsyncIterator[str]:
        """Async-iterate the names of this group's members."""
        ...

    def __aiter__(self) -> tx.AsyncIterator[str]:
        return self.keys()

    async def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        config: tx.Union[ArrayConfig, ArrayOptions, None] = None,
        **options: tx.Unpack[ArrayOptions],
    ) -> AsyncZarrArray:
        """Create a new array named *name* within this group.

        This method mirrors
        [ZarrGroup.create_array][abczarr.abc.sync.ZarrGroup.create_array].
        See that method for the parameters it accepts.
        """
        resolved = _resolve_array_config(
            shape, dtype, config, options, self.zarr_version
        )
        return await self._create_array(name, resolved)

    @abstractmethod
    async def _create_array(
        self, name: str, config: ArrayConfig
    ) -> AsyncZarrArray:
        """Create the array named *name* from a resolved *config*."""
        ...

    @abstractmethod
    async def create_group(
        self, name: str, overwrite: bool = False
    ) -> "AsyncZarrGroup":
        """Create or open a subgroup named *name*."""
        ...


class AsyncPathGroup(AsyncZarrGroup):
    """The async twin of [PathGroup][abczarr.abc.sync.PathGroup].

    Listing and navigating members is genuinely non-blocking, carried
    out through an async store rather than a synchronous group run in
    a thread. Array children come back in the async color. When the
    underlying backend is itself natively async, a child array is
    that backend's own native async array, not a synchronous array
    bridged through a thread. Creating a subgroup or an array still
    blocks, because writing metadata and building a backend handle
    are inherently synchronous operations.

    This group's own `"async"` capability always reports
    `Support.SYNTHESIZED`, even when the underlying store is itself
    natively async. `Support.NATIVE` is reserved for async support
    that a backend provides directly.
    """

    # a path group synthesizes group semantics over a store; async is
    # synthesized even when the store itself awaits natively
    _async_support = Support.SYNTHESIZED

    def __init__(self, sync: PathGroup) -> None:
        super().__init__(sync)
        self._store = AsyncPathBasedStore(str(sync.store_path))

    async def _node_at(
        self, prefix: str
    ) -> tx.Optional[tx.Tuple[tz.NodeType, tz.ZarrVersion]]:
        """The kind and Zarr version of the node at key *prefix* -- `""` for
        this group itself, a member name for a child -- read through the
        async store, or `None` when there is no Zarr node there.

        Mirrors [_node_at][abczarr.metadata.base] over the async store: a v3
        `zarr.json`'s `node_type`, else which v2/v1 metadata file is present.
        A *prefix* that names a plain file, not a directory (the group's own
        `zarr.json` shows up in the listing), is simply not a node.
        """
        base = (prefix + "/") if prefix else ""
        try:
            raw = await self._store.get(base + constants.Z3_JSON)
            if raw is not None:
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError):
                    return None
                node_type = data.get("node_type")
                if node_type in ("array", "group"):
                    return node_type, 3
                return None
            if await self._store.exists(base + constants.Z2ARRAY_JSON):
                return "array", 2
            if await self._store.exists(base + constants.Z2GROUP_JSON):
                return "group", 2
            if await self._store.exists(base + constants.Z1META_JSON):
                return "array", 1
        except OSError:
            # *prefix* is a file, so "<prefix>/<metadata>" is not a directory
            return None
        return None

    async def _version(self) -> tz.ZarrVersion:
        """This group's own Zarr format version, read through the async
        store."""
        detected = await self._node_at("")
        return detected[1] if detected is not None else self.zarr_version

    async def _member(
        self, name: str, version: tz.ZarrVersion
    ) -> tx.Optional[tx.Tuple[tz.NodeType, tz.ZarrVersion]]:
        """The child *name*'s kind and version when it is a member of this
        group -- a node written in the group's *version* -- else `None`. A
        Zarr hierarchy is written in a single version, so a child of another
        is not a member."""
        detected = await self._node_at(name)
        if detected is None or detected[1] != version:
            return None
        return detected

    async def keys(self) -> tx.AsyncIterator[str]:
        version = await self._version()
        async for name in self._store.list_dir(""):
            if await self._member(name, version) is not None:
                yield name

    async def getitem(self, key: str) -> AsyncZarrNode:
        detected = await self._member(key, await self._version())
        if detected is None:
            raise KeyError(key)
        child_path = self._sync.store_path / key
        if detected[0] == "group":
            child = type(self._sync)(child_path, self._sync._mode)
            return type(self)(child)
        # opening the array builds a backend handle (fast); its data I/O is
        # what the async color makes asynchronous, through as_async()
        array = self._sync._open_array(child_path)
        return array.as_async()

    async def _create_array(
        self, name: str, config: ArrayConfig
    ) -> AsyncZarrArray:
        array = await run_sync(self._sync._create_array, name, config)
        return tx.cast(AsyncZarrArray, array.as_async())

    async def create_group(
        self, name: str, overwrite: bool = False
    ) -> "AsyncPathGroup":
        child = await run_sync(self._sync.create_group, name, overwrite)
        return type(self)(child)


class ThreadedAsyncGroup(AsyncZarrGroup):
    """An [AsyncZarrGroup][abczarr.abc.asynchronous.AsyncZarrGroup]
    that runs a synchronous group's navigation and creation in a
    bounded thread pool.

    This class is the fallback for a group that is neither natively
    async nor path-based. A path-based backend gets the native
    [AsyncPathGroup][abczarr.abc.asynchronous.AsyncPathGroup]
    instead. Members this group opens are handed back in the async
    color.
    """

    async def getitem(self, key: str) -> AsyncZarrNode:
        child = await run_sync(self._sync.__getitem__, key)
        return child.as_async()

    async def keys(self) -> tx.AsyncIterator[str]:
        names = await run_sync(lambda: list(self._sync.keys()))
        for name in names:
            yield name

    async def _create_array(
        self, name: str, config: ArrayConfig
    ) -> AsyncZarrArray:
        child = await run_sync(self._sync._create_array, name, config)
        return tx.cast(AsyncZarrArray, child.as_async())

    async def create_group(
        self, name: str, overwrite: bool = False
    ) -> AsyncZarrGroup:
        child = await run_sync(self._sync.create_group, name, overwrite)
        return tx.cast(AsyncZarrGroup, child.as_async())
