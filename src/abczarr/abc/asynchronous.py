"""The asynchronous Zarr nodes: the coroutine twins of the sync surface.

[AsyncZarrNode][abczarr.abc.asynchronous.AsyncZarrNode] is the coroutine twin
of [ZarrNode][abczarr.abc.sync.ZarrNode], the common ancestor of the async
array and group. Its I/O is coroutines; the metadata, attributes, version
and capability query mirror the sync node and stay synchronous, because they
never touch the node's data path.

Every async node has a synchronous twin over the same backend handle, reached
with [as_sync][abczarr.abc.asynchronous.AsyncZarrNode.as_sync]; the sync
node's [as_async][abczarr.abc.sync.ZarrArray.as_async] returns the async one.
The non-blocking accessors delegate to the sync twin, so the two colors always
report the same metadata, attributes and backend features.

* [AsyncZarrArray][abczarr.abc.asynchronous.AsyncZarrArray] reads and writes
  through **methods**, not `[]`: `await array.getitem(index)` and
  `await array.setitem(index, value)`.
* [AsyncZarrGroup][abczarr.abc.asynchronous.AsyncZarrGroup] reaches a member
  with `await group.getitem(name)`, iterates member names with `async for`,
  and creates children with `await`. Its two concrete twins are the native
  [AsyncPathGroup][abczarr.abc.asynchronous.AsyncPathGroup] and the
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

        Reads are synchronous. They come from cached metadata, so there is
        nothing to await. Assigning a single key cannot be awaited, so no
        per-key async setter exists. Use
        [update_attributes][abczarr.abc.asynchronous.AsyncZarrNode.update_attributes]
        to persist a change.
        """
        return self.as_sync().attrs

    @property
    def ome(self) -> "tx.Optional[OME]":
        """This node's OME-Zarr metadata as a typed object -- read only.

        Reads stay synchronous, like
        [attrs][abczarr.abc.asynchronous.AsyncZarrNode.attrs]: the
        metadata is parsed from the cached attributes, so there is
        nothing to await. Returns the right version's
        [OME][abczarr.ome.base.OME] object, or `None` when the node
        carries none. Writing cannot be awaited through an assignment, so
        there is no setter; use
        [set_ome][abczarr.abc.asynchronous.AsyncZarrNode.set_ome],
        [update_ome][abczarr.abc.asynchronous.AsyncZarrNode.update_ome] or
        [del_ome][abczarr.abc.asynchronous.AsyncZarrNode.del_ome] to
        persist a change, the same reason the async node writes
        attributes with `update_attributes` rather than `[]`.
        """

        return read_ome(self)

    async def set_ome(
        self, value: "tx.Union[OME, tz.JsonDict]"
    ) -> "AsyncZarrNode":
        """Store OME metadata on this node, and persist it.

        The coroutine twin of assigning
        [ZarrNode.ome][abczarr.abc.sync.ZarrNode.ome]: serialize *value*
        into the envelope its version calls for (the ``"ome"`` attribute
        from 0.5 on, the bare attribute keys up to 0.4) and write it through
        the node's async persistence path, replacing any OME metadata
        already present and leaving unrelated attributes untouched.

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
        """Shallow-merge OME metadata into this node's, and persist it.

        The coroutine twin of
        [ZarrNode.update_ome][abczarr.abc.sync.ZarrNode.update_ome]: the
        top-level keys of *ome* replace those on the node's current OME
        metadata; the rest are kept. When the node has no OME metadata yet
        and the result still names no version, it defaults to the latest
        released OME version. The merge is shallow -- for a structured edit,
        read [ome][abczarr.abc.asynchronous.AsyncZarrNode.ome], change the
        typed object, and pass it to
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
        """Remove this node's OME-Zarr metadata, and persist the removal.

        The coroutine twin of `del node.ome`
        ([ZarrNode.ome][abczarr.abc.sync.ZarrNode.ome]): drops the OME
        attribute keys of whichever envelope the node uses, leaving
        unrelated attributes untouched. A node with no OME metadata is left
        unchanged.

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
        """Add or replace several attributes at once, and persist them.

        The coroutine twin of
        [ZarrNode.update_attributes][abczarr.abc.sync.ZarrNode.update_attributes]:
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

        The answer for `"async"` is this twin's own -- native or
        synthesized -- and every other name is answered by the sync twin,
        so the two colors report the same backend features.
        """
        if name == "async":
            return self._async_support
        return self.as_sync().capability(name)


class AsyncZarrArray(AsyncZarrNode):
    """The coroutine twin of a [ZarrArray][abczarr.abc.sync.ZarrArray].

    Read and write it with `await`:

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
    """An [AsyncZarrArray][abczarr.abc.asynchronous.AsyncZarrArray] that runs a
    sync array's reads and writes in a bounded thread pool.

    The default async array for a backend that has no coroutine surface of
    its own. It reports `"async"` as `Support.SYNTHESIZED`.
    """

    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        return await run_sync(self._sync.__getitem__, index)

    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        await run_sync(self._sync.__setitem__, index, value)


class AsyncZarrGroup(AsyncZarrNode):
    """The coroutine twin of a [ZarrGroup][abczarr.abc.sync.ZarrGroup].

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

        Mirrors
        [ZarrGroup.create_array][abczarr.abc.sync.ZarrGroup.create_array];
        see it for the parameters.
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

    Listing and navigating members is genuinely non-blocking. Array
    children come back in the async color. When the underlying backend
    is natively async, a child array is that backend's own native
    async array rather than a synchronous array run in a thread.
    Creating a subgroup or array still blocks, since writing metadata
    or building a backend handle is inherently synchronous work.

    `AsyncPathGroup`'s `"async"` capability is always
    `Support.SYNTHESIZED`, even when the underlying store is itself
    natively async. `NATIVE` is reserved for async support that a
    backend supplies directly.
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
    """An [AsyncZarrGroup][abczarr.abc.asynchronous.AsyncZarrGroup] that runs a
    sync group's navigation and creation in a bounded thread pool.

    The fallback for a group that is neither natively async nor path-based
    (a path-based backend gets the real
    [AsyncPathGroup][abczarr.abc.asynchronous.AsyncPathGroup] instead).
    Members it opens are handed back in the async color.
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
