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

        Reads stay synchronous on the async twin -- they come from the cached
        metadata, so there is nothing to await. Writing a single key cannot be
        awaited (an assignment expression is not a coroutine), so there is no
        per-key async setter; use
        [update_attributes][abczarr.abc.asynchronous.AsyncZarrNode.update_attributes]
        to persist a change, the same reason the async array writes with
        `setitem` rather than `[]`.
        """
        return self.as_sync().attrs

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

    A backend with no group object of its own -- TensorStore opens arrays
    only -- gets a real async group here: it does its own I/O through an
    [AsyncStore][abczarr.abc.store.AsyncStore] over the group's location,
    listing and navigating members with `await store.list_dir(...)` /
    `await store.get(...)` rather than threading the sync group. Array
    children are opened in the async color -- a natively async backend's
    array comes back as its native async array (whose own `"async"` is
    `NATIVE`).

    Its own `"async"` capability is `Support.SYNTHESIZED`: a path group is
    abczarr assembling group semantics over a key-value store, not a group
    a backend provides natively -- so it reports synthesized even when the
    underlying store awaits a natively async backend. `NATIVE` is reserved
    for a surface the backend itself supplies.

    Creation still writes through the sync group in a thread pool -- writing
    metadata or building a backend array blocks, and is not the listing work
    this group makes asynchronous.
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
