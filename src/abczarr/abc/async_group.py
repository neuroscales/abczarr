"""The async group interface: the coroutine twin of the Zarr group.

[AsyncZarrGroup][abczarr.abc.async_group.AsyncZarrGroup] reaches a member
with `await group.getitem(name)`, iterates member names with `async for`
(`keys` / `__aiter__`), and creates children with `await`.

Two concrete twins live here:

* [AsyncPathGroup][abczarr.abc.async_group.AsyncPathGroup] is the async twin
  of [PathGroup][abczarr.abc.group.PathGroup]: a backend with no group object
  of its own (TensorStore, zarrista) gets a real async group that does its
  own listing and navigation through an
  [AsyncStore][abczarr.abc.store.AsyncStore] -- `await store.list_dir(...)` /
  `await store.get(...)` -- and opens its array children in the async color.
* [ThreadedAsyncGroup][abczarr.abc.async_group.ThreadedAsyncGroup] is the
  fallback for a group that is neither natively async nor path-based: it runs
  the sync group's navigation in a thread pool.
"""

__all__ = [
    "AsyncZarrGroup",
    "AsyncPathGroup",
    "ThreadedAsyncGroup",
]

# stdlib
import json
from abc import abstractmethod

# dependencies
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import constants
from abczarr._core import typing as tz
from abczarr._core.asyncutils import run_sync
from abczarr.api.config import ArrayConfig, ArrayOptions

# locals
from .async_array import AsyncZarrArray
from .async_node import AsyncZarrNode
from .capabilities import Support
from .group import PathGroup, ZarrGroup, _resolve_array_config
from .store import AsyncPathBasedStore


class AsyncZarrGroup(AsyncZarrNode):
    """The coroutine twin of a [ZarrGroup][abczarr.abc.group.ZarrGroup].

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
        """The synchronous [ZarrGroup][abczarr.abc.group.ZarrGroup] twin."""
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
        [ZarrGroup.create_array][abczarr.abc.group.ZarrGroup.create_array];
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
    """The async twin of [PathGroup][abczarr.abc.group.PathGroup].

    A backend with no group object of its own -- TensorStore opens arrays
    only -- gets a real async group here: it does its own I/O through an
    [AsyncStore][abczarr.abc.store.AsyncStore] over the group's location,
    listing and navigating members with `await store.list_dir(...)` /
    `await store.get(...)` rather than threading the sync group. Array
    children are opened in the async color -- a natively async backend's
    array comes back as its native async array. Its `"async"` capability is
    `Support.NATIVE`, because the listing runs on the async store.

    Creation still writes through the sync group in a thread pool -- writing
    metadata or building a backend array blocks, and is not the listing work
    this group makes asynchronous.
    """

    # the listing runs on the async store, not by threading the sync group
    _async_support = Support.NATIVE

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
    """An [AsyncZarrGroup][abczarr.abc.async_group.AsyncZarrGroup] that runs a
    sync group's navigation and creation in a bounded thread pool.

    The fallback for a group that is neither natively async nor path-based
    (a path-based backend gets the real
    [AsyncPathGroup][abczarr.abc.async_group.AsyncPathGroup] instead). Members
    it opens are handed back in the async color.
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
