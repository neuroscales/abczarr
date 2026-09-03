"""The coroutine twins of the node surface.

Every sync node -- a [ZarrArray][abczarr.abc.array.ZarrArray] or a
[ZarrGroup][abczarr.abc.group.ZarrGroup] -- has an async twin with the same
shape, reached with
[as_async][abczarr.abc.array.ZarrArray.as_async]. The twin's I/O is
coroutines; a backend that is natively async (tensorstore, zarr-python)
awaits its own futures, and one that is not runs its blocking ops in a
bounded thread pool.

The array twin reads and writes through **methods**, not `[]`:
`await array.getitem(index)` and `await array.setitem(index, value)`. An
assignment expression cannot be awaited, so `__setitem__` cannot be a
coroutine -- the same choice zarr-python made. The read-only properties
(`shape`, `dtype`, `chunks`, ...) never block, so they stay synchronous on
both colors.

Convert between colors without re-opening -- both share one backend handle:

!!! example
    ```python
    async_array = sync_array.as_async()
    sync_array = async_array.as_sync()
    ```
"""

__all__ = [
    "AsyncZarrNode",
    "AsyncZarrArray",
    "AsyncZarrGroup",
    "ThreadedAsyncArray",
    "ThreadedAsyncGroup",
]

# stdlib
import os
from abc import ABC, abstractmethod

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.asyncutils import run_sync
from abczarr._core.attributes import Attributes
from abczarr.api.config import ArrayConfig, ArrayOptions
from abczarr.metadata.base import NodeMetadata

# locals
from .array import ZarrArray
from .capabilities import Support, SupportsCapabilities
from .group import ZarrGroup, _resolve_array_config
from .node import ZarrNode


class AsyncZarrNode(SupportsCapabilities, ABC):
    """The coroutine twin of a [ZarrNode][abczarr.abc.node.ZarrNode].

    An async node wraps its sync twin and shares the same backend handle,
    metadata, attributes and version. Only the I/O is different: it is
    coroutines. Use
    [capability][abczarr.abc.capabilities.SupportsCapabilities.capability]
    or [supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
    to check whether the async surface is native to the backend or
    synthesized from its sync one in a thread pool.
    """

    #: How the async surface is provided: `Support.NATIVE` when the backend
    #: has its own coroutine I/O, `Support.SYNTHESIZED` when abczarr runs the
    #: sync ops in a thread pool. A native twin overrides this.
    _async_support = Support.SYNTHESIZED  # type: tx.ClassVar[Support]

    def __init__(self, sync: ZarrNode) -> None:
        self._sync = sync

    @property
    def store_path(self) -> os.PathLike:
        """The path to this node's location in its store."""
        return self._sync.store_path

    @property
    def native(self) -> tx.Any:
        """The underlying backend object, or `None` -- the escape hatch for
        anything the uniform surface does not name."""
        return self._sync.native

    @property
    def metadata(self) -> NodeMetadata:
        """This node's Zarr metadata."""
        return self._sync.metadata

    @property
    def attrs(self) -> Attributes:
        """This node's user attributes, as a live, write-through mapping.

        Attributes stay synchronous on the async twin: the write-through
        mapping goes straight to the metadata file rather than through the
        node's data path, so there is nothing to await. An async, batched
        attribute surface is a separate concern.
        """
        return self._sync.attrs

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        """The Zarr format version this node was written with."""
        return self._sync.zarr_version

    def capability(self, name: str) -> Support:
        """How this async node provides the capability *name*.

        The answer for `"async"` is this twin's own -- native or
        synthesized -- and every other name is answered by the sync twin,
        so the two colors report the same backend features.
        """
        if name == "async":
            return self._async_support
        return self._sync.capability(name)

    @abstractmethod
    def as_sync(self) -> ZarrNode:
        """The synchronous twin over the same backend handle."""
        ...


class AsyncZarrArray(AsyncZarrNode, ABC):
    """The coroutine twin of a [ZarrArray][abczarr.abc.array.ZarrArray].

    Read and write it with `await`:

    !!! example
        ```python
        block = await array.getitem((slice(0, 64), slice(0, 64)))
        await array.setitem((slice(0, 64), slice(0, 64)), block * 2)
        ```
    """

    def __init__(self, sync: ZarrArray) -> None:
        super().__init__(sync)

    def as_sync(self) -> ZarrArray:
        """The synchronous [ZarrArray][abczarr.abc.array.ZarrArray] twin."""
        return tx.cast(ZarrArray, self._sync)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        return self.as_sync().ndim

    @property
    def shape(self) -> tz.Shape:
        """The shape of the array."""
        return self.as_sync().shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the array."""
        return self.as_sync().dtype

    @property
    def chunks(self) -> tz.Shape:
        """The chunk shape of the array."""
        return self.as_sync().chunks

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        """The shard shape of the array, or `None` if it is not sharded."""
        return self.as_sync().shards

    @abstractmethod
    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        """Read data from the array at *index* (a NumPy-style selection)."""
        ...

    @abstractmethod
    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        """Write *value* at *index* (a NumPy-style selection)."""
        ...


class AsyncZarrGroup(AsyncZarrNode, ABC):
    """The coroutine twin of a [ZarrGroup][abczarr.abc.group.ZarrGroup].

    Reach a member with `await group.getitem(name)`, iterate the member
    names with `async for`, and create children with `await`:

    !!! example
        ```python
        child = await group.getitem("images")
        async for name in group:
            ...
        array = await group.create_array("labels", (8, 8), "uint8")
        ```
    """

    def __init__(self, sync: ZarrGroup) -> None:
        super().__init__(sync)

    def as_sync(self) -> ZarrGroup:
        """The synchronous [ZarrGroup][abczarr.abc.group.ZarrGroup] twin."""
        return tx.cast(ZarrGroup, self._sync)

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


# --------------------------------------------------------------------------
#   Thread-synthesized default twins
# --------------------------------------------------------------------------
#
# The fallback for any backend with no native coroutine surface: run the sync
# node's blocking ops in the bounded thread pool. Honest about it -- the async
# capability reports SYNTHESIZED. A driver whose backend is natively async
# overrides ``as_async`` on its sync node to return a NATIVE twin instead.


class ThreadedAsyncArray(AsyncZarrArray):
    """An [AsyncZarrArray][abczarr.abc.asyncnode.AsyncZarrArray] that runs a
    sync array's reads and writes in a bounded thread pool.

    The default async array for a backend that has no coroutine surface of
    its own. It reports `"async"` as `Support.SYNTHESIZED`.
    """

    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        return await run_sync(self.as_sync().__getitem__, index)

    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        await run_sync(self.as_sync().__setitem__, index, value)


class ThreadedAsyncGroup(AsyncZarrGroup):
    """An [AsyncZarrGroup][abczarr.abc.asyncnode.AsyncZarrGroup] that runs a
    sync group's navigation and creation in a bounded thread pool.

    Members it opens are handed back in the async color too: a child array
    from a natively-async backend comes back as that backend's native async
    array, so navigating a synthesized group still reaches native I/O.
    """

    async def getitem(self, key: str) -> AsyncZarrNode:
        child = await run_sync(self.as_sync().__getitem__, key)
        return child.as_async()

    async def keys(self) -> tx.AsyncIterator[str]:
        names = await run_sync(lambda: list(self.as_sync().keys()))
        for name in names:
            yield name

    async def _create_array(
        self, name: str, config: ArrayConfig
    ) -> AsyncZarrArray:
        child = await run_sync(self.as_sync()._create_array, name, config)
        return tx.cast(AsyncZarrArray, child.as_async())

    async def create_group(
        self, name: str, overwrite: bool = False
    ) -> AsyncZarrGroup:
        child = await run_sync(self.as_sync().create_group, name, overwrite)
        return tx.cast(AsyncZarrGroup, child.as_async())
