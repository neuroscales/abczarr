"""The store surface: a key to bytes map beneath every Zarr node.

A store is what every Zarr array and group reads and writes
through. Keys are `"/"`-joined relative strings (`"zarr.json"`,
`"c/0/0"`); values are bytes. The surface is deliberately small --
five primitives:

* [get][abczarr.abc.store.Store.get] -- read a key, `None` when it
  is missing.
* [set][abczarr.abc.store.Store.set] -- write a key.
* [delete][abczarr.abc.store.Store.delete] -- remove a key.
* [exists][abczarr.abc.store.Store.exists] -- whether a key is
  present.
* [list_keys][abczarr.abc.store.Store.list_keys] -- iterate the keys
  under a prefix.

Everything richer -- listing one directory level, clearing the
store, sizing a key, batching writes into a transaction -- is built
from those five for free, so a new store only has to implement the
primitives. A backend that can do better than the built-in version
of one of those operations overrides it and advertises the faster
path through
[supports][abczarr.abc.capabilities.SupportsCapabilities.supports].

[PathBasedStore][abczarr.abc.store.PathBasedStore] is the default store: it
turns each key into a path under a root and delegates to
`bagof.paths`, so a local directory and every fsspec or cloud scheme
(`s3://`, `gs://`, `memory://`, ...) work with no extra code.
[AsyncStore][abczarr.abc.store.AsyncStore] and
[AsyncPathBasedStore][abczarr.abc.store.AsyncPathBasedStore] are the coroutine
twins, built over `bagof.paths.AsyncPath`.
"""

__all__ = [
    "StorePath",
    "AsyncStorePath",
    "Store",
    "PathBasedStore",
    "AsyncStore",
    "AsyncPathBasedStore",
]

# stdlib
import asyncio
import os
from abc import ABC, abstractmethod
from types import TracebackType

# dependencies
import typing_extensions as tx
from bagof.paths import AsyncPath, Path

from abczarr.errors import UnsupportedZarrOperation

# locals
from .capabilities import Support, SupportsCapabilities

if tx.TYPE_CHECKING:
    from .transactions import AsyncTransaction, Transaction

#: The character that separates the segments of a key (``"c/0/0"``).
_SEP = "/"

#: What a store accepts as a value: bytes or any buffer-protocol
#: object, so a native wrapper can pass its backend's buffer through
#: without a copy.
_BytesLike = tx.Union[bytes, bytearray, memoryview]

#: Capabilities every store can build from the primitives, so
#: ``support`` reports at least ``Support.SYNTHESIZED`` for them
#: unless a store declares a better answer.
_SYNTHESIZED_FLOOR = {
    "partial_read": Support.SYNTHESIZED,
    "transactions": Support.SYNTHESIZED,
}


# -- store paths -----------------------------------------------------------
# A store is addressed under a StorePath: a normal bagof.paths.Path -- one
# API over local and cloud paths -- with one extra bit of state, read_only.
# bagof.paths already dispatches on the URL scheme (s3://, gs://, memory://,
# a local path, ...), so StorePath("s3://bucket/key") picks the right backend
# on its own, no per-protocol subclass needed, and the underlying backend
# object is always reachable through .wrapped. read_only rides along onto
# every path derived from it (.parent, /, iterdir(), ...), so a child of a
# read-only store is read-only too.


class StorePath(Path):
    """A Zarr store location: a path, plus a `read_only` flag.

    Accepts everything `bagof.paths.Path` accepts -- a local path or
    a URL for any scheme it supports -- plus a `read_only` keyword.
    """

    def __init__(
        self, *args: tx.Any, read_only: bool = False, **kwargs: tx.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.read_only = read_only


class AsyncStorePath(AsyncPath):
    """The async counterpart of
    [StorePath][abczarr.abc.store.StorePath]."""

    # Run StorePath's synchronous surface in the worker thread, so any
    # sync override here is honoured on the async side too.
    _sync_type: tx.ClassVar[tx.Type[Path]] = StorePath

    def __init__(
        self, *args: tx.Any, read_only: bool = False, **kwargs: tx.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.read_only = read_only


def _child(prefix: str, key: str) -> str:
    """The first key segment below *prefix*, e.g. ``"c/0/1"`` under
    ``"c/"`` is ``"0"``. An empty *prefix* returns the leading
    segment of *key*.
    """
    rest = key[len(prefix):] if prefix and key.startswith(prefix) else key
    rest = rest.lstrip(_SEP)
    return rest.split(_SEP, 1)[0]


def _as_store_path_str(store_path: tx.Any) -> tx.Any:
    """A path object becomes its path string, so a store wraps it in a
    [StorePath][abczarr.abc.store.StorePath] (bagof.paths, URL-aware) rather
    than using it raw. A str, a StorePath, or ``None`` passes through.
    """
    if isinstance(store_path, os.PathLike) and not isinstance(
        store_path, (StorePath, AsyncStorePath)
    ):
        return os.fspath(store_path)
    return store_path


class Store(SupportsCapabilities, ABC):
    """A key to bytes map, addressed under a
    [StorePath][abczarr.abc.store.StorePath] root.

    A subclass implements the five primitives -- `get`, `set`,
    `delete`, `exists`, `list_keys` -- and gets everything else for
    free. Keys are `"/"`-separated relative strings (`"zarr.json"`,
    `"c/0/0"`); values are `bytes`. Use
    [capability][abczarr.abc.capabilities.SupportsCapabilities.capability]
    or
    [supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
    to check whether a given operation is native to the backend or
    built from the primitives.

    A `Store` is also a context manager, so it can be used with
    `with` to make sure its resources are released.

    !!! example
        ```pycon
        >>> store = PathBasedStore("/tmp/demo-store")
        >>> store.set("a", b"1")
        >>> store.get("a")
        b'1'
        >>> store.get("missing") is None
        True
        >>> list(store.list_keys())
        ['a']
        ```
    """

    def __init__(
        self,
        store_path: tx.Optional[tx.Union[str, os.PathLike, StorePath]] = None,
    ) -> None:
        if isinstance(store_path, Store):
            store_path = store_path._store_path
        store_path = _as_store_path_str(store_path)
        if isinstance(store_path, (str, bytes)):
            store_path = StorePath(store_path)
        #: The store's root, or ``None`` for a store that has no path -- a
        #: native memory store, a session store, an in-process backend.
        self._store_path = store_path
        #: The backend object the store speaks through -- a bagof.paths
        #: `Path` for `PathBasedStore`, a native store for a driver. The
        #: escape hatch for anything the surface does not name.
        self._native: tx.Any = None

    # -- primitives --------------------------------------------------------

    @abstractmethod
    def get(self, key: str) -> tx.Optional[bytes]:
        """Read *key*.

        Parameters
        ----------
        key : str
            The key to read.

        Returns
        -------
        bytes or None
            The stored value, or `None` when *key* is not present.
            The result is bytes-like; wrap it in `bytes(...)` for
            exactly `bytes`.
        """
        ...

    @abstractmethod
    def set(self, key: str, value: _BytesLike) -> None:
        """Write *value* at *key*, creating any parent structure.

        Parameters
        ----------
        key : str
            The key to write.
        value : bytes-like
            The value to store -- `bytes` or any buffer object such
            as `bytearray` or `memoryview`.
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove *key*. Removing a missing key is not an error."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether *key* is present."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> tx.Iterator[str]:
        """Iterate every key at or below *prefix*, `"/"`-joined."""
        ...

    # -- capability query, with the synthesized floor ----------------------

    def capability(self, capability: str) -> Support:
        """How this store provides *capability*.

        A store declares what it does natively; for the operations
        this base class can always build from the primitives (such
        as a byte-range read), the answer is at least
        `Support.SYNTHESIZED` even when the store declares nothing.
        """
        declared = self._CAPABILITIES.get(capability)
        if declared is not None:
            return declared
        return _SYNTHESIZED_FLOOR.get(capability, Support.NONE)

    # -- synthesized from the primitives -----------------------------------

    def get_many(
        self, keys: tx.Iterable[str]
    ) -> tx.Dict[str, tx.Optional[bytes]]:
        """Read several keys at once.

        Returns
        -------
        dict
            Maps each key to its value, or to `None` when missing.
            A store with a real batched read overrides this; the
            default reads one key at a time.
        """
        return {key: self.get(key) for key in keys}

    def get_partial(
        self, key: str, start: int, length: tx.Optional[int] = None
    ) -> tx.Optional[bytes]:
        """Read a byte range of *key*.

        Parameters
        ----------
        key : str
            The key to read.
        start : int
            The offset, in bytes, to start reading from.
        length : int, optional
            How many bytes to read. Reads to the end of the value
            when omitted.

        Returns
        -------
        bytes or None
            The requested range, or `None` when *key* is missing.
            The default reads the whole value and slices it; a store
            with a native byte-range read overrides this and
            declares `"partial_read"` as `Support.NATIVE`.
        """
        value = self.get(key)
        if value is None:
            return None
        end = None if length is None else start + length
        return bytes(value[start:end])

    def set_if_not_exists(self, key: str, value: _BytesLike) -> bool:
        """Write *value* only when *key* is absent.

        Returns
        -------
        bool
            Whether the write happened.

        Built from `exists` then `set`, so it is racy under
        concurrent writers -- a store with an atomic put overrides
        it.
        """
        if self.exists(key):
            return False
        self.set(key, value)
        return True

    def delete_prefix(self, prefix: str = "") -> None:
        """Delete every key at or below *prefix*."""
        for key in list(self.list_keys(prefix)):
            self.delete(key)

    def transaction(self, *, atomic: bool = True) -> "Transaction":
        """Open a transaction: a store view whose writes commit
        together.

        A store whose backend has real transactions returns a native
        one. Every other store returns a
        [BufferedTransaction][abczarr.abc.transactions.BufferedTransaction],
        which can only honor `atomic=False` -- asking for
        `atomic=True` on such a store raises rather than pretend a
        partial write is atomic.

        Parameters
        ----------
        atomic : bool, optional
            Whether the commit must be all-or-nothing.

        Returns
        -------
        Transaction
        """
        if self.capability("transactions") is Support.NATIVE:
            return self._native_transaction(atomic=atomic)
        if atomic:
            raise UnsupportedZarrOperation(
                "atomic transaction", driver=type(self).__name__
            )
        from .transactions import BufferedTransaction

        return BufferedTransaction(self)

    def _native_transaction(self, *, atomic: bool) -> "Transaction":
        """A driver with native transactions overrides this."""
        raise UnsupportedZarrOperation(
            "transaction", driver=type(self).__name__
        )

    def list_dir(self, prefix: str = "") -> tx.Iterator[str]:
        """Iterate the immediate child names one level below *prefix*.

        `list_keys` walks the whole subtree; this yields only the
        distinct first segments below *prefix*, the way `listdir`
        names one level.
        """
        seen = set()  # type: tx.Set[str]
        for key in self.list_keys(prefix):
            name = _child(prefix, key)
            if name and name not in seen:
                seen.add(name)
                yield name

    def getsize(self, key: str) -> tx.Optional[int]:
        """The size of *key* in bytes, or `None` when it is missing."""
        value = self.get(key)
        return None if value is None else len(value)

    def clear(self) -> None:
        """Remove every key in the store."""
        for key in list(self.list_keys()):
            self.delete(key)

    def close(self) -> None:  # noqa: B027 -- an overridable no-op, not abstract
        """Release any resources the store holds. A no-op by default."""

    # -- dunder conveniences ----------------------------------------------

    def __contains__(self, key: str) -> bool:
        return self.exists(key)

    def __iter__(self) -> tx.Iterator[str]:
        return self.list_keys()

    def __enter__(self) -> tx.Self:
        return self

    def __exit__(
        self,
        exc_type: tx.Optional[tx.Type[BaseException]],
        exc_value: tx.Optional[BaseException],
        traceback: tx.Optional[TracebackType],
    ) -> None:
        self.close()

    # -- location ----------------------------------------------------------

    @property
    def native(self) -> tx.Any:
        """The underlying backend object -- the escape hatch for
        anything the uniform surface does not name."""
        return self._native

    @property
    def store_path(self) -> tx.Optional[StorePath]:
        """The store's root, or `None` when it has no path.

        Returns
        -------
        [StorePath][abczarr.abc.store.StorePath] or None
        """
        return self._store_path

    @property
    def read_only(self) -> bool:
        """Whether the store refuses writes."""
        return self._store_path is not None and self._store_path.read_only

    @property
    def url(self) -> tx.Optional[str]:
        """The store root as a URL, or `None` when it has no path."""
        if self._store_path is None:
            return None
        return self._store_path.as_uri()


class PathBasedStore(Store):
    """The default store: keys are paths under a root.

    A `PathBasedStore` turns each key into a path below its root and
    delegates to `bagof.paths`. Every filesystem and cloud scheme
    bagof.paths understands works here with no extra code -- the
    root's scheme picks the driver, and credentials ride on the root
    path's `storage_options`.

    !!! example
        ```pycon
        >>> store = PathBasedStore("/tmp/demo-store")
        >>> store.set("zarr.json", b'{"zarr_format": 3}')
        >>> store.get("zarr.json")
        b'{"zarr_format": 3}'
        ```

    Parameters
    ----------
    store_path : str or [StorePath][abczarr.abc.store.StorePath]
        The root all keys are resolved under -- a local path or a
        URL such as `"s3://bucket/prefix"`.
    """

    _CAPABILITIES = {
        "listing": Support.NATIVE,
        "writes": Support.NATIVE,
        "deletes": Support.NATIVE,
    }

    def __init__(self, store_path: tx.Union[str, StorePath]) -> None:
        super().__init__(store_path)
        if self._store_path is None:
            raise ValueError(
                "a PathBasedStore needs a location; pass a path or URL"
            )
        # the bagof.paths root every key is resolved against
        self._native = self._store_path

    def _key_path(self, key: str) -> StorePath:
        return self._store_path.joinpath(*key.split(_SEP)) if key else (
            self._store_path
        )

    def get(self, key: str) -> tx.Optional[bytes]:
        try:
            return self._key_path(key).read_bytes()
        except FileNotFoundError:
            return None

    def set(self, key: str, value: _BytesLike) -> None:
        target = self._key_path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)

    def delete(self, key: str) -> None:
        self._key_path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._key_path(key).exists()

    def list_keys(self, prefix: str = "") -> tx.Iterator[str]:
        root = self._key_path(prefix)
        if not root.exists():
            return
        base = self._store_path
        for dirpath, _dirnames, filenames in root.walk():
            for name in filenames:
                yield (dirpath / name).relative_to(base).as_posix()


class AsyncStore(SupportsCapabilities, ABC):
    """The coroutine twin of [Store][abczarr.abc.store.Store].

    The five primitives are coroutines and
    [list_keys][abczarr.abc.store.AsyncStore.list_keys] is an async
    iterator; everything else mirrors `Store`. Location and
    capability queries never touch the backend, so they stay
    synchronous.
    """

    def __init__(
        self,
        store_path: tx.Optional[
            tx.Union[str, os.PathLike, AsyncStorePath]
        ] = None,
    ) -> None:
        if isinstance(store_path, AsyncStore):
            store_path = store_path._store_path
        store_path = _as_store_path_str(store_path)
        if isinstance(store_path, (str, bytes)):
            store_path = AsyncStorePath(store_path)
        self._store_path = store_path
        self._native: tx.Any = None

    # -- primitives --------------------------------------------------------

    @abstractmethod
    async def get(self, key: str) -> tx.Optional[bytes]:
        """Read *key*, or `None` when it is not present (bytes-like)."""
        ...

    @abstractmethod
    async def set(self, key: str, value: _BytesLike) -> None:
        """Write *value* (bytes or a buffer) at *key*, making parents."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove *key*. Removing a missing key is not an error."""
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Whether *key* is present."""
        ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> tx.AsyncIterator[str]:
        """Async-iterate every key at or below *prefix*."""
        ...

    # -- capability query, with the synthesized floor ----------------------

    def capability(self, capability: str) -> Support:
        """How this store provides *capability*.

        See [Store.capability][abczarr.abc.store.Store.capability].
        """
        declared = self._CAPABILITIES.get(capability)
        if declared is not None:
            return declared
        return _SYNTHESIZED_FLOOR.get(capability, Support.NONE)

    # -- synthesized -------------------------------------------------------

    async def get_many(
        self, keys: tx.Iterable[str]
    ) -> tx.Dict[str, tx.Optional[bytes]]:
        """Read several keys concurrently.

        Returns a dict mapping each key to its value, or to `None`
        when missing.
        """
        keys = list(keys)
        values = await asyncio.gather(*(self.get(key) for key in keys))
        return dict(zip(keys, values))

    async def get_partial(
        self, key: str, start: int, length: tx.Optional[int] = None
    ) -> tx.Optional[bytes]:
        """Read *length* bytes of *key* from *start*.

        Reads to the end of the value when *length* is `None`.
        `None` when *key* is missing. The default reads the whole
        value and slices it; a store with a native byte-range read
        overrides this and declares `"partial_read"` as
        `Support.NATIVE`.
        """
        value = await self.get(key)
        if value is None:
            return None
        end = None if length is None else start + length
        return bytes(value[start:end])

    async def set_if_not_exists(self, key: str, value: _BytesLike) -> bool:
        """Write *value* only when *key* is absent.

        Returns whether the write happened. Built from `exists` then
        `set`, so it is racy under concurrent writers.
        """
        if await self.exists(key):
            return False
        await self.set(key, value)
        return True

    async def delete_prefix(self, prefix: str = "") -> None:
        """Delete every key at or below *prefix*."""
        keys = [key async for key in self.list_keys(prefix)]
        for key in keys:
            await self.delete(key)

    def transaction(self, *, atomic: bool = True) -> "AsyncTransaction":
        """Open a transaction.

        See [Store.transaction][abczarr.abc.store.Store.transaction].
        """
        if self.capability("transactions") is Support.NATIVE:
            return self._native_transaction(atomic=atomic)
        if atomic:
            raise UnsupportedZarrOperation(
                "atomic transaction", driver=type(self).__name__
            )
        from .transactions import AsyncBufferedTransaction

        return AsyncBufferedTransaction(self)

    def _native_transaction(self, *, atomic: bool) -> "AsyncTransaction":
        """A driver with native transactions overrides this."""
        raise UnsupportedZarrOperation(
            "transaction", driver=type(self).__name__
        )

    async def list_dir(self, prefix: str = "") -> tx.AsyncIterator[str]:
        """Async-iterate the child names one level below *prefix*."""
        seen = set()  # type: tx.Set[str]
        async for key in self.list_keys(prefix):
            name = _child(prefix, key)
            if name and name not in seen:
                seen.add(name)
                yield name

    async def getsize(self, key: str) -> tx.Optional[int]:
        """The size of *key* in bytes, or `None` when it is missing."""
        value = await self.get(key)
        return None if value is None else len(value)

    async def clear(self) -> None:
        """Remove every key in the store."""
        keys = [key async for key in self.list_keys()]
        for key in keys:
            await self.delete(key)

    async def close(self) -> None:  # noqa: B027 -- overridable no-op
        """Release any resources the store holds. A no-op by default."""

    # -- dunder conveniences ----------------------------------------------

    async def __aenter__(self) -> tx.Self:
        return self

    async def __aexit__(
        self,
        exc_type: tx.Optional[tx.Type[BaseException]],
        exc_value: tx.Optional[BaseException],
        traceback: tx.Optional[TracebackType],
    ) -> None:
        await self.close()

    # -- location ----------------------------------------------------------

    @property
    def native(self) -> tx.Any:
        """The underlying backend object -- the escape hatch for
        anything the uniform surface does not name."""
        return self._native

    @property
    def store_path(self) -> tx.Optional[AsyncStorePath]:
        """The store's root, or `None` when it has no path.

        Returns
        -------
        [AsyncStorePath][abczarr.abc.store.AsyncStorePath] or None
        """
        return self._store_path

    @property
    def read_only(self) -> bool:
        """Whether the store refuses writes."""
        return self._store_path is not None and self._store_path.read_only

    @property
    def url(self) -> tx.Optional[str]:
        """The store root as a URL, or `None` when it has no path."""
        if self._store_path is None:
            return None
        return self._store_path.as_uri()


class AsyncPathBasedStore(AsyncStore):
    """The default async store, built over `bagof.paths.AsyncPath`.

    A local directory and every fsspec or cloud scheme work here with
    no extra code. A synchronous backend runs in a worker thread; a
    natively async one (such as an fsspec cloud backend) is awaited
    directly -- both present the same coroutine surface.
    """

    _CAPABILITIES = {
        "listing": Support.NATIVE,
        "writes": Support.NATIVE,
        "deletes": Support.NATIVE,
        "async": Support.NATIVE,
    }

    def __init__(self, store_path: tx.Union[str, AsyncStorePath]) -> None:
        super().__init__(store_path)
        if self._store_path is None:
            raise ValueError(
                "an AsyncPathBasedStore needs a location; pass a path or URL"
            )
        self._native = self._store_path

    def _key_path(self, key: str) -> AsyncStorePath:
        return self._store_path.joinpath(*key.split(_SEP)) if key else (
            self._store_path
        )

    async def get(self, key: str) -> tx.Optional[bytes]:
        try:
            return await self._key_path(key).read_bytes()
        except FileNotFoundError:
            return None

    async def set(self, key: str, value: _BytesLike) -> None:
        target = self._key_path(key)
        await target.parent.mkdir(parents=True, exist_ok=True)
        await target.write_bytes(value)

    async def delete(self, key: str) -> None:
        await self._key_path(key).unlink(missing_ok=True)

    async def exists(self, key: str) -> bool:
        return await self._key_path(key).exists()

    async def list_keys(self, prefix: str = "") -> tx.AsyncIterator[str]:
        root = self._key_path(prefix)
        if not await root.exists():
            return
        base = self._store_path
        async for dirpath, _dirnames, filenames in root.walk():
            for name in filenames:
                yield (dirpath / name).relative_to(base).as_posix()
