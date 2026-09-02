"""The store surface: a key -> bytes map beneath every zarr node.

A store is the one seam shared by every driver, every metadata-version
reader, and every synthesized fallback. It is deliberately small -- five
primitives and a capability query:

* :meth:`~Store.get` -- read a key, ``None`` when it is missing;
* :meth:`~Store.set` -- write a key;
* :meth:`~Store.delete` -- remove a key;
* :meth:`~Store.exists` -- whether a key is present;
* :meth:`~Store.list_keys` -- iterate the keys under a prefix.

Everything richer -- listing one directory level, clearing the store,
sizing a key -- is *synthesized* from those five, so a new store implements
only the primitives. A backend whose native store can do one of the
synthesized operations better overrides it and advertises that reach through
:meth:`~Store.supports`.

:class:`PathStore` is the default: it turns each key into a path under a
store root and delegates to :mod:`bagof.paths`, so a local directory and
every fsspec / cloud scheme (``s3://``, ``gs://``, ``memory://``, ...) work
with no per-backend code. :class:`AsyncStore` and :class:`AsyncPathStore`
are the coroutine twins, over :class:`bagof.paths.AsyncPath`.
"""

__all__ = [
    "Store",
    "PathStore",
    "AsyncStore",
    "AsyncPathStore",
]

# stdlib
from abc import ABC, abstractmethod
from types import TracebackType

# dependencies
import typing_extensions as tx

# locals
from .capabilities import Support, SupportsCapabilities
from .path import AsyncStorePath, StorePath

#: The character that separates the segments of a zarr key (``"c/0/0"``).
_SEP = "/"


def _child(prefix: str, key: str) -> str:
    """The first key segment below *prefix*, e.g. ``("c/", "c/0/1") -> "0"``.

    An empty *prefix* returns the leading segment of *key*.
    """
    rest = key[len(prefix):] if prefix and key.startswith(prefix) else key
    rest = rest.lstrip(_SEP)
    return rest.split(_SEP, 1)[0]


class Store(SupportsCapabilities, ABC):
    """A key -> bytes map, addressed under a :class:`StorePath` root.

    Subclasses implement the five primitives; the rest is synthesized here.
    Keys are ``"/"``-separated relative strings (``"zarr.json"``,
    ``"c/0/0"``); values are :class:`bytes`. What a store provides natively
    versus by synthesis is declared in :attr:`_CAPABILITIES` and answered by
    :meth:`support` / :meth:`supports`.
    """

    def __init__(self, store_path: tx.Union[str, StorePath]) -> None:
        if isinstance(store_path, Store):
            store_path = store_path._store_path
        if isinstance(store_path, (str, bytes)):
            store_path = StorePath(store_path)
        self._store_path = store_path
        #: The backend object the store speaks through -- a bagof.paths
        #: ``Path`` for :class:`PathStore`, a native store for a driver. The
        #: escape hatch for anything the surface does not name.
        self._native: tx.Any = None

    # -- primitives --------------------------------------------------------

    @abstractmethod
    def get(self, key: str) -> tx.Optional[bytes]:
        """Read *key*, or ``None`` when it is not present."""
        ...

    @abstractmethod
    def set(self, key: str, value: bytes) -> None:
        """Write *value* at *key*, creating any parent structure."""
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
        """Iterate every key at or below *prefix*, ``"/"``-joined."""
        ...

    # -- synthesized from the primitives -----------------------------------

    def list_dir(self, prefix: str = "") -> tx.Iterator[str]:
        """Iterate the immediate child names one level below *prefix*.

        ``list_keys`` walks the whole subtree; this yields only the distinct
        first segments below *prefix*, the way ``listdir`` names one level.
        """
        seen = set()  # type: tx.Set[str]
        for key in self.list_keys(prefix):
            name = _child(prefix, key)
            if name and name not in seen:
                seen.add(name)
                yield name

    def getsize(self, key: str) -> tx.Optional[int]:
        """The size of *key* in bytes, or ``None`` when it is missing."""
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
        """The underlying backend object -- the escape hatch."""
        return self._native

    @property
    def store_path(self) -> StorePath:
        """The store's root :class:`StorePath`."""
        return self._store_path

    @property
    def read_only(self) -> bool:
        """Whether the store refuses writes."""
        return self._store_path.read_only

    @property
    def url(self) -> str:
        """The store root as a URL."""
        return self._store_path.as_uri()


class PathStore(Store):
    """The default store: keys are paths under a root, over :mod:`bagof.paths`.

    Every filesystem and cloud scheme bagof.paths understands works here with
    no extra code -- the root's scheme picks the driver, and credentials ride
    on the root path's ``storage_options``.
    """

    _CAPABILITIES = {
        "listing": Support.NATIVE,
        "writes": Support.NATIVE,
        "deletes": Support.NATIVE,
    }

    def __init__(self, store_path: tx.Union[str, StorePath]) -> None:
        super().__init__(store_path)
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

    def set(self, key: str, value: bytes) -> None:
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
    """The coroutine twin of :class:`Store`.

    The five primitives are coroutines and :meth:`list_keys` is an async
    iterator; the synthesized members mirror :class:`Store`. Location and
    capability queries never touch the backend, so they stay synchronous.
    """

    def __init__(self, store_path: tx.Union[str, AsyncStorePath]) -> None:
        if isinstance(store_path, AsyncStore):
            store_path = store_path._store_path
        if isinstance(store_path, (str, bytes)):
            store_path = AsyncStorePath(store_path)
        self._store_path = store_path
        self._native: tx.Any = None

    # -- primitives --------------------------------------------------------

    @abstractmethod
    async def get(self, key: str) -> tx.Optional[bytes]:
        """Read *key*, or ``None`` when it is not present."""
        ...

    @abstractmethod
    async def set(self, key: str, value: bytes) -> None:
        """Write *value* at *key*, creating any parent structure."""
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

    # -- synthesized -------------------------------------------------------

    async def list_dir(self, prefix: str = "") -> tx.AsyncIterator[str]:
        """Async-iterate the immediate child names one level below *prefix*."""
        seen = set()  # type: tx.Set[str]
        async for key in self.list_keys(prefix):
            name = _child(prefix, key)
            if name and name not in seen:
                seen.add(name)
                yield name

    async def getsize(self, key: str) -> tx.Optional[int]:
        """The size of *key* in bytes, or ``None`` when it is missing."""
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
        """The underlying backend object -- the escape hatch."""
        return self._native

    @property
    def store_path(self) -> AsyncStorePath:
        """The store's root :class:`AsyncStorePath`."""
        return self._store_path

    @property
    def read_only(self) -> bool:
        """Whether the store refuses writes."""
        return self._store_path.read_only

    @property
    def url(self) -> str:
        """The store root as a URL."""
        return self._store_path.as_uri()


class AsyncPathStore(AsyncStore):
    """The default async store, over :class:`bagof.paths.AsyncPath`.

    A synchronous bagof.paths driver runs in a worker thread; a natively
    async one (the fsspec cloud path) is awaited directly -- both behind the
    same coroutine surface, so nothing here is written twice.
    """

    _CAPABILITIES = {
        "listing": Support.NATIVE,
        "writes": Support.NATIVE,
        "deletes": Support.NATIVE,
        "async": Support.NATIVE,
    }

    def __init__(self, store_path: tx.Union[str, AsyncStorePath]) -> None:
        super().__init__(store_path)
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

    async def set(self, key: str, value: bytes) -> None:
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
