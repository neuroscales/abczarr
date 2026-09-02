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
import asyncio
from abc import ABC, abstractmethod
from types import TracebackType

# dependencies
import typing_extensions as tx

# locals
from .capabilities import Support, SupportsCapabilities
from .errors import UnsupportedZarrOperation
from .path import AsyncStorePath, StorePath

if tx.TYPE_CHECKING:
    from .transactions import AsyncTransaction, Transaction

#: The character that separates the segments of a zarr key (``"c/0/0"``).
_SEP = "/"

#: What a store accepts as a value: bytes or any object that exposes the
#: buffer protocol, so a native wrapper can pass its backend's buffer
#: through without a copy.
_BytesLike = tx.Union[bytes, bytearray, memoryview]

#: Capabilities the base store can always synthesize from the primitives, so
#: ``support`` reports at least :attr:`Support.SYNTHESIZED` for them unless a
#: store declares a better answer. ``transactions`` is synthesizable
#: (a buffered, non-atomic batch); ``atomic_transactions`` is *not*, so it is
#: absent here and only a native store may declare it.
_SYNTHESIZED_FLOOR = {
    "partial_read": Support.SYNTHESIZED,
    "transactions": Support.SYNTHESIZED,
}


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

    def __init__(
        self, store_path: tx.Optional[tx.Union[str, StorePath]] = None
    ) -> None:
        if isinstance(store_path, Store):
            store_path = store_path._store_path
        if isinstance(store_path, (str, bytes)):
            store_path = StorePath(store_path)
        #: The store's root, or ``None`` for a store that has no path -- a
        #: native memory store, a session store, an in-process backend.
        self._store_path = store_path
        #: The backend object the store speaks through -- a bagof.paths
        #: ``Path`` for :class:`PathStore`, a native store for a driver. The
        #: escape hatch for anything the surface does not name.
        self._native: tx.Any = None

    # -- primitives --------------------------------------------------------

    @abstractmethod
    def get(self, key: str) -> tx.Optional[bytes]:
        """Read *key*, or ``None`` when it is not present.

        The result is bytes-like; a caller that needs exactly :class:`bytes`
        should wrap it in ``bytes(...)``.
        """
        ...

    @abstractmethod
    def set(self, key: str, value: _BytesLike) -> None:
        """Write *value* at *key*, creating any parent structure.

        *value* is bytes or any buffer (``bytearray``, ``memoryview``), so a
        native store can pass its backend's buffer through without a copy.
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
        """Iterate every key at or below *prefix*, ``"/"``-joined."""
        ...

    # -- capability query, with the synthesized floor ----------------------

    def support(self, capability: str) -> Support:
        """How this store provides *capability*.

        A store advertises what it does natively in :attr:`_CAPABILITIES`;
        for the members the base can always build from the primitives (a
        byte-range read), the answer is at least
        :attr:`Support.SYNTHESIZED`.
        """
        declared = self._CAPABILITIES.get(capability)
        if declared is not None:
            return declared
        return _SYNTHESIZED_FLOOR.get(capability, Support.NONE)

    # -- synthesized from the primitives -----------------------------------

    def get_many(
        self, keys: tx.Iterable[str]
    ) -> tx.Dict[str, tx.Optional[bytes]]:
        """Read several keys at once, ``{key: value-or-None}``.

        A store whose backend has a real batched read overrides this to use
        it; the default reads one key at a time.
        """
        return {key: self.get(key) for key in keys}

    def get_partial(
        self, key: str, start: int, length: tx.Optional[int] = None
    ) -> tx.Optional[bytes]:
        """Read *length* bytes of *key* from *start* (to the end if ``None``).

        ``None`` when the key is missing. The default reads the whole value
        and slices; a store with native byte-range reads overrides this and
        declares ``partial_read`` :attr:`Support.NATIVE`.
        """
        value = self.get(key)
        if value is None:
            return None
        end = None if length is None else start + length
        return bytes(value[start:end])

    def set_if_not_exists(self, key: str, value: _BytesLike) -> bool:
        """Write *value* only when *key* is absent; return whether it wrote.

        Synthesized as :meth:`exists` then :meth:`set`, so it is racy under
        concurrent writers -- a store with an atomic put overrides it.
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
        """Open a transaction -- a store view whose writes commit together.

        A store whose backend has real transactions returns a native one. Any
        other store returns a buffered, non-atomic transaction, so it can only
        honour ``atomic=False``; ``atomic=True`` on such a store raises rather
        than pretend a torn write is atomic.
        """
        if self.support("transactions") is Support.NATIVE:
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
    def store_path(self) -> tx.Optional[StorePath]:
        """The store's root :class:`StorePath`, or ``None`` when unset."""
        return self._store_path

    @property
    def read_only(self) -> bool:
        """Whether the store refuses writes."""
        return self._store_path is not None and self._store_path.read_only

    @property
    def url(self) -> tx.Optional[str]:
        """The store root as a URL, or ``None`` when it has no path."""
        if self._store_path is None:
            return None
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
        if self._store_path is None:
            raise ValueError(
                "a PathStore needs a location; pass a path or URL"
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
    """The coroutine twin of :class:`Store`.

    The five primitives are coroutines and :meth:`list_keys` is an async
    iterator; the synthesized members mirror :class:`Store`. Location and
    capability queries never touch the backend, so they stay synchronous.
    """

    def __init__(
        self, store_path: tx.Optional[tx.Union[str, AsyncStorePath]] = None
    ) -> None:
        if isinstance(store_path, AsyncStore):
            store_path = store_path._store_path
        if isinstance(store_path, (str, bytes)):
            store_path = AsyncStorePath(store_path)
        self._store_path = store_path
        self._native: tx.Any = None

    # -- primitives --------------------------------------------------------

    @abstractmethod
    async def get(self, key: str) -> tx.Optional[bytes]:
        """Read *key*, or ``None`` when it is not present (bytes-like)."""
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

    def support(self, capability: str) -> Support:
        """How this store provides *capability* (see :meth:`Store.support`)."""
        declared = self._CAPABILITIES.get(capability)
        if declared is not None:
            return declared
        return _SYNTHESIZED_FLOOR.get(capability, Support.NONE)

    # -- synthesized -------------------------------------------------------

    async def get_many(
        self, keys: tx.Iterable[str]
    ) -> tx.Dict[str, tx.Optional[bytes]]:
        """Read several keys concurrently, ``{key: value-or-None}``."""
        keys = list(keys)
        values = await asyncio.gather(*(self.get(key) for key in keys))
        return dict(zip(keys, values))

    async def get_partial(
        self, key: str, start: int, length: tx.Optional[int] = None
    ) -> tx.Optional[bytes]:
        """Read *length* bytes of *key* from *start* (to the end if ``None``).

        The default reads the whole value and slices; a store with native
        byte-range reads overrides this and declares ``partial_read``
        :attr:`Support.NATIVE`.
        """
        value = await self.get(key)
        if value is None:
            return None
        end = None if length is None else start + length
        return bytes(value[start:end])

    async def set_if_not_exists(self, key: str, value: _BytesLike) -> bool:
        """Write *value* only when *key* is absent; return whether it wrote.

        Synthesized, so racy under concurrent writers.
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
        """Open a transaction (see :meth:`Store.transaction`)."""
        if self.support("transactions") is Support.NATIVE:
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
    def store_path(self) -> tx.Optional[AsyncStorePath]:
        """The store's root :class:`AsyncStorePath`, or ``None``."""
        return self._store_path

    @property
    def read_only(self) -> bool:
        """Whether the store refuses writes."""
        return self._store_path is not None and self._store_path.read_only

    @property
    def url(self) -> tx.Optional[str]:
        """The store root as a URL, or ``None`` when it has no path."""
        if self._store_path is None:
            return None
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
        if self._store_path is None:
            raise ValueError(
                "an AsyncPathStore needs a location; pass a path or URL"
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
