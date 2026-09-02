"""Transactions over a store: batch a set of writes into one commit.

A transaction is a **view** of a store, not a new primitive -- its
:attr:`~Transaction.store` is an ordinary :class:`~abczarr.abc.store.Store`
that reads its own pending writes back (read-your-writes) and applies nothing
until :meth:`~Transaction.commit`.

Two flavours:

* a driver whose backend has real transactions (tensorstore, an Icechunk
  session) returns a native transaction from ``store.transaction()`` and
  declares ``transactions`` :attr:`~abczarr.abc.capabilities.Support.NATIVE`;
* every other store gets :class:`BufferedTransaction`, which buffers writes
  and flushes them on commit. It is **never atomic** -- a failure mid-flush
  leaves a partial result -- so it is only offered for
  ``transaction(atomic=False)``. The one rule: an atomic transaction is never
  synthesized.
"""

__all__ = [
    "Transaction",
    "BufferedTransaction",
    "AsyncTransaction",
    "AsyncBufferedTransaction",
]

# stdlib
from abc import ABC, abstractmethod
from types import TracebackType

# dependencies
import typing_extensions as tx

# locals
from .capabilities import Support
from .store import AsyncStore, Store

_SEP = "/"


# ======================================================================
#   sync
# ======================================================================


class Transaction(ABC):
    """A batch of store operations that commit or abort together."""

    #: Whether a commit is all-or-nothing. A synthesized transaction sets
    #: this ``False`` and says so, so a caller can tell what it actually got.
    atomic: bool = False

    @property
    @abstractmethod
    def store(self) -> Store:
        """A store whose reads see this transaction's pending writes."""
        ...

    @abstractmethod
    def commit(self, message: tx.Optional[str] = None) -> None:
        """Apply the batch. *message* is used by backends that record one
        (Icechunk); others ignore it. Raises
        :class:`~abczarr.abc.errors.TransactionConflict` if the store moved on.
        """
        ...

    @abstractmethod
    def abort(self) -> None:
        """Discard the batch without applying it."""
        ...

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(
        self,
        exc_type: tx.Optional[tx.Type[BaseException]],
        exc_value: tx.Optional[BaseException],
        traceback: tx.Optional[TracebackType],
    ) -> None:
        # commit on a clean exit, abort when the body raised
        if exc_type is not None:
            self.abort()
        else:
            self.commit()


class _BufferedView(Store):
    """A store that reads a parent through a pending write/delete buffer."""

    def __init__(
        self,
        parent: Store,
        writes: "tx.Dict[str, bytes]",
        deletes: "tx.Set[str]",
    ) -> None:
        super().__init__(parent.store_path)
        self._parent = parent
        self._writes = writes
        self._deletes = deletes
        self._native = parent.native

    def support(self, capability: str) -> Support:
        # a buffered view can do whatever its parent can
        return self._parent.support(capability)

    def get(self, key: str) -> tx.Optional[bytes]:
        if key in self._deletes:
            return None
        if key in self._writes:
            return self._writes[key]
        return self._parent.get(key)

    def set(self, key: str, value: tx.Any) -> None:
        self._writes[key] = bytes(value)
        self._deletes.discard(key)

    def delete(self, key: str) -> None:
        self._deletes.add(key)
        self._writes.pop(key, None)

    def exists(self, key: str) -> bool:
        if key in self._deletes:
            return False
        if key in self._writes:
            return True
        return self._parent.exists(key)

    def list_keys(self, prefix: str = "") -> tx.Iterator[str]:
        seen = set()  # type: tx.Set[str]
        for key in self._parent.list_keys(prefix):
            if key in self._deletes:
                continue
            seen.add(key)
            yield key
        for key in self._writes:
            if key not in seen and _under(prefix, key):
                yield key


class BufferedTransaction(Transaction):
    """A non-atomic transaction: buffer writes, flush them on commit.

    Reads through :attr:`store` see the buffer; :meth:`commit` writes the
    buffered keys to the parent (and deletes the removed ones) one at a time,
    so a failure part-way leaves a partial result. Deferred, coalesced,
    read-your-writes -- useful for batching many small writes -- but not
    atomic, and it says so through :attr:`atomic`.
    """

    atomic = False

    def __init__(self, parent: Store) -> None:
        self._parent = parent
        self._writes = {}  # type: tx.Dict[str, bytes]
        self._deletes = set()  # type: tx.Set[str]
        self._view = _BufferedView(parent, self._writes, self._deletes)
        self._closed = False

    @property
    def store(self) -> Store:
        return self._view

    def commit(self, message: tx.Optional[str] = None) -> None:
        if self._closed:
            return
        for key, value in self._writes.items():
            self._parent.set(key, value)
        for key in self._deletes:
            self._parent.delete(key)
        self._reset()

    def abort(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._writes.clear()
        self._deletes.clear()
        self._closed = True


# ======================================================================
#   async
# ======================================================================


class AsyncTransaction(ABC):
    """The coroutine twin of :class:`Transaction`."""

    atomic: bool = False

    @property
    @abstractmethod
    def store(self) -> AsyncStore:
        """A store whose reads see this transaction's pending writes."""
        ...

    @abstractmethod
    async def commit(self, message: tx.Optional[str] = None) -> None:
        """Apply the batch (see :meth:`Transaction.commit`)."""
        ...

    @abstractmethod
    async def abort(self) -> None:
        """Discard the batch."""
        ...

    async def __aenter__(self) -> "AsyncTransaction":
        return self

    async def __aexit__(
        self,
        exc_type: tx.Optional[tx.Type[BaseException]],
        exc_value: tx.Optional[BaseException],
        traceback: tx.Optional[TracebackType],
    ) -> None:
        if exc_type is not None:
            await self.abort()
        else:
            await self.commit()


class _AsyncBufferedView(AsyncStore):
    """The async counterpart of :class:`_BufferedView`."""

    def __init__(
        self,
        parent: AsyncStore,
        writes: "tx.Dict[str, bytes]",
        deletes: "tx.Set[str]",
    ) -> None:
        super().__init__(parent.store_path)
        self._parent = parent
        self._writes = writes
        self._deletes = deletes
        self._native = parent.native

    def support(self, capability: str) -> Support:
        return self._parent.support(capability)

    async def get(self, key: str) -> tx.Optional[bytes]:
        if key in self._deletes:
            return None
        if key in self._writes:
            return self._writes[key]
        return await self._parent.get(key)

    async def set(self, key: str, value: tx.Any) -> None:
        self._writes[key] = bytes(value)
        self._deletes.discard(key)

    async def delete(self, key: str) -> None:
        self._deletes.add(key)
        self._writes.pop(key, None)

    async def exists(self, key: str) -> bool:
        if key in self._deletes:
            return False
        if key in self._writes:
            return True
        return await self._parent.exists(key)

    async def list_keys(self, prefix: str = "") -> tx.AsyncIterator[str]:
        seen = set()  # type: tx.Set[str]
        async for key in self._parent.list_keys(prefix):
            if key in self._deletes:
                continue
            seen.add(key)
            yield key
        for key in self._writes:
            if key not in seen and _under(prefix, key):
                yield key


class AsyncBufferedTransaction(AsyncTransaction):
    """The async, non-atomic buffered transaction (see
    :class:`BufferedTransaction`)."""

    atomic = False

    def __init__(self, parent: AsyncStore) -> None:
        self._parent = parent
        self._writes = {}  # type: tx.Dict[str, bytes]
        self._deletes = set()  # type: tx.Set[str]
        self._view = _AsyncBufferedView(parent, self._writes, self._deletes)
        self._closed = False

    @property
    def store(self) -> AsyncStore:
        return self._view

    async def commit(self, message: tx.Optional[str] = None) -> None:
        if self._closed:
            return
        for key, value in self._writes.items():
            await self._parent.set(key, value)
        for key in self._deletes:
            await self._parent.delete(key)
        self._reset()

    async def abort(self) -> None:
        self._reset()

    def _reset(self) -> None:
        self._writes.clear()
        self._deletes.clear()
        self._closed = True


def _under(prefix: str, key: str) -> bool:
    """Whether *key* is at or below *prefix* (matching store prefix rules)."""
    if not prefix:
        return True
    return key == prefix or key.startswith(prefix.rstrip(_SEP) + _SEP)
