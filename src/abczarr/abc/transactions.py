"""Transactions over a store: batch a set of writes into one commit.

A transaction is a **view** of a store, not a new kind of object --
its [store][abczarr.abc.transactions.Transaction.store] property is
an ordinary [Store][abczarr.abc.store.Store] whose reads see the
transaction's own pending writes, and nothing is applied to the
underlying store until
[commit][abczarr.abc.transactions.Transaction.commit] is called.

Open one with `store.transaction()`, ideally as a context manager --
it commits on a clean exit and aborts if the block raises:

```python
with store.transaction(atomic=False) as txn:
    txn.store.set("a", b"1")
    txn.store.set("b", b"2")
# both writes land together here
```

Two flavours:

* a backend with real transactions (tensorstore, an Icechunk
  session) returns a native transaction from `store.transaction()`;
* every other store gets
  [BufferedTransaction][abczarr.abc.transactions.BufferedTransaction],
  which buffers writes and flushes them on commit. It is **never
  atomic** -- a failure part-way through the flush leaves a partial
  result -- so it is only offered for `transaction(atomic=False)`.
  An atomic transaction is never built this way; a store without
  native support for one raises instead of pretending.
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

    #: Whether a commit is all-or-nothing. A non-native transaction
    #: sets this `False` and says so, so a caller can tell what it
    #: actually got.
    atomic: bool = False

    @property
    @abstractmethod
    def store(self) -> Store:
        """A store whose reads see this transaction's pending writes."""
        ...

    @abstractmethod
    def commit(self, message: tx.Optional[str] = None) -> None:
        """Apply the batch.

        Parameters
        ----------
        message : str, optional
            Recorded by a backend that keeps commit messages
            (Icechunk); ignored by others.

        Raises
        ------
        [TransactionConflict][abczarr.abc.errors.TransactionConflict]
            If the store moved on underneath this transaction.
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

    def capability(self, capability: str) -> Support:
        # a buffered view can do whatever its parent can
        return self._parent.capability(capability)

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

    Reads through `store` see the buffered writes and deletes;
    `commit` then applies them to the parent store one at a time, so
    a failure part-way through leaves a partial result. Useful for
    batching many small writes on a backend with no native
    transaction support, but never atomic -- `atomic` is always
    `False`.
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
    """The coroutine twin of
    [Transaction][abczarr.abc.transactions.Transaction]."""

    atomic: bool = False

    @property
    @abstractmethod
    def store(self) -> AsyncStore:
        """A store whose reads see this transaction's pending writes."""
        ...

    @abstractmethod
    async def commit(self, message: tx.Optional[str] = None) -> None:
        """Apply the batch.

        See
        [Transaction.commit][abczarr.abc.transactions.Transaction.commit].
        """
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
    """The async counterpart of `_BufferedView`."""

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

    def capability(self, capability: str) -> Support:
        return self._parent.capability(capability)

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
    """The async, non-atomic buffered transaction.

    See
    [BufferedTransaction][abczarr.abc.transactions.BufferedTransaction].
    """

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
