"""Transactions: the buffered (non-atomic) default, the atomic guard, and
the native-transaction hook -- over the default bagof-paths store, with no
backend installed.
"""

import asyncio
import pathlib

import pytest
import typing_extensions as tx

from abczarr._errors import UnsupportedZarrOperation
from abczarr.abc.capabilities import Support
from abczarr.abc.store import PathBasedStore, Store
from abczarr.abc.transactions import BufferedTransaction, Transaction

# --------------------------------------------------------------------------
# capability
# --------------------------------------------------------------------------


def test_transactions_are_synthesized_not_atomic(
    tmp_path: pathlib.Path,
) -> None:
    s = PathBasedStore(str(tmp_path))
    assert s.capability("transactions") is Support.SYNTHESIZED
    assert s.supports("transactions") is True
    assert s.supports("atomic_transactions") is False


# --------------------------------------------------------------------------
# the buffered transaction
# --------------------------------------------------------------------------


def test_atomic_transaction_is_refused_not_faked(
    tmp_path: pathlib.Path,
) -> None:
    s = PathBasedStore(str(tmp_path))
    with pytest.raises(UnsupportedZarrOperation, match="atomic"):
        s.transaction(atomic=True)


def test_buffered_transaction_defers_until_commit(
    tmp_path: pathlib.Path,
) -> None:
    s = PathBasedStore(str(tmp_path))
    s.set("a", b"1")
    txn = s.transaction(atomic=False)
    assert isinstance(txn, BufferedTransaction)
    assert txn.atomic is False

    txn.store.set("a", b"2")
    txn.store.set("b", b"new")
    # read-your-writes on the view, nothing on the parent yet
    assert txn.store.get("a") == b"2"
    assert txn.store.get("b") == b"new"
    assert s.get("a") == b"1"
    assert s.get("b") is None

    txn.commit()
    assert s.get("a") == b"2"
    assert s.get("b") == b"new"


def test_buffered_delete_is_visible_and_applied(
    tmp_path: pathlib.Path,
) -> None:
    s = PathBasedStore(str(tmp_path))
    s.set("a", b"1")
    txn = s.transaction(atomic=False)
    txn.store.delete("a")
    assert txn.store.get("a") is None
    assert txn.store.exists("a") is False
    assert s.get("a") == b"1"  # parent untouched pre-commit
    txn.commit()
    assert s.get("a") is None


def test_context_manager_commits_on_clean_exit(
    tmp_path: pathlib.Path,
) -> None:
    s = PathBasedStore(str(tmp_path))
    with s.transaction(atomic=False) as txn:
        txn.store.set("c", b"ctx")
    assert s.get("c") == b"ctx"


def test_context_manager_aborts_on_exception(
    tmp_path: pathlib.Path,
) -> None:
    s = PathBasedStore(str(tmp_path))
    with pytest.raises(RuntimeError, match="boom"):
        with s.transaction(atomic=False) as txn:
            txn.store.set("d", b"lost")
            raise RuntimeError("boom")
    assert s.get("d") is None


def test_abort_discards_the_batch(tmp_path: pathlib.Path) -> None:
    s = PathBasedStore(str(tmp_path))
    txn = s.transaction(atomic=False)
    txn.store.set("x", b"1")
    txn.abort()
    assert s.get("x") is None


def test_view_lists_pending_writes_and_hides_deletes(
    tmp_path: pathlib.Path,
) -> None:
    s = PathBasedStore(str(tmp_path))
    s.set("keep", b"1")
    s.set("gone", b"2")
    txn = s.transaction(atomic=False)
    txn.store.set("added", b"3")
    txn.store.delete("gone")
    assert sorted(txn.store.list_keys()) == ["added", "keep"]


# --------------------------------------------------------------------------
# a store that declares native transactions overrides the hook
# --------------------------------------------------------------------------


class _RecordingTransaction(Transaction):
    atomic = True

    def __init__(self, parent: Store) -> None:
        self._parent = parent
        self.committed = False

    @property
    def store(self) -> Store:
        return self._parent

    def commit(self, message: tx.Optional[str] = None) -> None:
        self.committed = True

    def abort(self) -> None:
        ...


class _NativeTxnStore(PathBasedStore):
    _CAPABILITIES = dict(
        PathBasedStore._CAPABILITIES, transactions=Support.NATIVE
    )

    def _native_transaction(self, *, atomic: bool) -> Transaction:
        return _RecordingTransaction(self)


def test_native_transaction_hook_is_used(tmp_path: pathlib.Path) -> None:
    s = _NativeTxnStore(str(tmp_path))
    assert s.capability("transactions") is Support.NATIVE
    # a native store may honour atomic=True -- it does not go through the guard
    txn = s.transaction(atomic=True)
    assert isinstance(txn, _RecordingTransaction)
    txn.commit()
    assert txn.committed is True


# --------------------------------------------------------------------------
# async
# --------------------------------------------------------------------------


def test_async_buffered_transaction(tmp_path: pathlib.Path) -> None:
    from abczarr.abc.store import AsyncPathBasedStore

    async def scenario() -> None:
        a = AsyncPathBasedStore(str(tmp_path))
        await a.set("a", b"1")
        async with a.transaction(atomic=False) as txn:
            await txn.store.set("b", b"new")
            assert await txn.store.get("b") == b"new"
            assert await a.get("b") is None  # deferred
        assert await a.get("b") == b"new"

    asyncio.run(scenario())


def test_async_atomic_transaction_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    from abczarr.abc.store import AsyncPathBasedStore

    a = AsyncPathBasedStore(str(tmp_path))
    with pytest.raises(UnsupportedZarrOperation, match="atomic"):
        a.transaction(atomic=True)
