# abczarr: async & transactions design

Status: **design** (nothing here is built yet). Python floor 3.8. This is the
plan for giving abczarr's **nodes** — arrays and groups, the objects users
actually hold — a sync surface, an async surface, and a transaction surface,
each leveraging a backend's native machinery where it has it.

It extends decision 4 of the [roadmap](./roadmap.md) ("sync-first, native-async
where the backend earns it, defer the hard direction") with the concrete node
model, the transaction object model, and a per-backend compatibility analysis.

Findings below were verified against the repository tree and against live
probes of the installed backends (zarr-python 3.1.6, tensorstore, zarrista
0.1.0), not from memory. Icechunk is **not installed here**, so its analysis is
reasoned from its documented API (`repo.writable_session` / `session.commit`)
and is marked where it is inference.

---

## 0. The short version

Six decisions carry the plan:

1. **The surface lives on nodes, not stores.** In abczarr a `Store` is mostly a
   path-holder; no node's chunk I/O passes through one. Sync / async /
   transactional APIs belong on `ZarrArray` / `ZarrGroup`.
2. **Two colors, one native.** A driver implements its node ops once, in
   whichever color its backend is native to, and declares it. abczarr
   synthesizes the **async-from-sync** direction (thread pool) only. It does
   **not** build the background-loop `sync()` (async-from-sync's reverse) in v1
   — all three real backends are natively bi-colored, so it has no customer yet.
3. **Transactions are first-class objects, spawned by the node.** A
   `Transaction` collects operations; `node.with_transaction(txn)` returns a
   node *view* whose writes enlist in it; one transaction can span many nodes.
4. **`with_transaction` means "an equivalent node whose writes enlist" — not
   "rebind this handle."** tensorstore rebinds an open handle; Icechunk must
   re-open the node on a session. Both satisfy the same contract only if it is
   written the general way from day one.
5. **No faked transactions.** A backend with no native support reports
   `transactions: NONE` and `with_transaction` raises. No buffered *node*
   fallback — buffering arbitrary NumPy selections is not read-your-writes.
6. **Atomicity is a property of the (driver × store), decided at commit.** Not a
   class constant: tensorstore's `atomic=True` *succeeds* on `memory`/`ocdbt`
   and *fails at commit* on the `file` kvstore, for even a single chunk.

---

## 1. Where it stands today (verified)

| Fact | Evidence |
| --- | --- |
| Nodes are **sync-only**. | `abc/array.py` (`__getitem__`/`__setitem__`), `abc/group.py`, `abc/node.py` — no coroutines. |
| The async + transaction scaffolding lives at the **store** layer and **no driver uses it**. | `abc/store.py` (`AsyncStore`, `transaction`, `_native_transaction`), `abc/transactions.py`; grep finds no `_native_transaction` override and no driver building an `AsyncStore`. |
| The tensorstore node reads via a **blocking** `.result()` yet declares `async: NATIVE`. | `drivers/tensorstore.py:158-161, 45, 115`. The declaration is aspirational — there is no async code path. |
| `_SYNTHESIZED_FLOOR["transactions"] = SYNTHESIZED` advertises transactions for **every** store, but that is only ever true of raw key writes. | `abc/store.py:68-71`; the only `Store` consumers read a single key (`_peek_node_type`). |
| **Nodes have no link to their driver.** | `TensorStoreArray.__init__(array)`, `ZarrPythonNode.__init__(obj)` take only the backend object; `available_drivers()` builds a fresh stateless `Driver` per call (`api/registry.py:54-63`). |
| **Attributes bypass every layer.** | `ZarrNode.attrs` → `Attributes(self)` writes the metadata file directly (`mkstemp`+`fsync`+`replace`, `_core/attributes.py:206-232`); goes through neither `Store` nor backend, so it cannot enlist in a transaction (and is local-FS-only in practice). |

Backend facts established by probe:

- **tensorstore.** Every op returns an awaitable `ts.Future` (has both
  `__await__` and `.result()`). `ts.Transaction(atomic=…)` with
  `commit_async` / `commit_sync` / `abort` / `atomic`; `with_transaction` on
  both `TensorStore` and `KvStore`. A transaction is a standalone object bound
  via `arr.with_transaction(txn)`, returning a **new** handle (`v is arr` is
  `False`). Verified lifecycle: read-your-writes inside the view, invisible
  outside until commit; reuse after commit → `ValueError("Transaction not
  open")`; double `commit` is a silent no-op; `abort` after commit is a no-op;
  two non-atomic transactions on the same chunk commit **last-writer-wins with
  no conflict** (so `TransactionConflict` never fires for tensorstore unless
  `repeatable_read=True`). **`atomic=True` fails at commit on the `file`
  kvstore** ("Cannot … as single atomic transaction") because it folds
  `zarr.json` in and `file` guarantees only single-key atomicity; the identical
  code commits atomically on `memory` and `ocdbt` (4-chunk and cross-array
  both verified).
- **zarr-python 3.1.6** is async-first: `AsyncArray`/`AsyncGroup` are the real
  coroutine implementations; sync `Array`/`Group` hold an `async_array` and
  drive it through `zarr.core.sync.sync()` on a daemon-thread loop. Its
  re-entrancy guard only raises when called from *its own* loop — called from a
  user's running loop it silently blocks it.
- **zarrista 0.1.0** ships both `Array` and `AsyncArray` and its own thread
  pool — natively bi-colored too.

The consequence of the last three: **no backend in the driver set needs
sync-from-async synthesis.** Each already offers both colors.

---

## 2. The user surface (proposed)

> The spellings below are the proposed surface, not existing API. `open(...)`
> is abczarr's façade; `asynchronous=True` selects the async twin. Async uses
> `getitem`/`setitem` **methods**, not `[]`, because an assignment expression
> cannot be awaited — the same choice zarr-python made.

### 2.1 Sync — unchanged from today

```python
import abczarr

arr = abczarr.open("data.zarr", mode="r")
block = arr[0:64, 0:64]                 # __getitem__
arr[0:64, 0:64] = block * 2             # __setitem__ (mode="a")
```

### 2.2 Async

```python
import abczarr, asyncio

arr = abczarr.open("data.zarr", mode="a", asynchronous=True)   # AsyncZarrArray

block = await arr.getitem((slice(0, 64), slice(0, 64)))
await arr.setitem((slice(0, 64), slice(0, 64)), block * 2)

# fan many chunk reads out concurrently onto the backend's own executor
regions = [(slice(i, i + 64), slice(0, 64)) for i in range(0, 512, 64)]
blocks = await asyncio.gather(*(arr.getitem(r) for r in regions))
```

For tensorstore each `getitem` is `await handle[r].read()` (a native Future);
for zarr-python it delegates to `AsyncArray.getitem`; for a sync-only backend
it runs the sync read in a bounded thread pool. Which one you got is reported
honestly:

```python
arr.supports("async")     # NATIVE (tensorstore, zarr-python, zarrista) or SYNTHESIZED (threaded)
```

Convert between colors without re-opening — the two facades share one core:

```python
sync_arr  = async_arr.as_sync()
async_arr = sync_arr.as_async()
```

### 2.3 Transactions — one node

```python
arr = abczarr.open("data.zarr", mode="a")

with arr.transaction(atomic=True) as txn:      # spawns a Transaction from arr's driver
    arr.with_transaction(txn)[0:2, 0:2] = 1    # a transactional VIEW of arr
    arr.with_transaction(txn)[2:4, 2:4] = 2
# clean exit commits; an exception aborts (tensorstore: memory/ocdbt kvstore)
```

`arr.transaction(...)` is the convenience spawner; the transaction is a
standalone object, so it is equally `txn = driver.transaction()` bound later.

### 2.4 Transactions — spanning several nodes (the reason it is an object)

```python
group  = abczarr.open("dataset.zarr", mode="a")
raw    = group["raw"]
labels = group["labels"]

with group.transaction(atomic=True) as txn:
    raw.with_transaction(txn)[0:2, 0:2]    = data
    labels.with_transaction(txn)[0:2, 0:2] = mask
# both arrays land together, or neither
```

### 2.5 Transactions — commit message, and identical code across backends

```python
txn = group.transaction()
group["raw"].with_transaction(txn)[...] = data
txn.commit(message="ingest run 2026-09-03")   # Icechunk records it; tensorstore ignores it
```

The **same** user code works whether the backend rebinds an open handle
(tensorstore) or re-opens the node on a session (Icechunk) — that is exactly
what the "equivalent node whose writes enlist" contract (§3.2) buys.

### 2.6 Async + transaction

```python
arr = abczarr.open("data.zarr", mode="a", asynchronous=True)

async with arr.transaction(atomic=True) as txn:     # AsyncTransaction
    view = arr.with_transaction(txn)
    await view.setitem((slice(0, 2),), 1)
    await view.setitem((slice(2, 4),), 2)
# clean exit awaits txn.commit() (commit_async under the hood)
```

### 2.7 What raises

```python
# a backend with no native transaction support (v1: no buffered node fallback)
arr.transaction(atomic=False)          # -> UnsupportedZarrOperation

# atomic asked of a (driver × store) that cannot honour it
arr.supports("atomic_transactions")    # answered per instance: file->False, memory/ocdbt->True
```

---

## 3. The model

### 3.1 Node color bridge — synthesize only async-from-sync

A driver implements node ops once, in its native color, and declares the color.
The sync/async facades adapt each op to the caller's color:

| caller ↓ / op → | native sync | native async |
| --- | --- | --- |
| **sync facade** | direct | *not synthesized in v1* (would need background-loop `sync()`) |
| **async facade** | `await to_thread(fn)` (bounded pool) | direct (`await future` / `await coro`) |

Because every real backend is bi-colored, the top-right cell has no customer;
building it is building for a hypothetical pure-coroutine Python backend. Defer
it (§6). `async: NATIVE` therefore means "the async twin's ops are native
coroutines/futures", true for both drivers once the twins wrap
`ts.Future` / `zarr.AsyncArray`.

Thread-synthesis rules (for a driver that ships no async surface): a
**dedicated bounded executor** (not the default pool), `get_running_loop()` not
the deprecated `get_event_loop()`, a default semaphore on the `gather`
fan-out (`concurrent_map` / `get_many` are unbounded today), and the documented
truth that a cancelled `await` cannot interrupt the running thread — the write
may still land.

### 3.2 Transactions as first-class objects, with a re-open contract

- `Transaction` / `AsyncTransaction` are objects that collect operations:
  `commit(message=None)`, `abort()`, `atomic`, context manager. (The context
  manager already in `abc/transactions.py:95-108` matches tensorstore's
  `__exit__` verbatim — keep it.)
- **Spawned by the node**, not the `Driver`: `ZarrNode._spawn_transaction(*,
  atomic) -> Transaction`, default raising. The node carries the backend handle
  a native transaction needs; `Driver` stays stateless (see §1). `node.
  transaction()` is the public convenience over it.
- `node.with_transaction(txn) -> Self` returns **an equivalent node whose
  writes enlist in `txn`** — for tensorstore a rebound handle
  (`native.with_transaction`), for Icechunk a node re-opened on
  `session.store`. The contract is deliberately *not* "rebind this handle", so
  the re-open backend fits.
- Because the transaction is standalone, **it spans nodes**: bind as many as
  you like, commit once.

Four rules for the view:

1. It is a **new node**, never a mutation: `base.native` stays unbound.
2. Identity stays distinct (nodes have no `__eq__`/`__hash__` — keep it).
3. **Lifecycle:** after commit/abort the view is dead. Wrap tensorstore's
   `ValueError("Transaction not open")` in a new `TransactionClosed`
   (`abc/errors.py`) that names the node and op; rebinding a live view is
   refused uniformly (tensorstore refuses it too). `to_dask()` on a view is a
   trap — the graph reads *after* commit — so refuse it on views or document
   loudly.
4. **Derivation:** children of a view inherit the transaction.
   `group.with_transaction(txn)["a"]` must be transactional. `PathGroup`
   builds children in four places with no single hook (`abc/group.py:220-221,
   246, 263`) — add a `_derive(child_native) -> Self` seam (bagof-paths'
   `with_wrapped` equivalent) **before** views exist, or the transaction leaks
   at the first `__getitem__`.

Group-structure and attribute writes: because `attrs` and `create_group`
bypass the store (§1), a transactional view's `attrs["k"] = v` /
`create_group(...)` would silently **escape** the transaction. v1 contract:
they **raise** on a view unless the driver routes them (tensorstore can, via
`ts.KvStore.with_transaction`, as a later increment).

### 3.3 Colored transactions: shared core + two thin facades

A transaction is itself colored (`commit_sync` vs `commit_async`). Neither one
object with both methods (invites `with atxn:` misuse) nor two disjoint classes
(no way to commit an async-written batch from sync code, and duplicate state).
Instead:

- a color-neutral `_TransactionCore` holds the native handle and
  `open`/`committed`/`aborted`/`atomic`;
- `Transaction` and `AsyncTransaction` are thin facades over one core, mirroring
  the node twins; `txn.as_async()` / `atxn.as_sync()` return the other facade
  over the **same** core (so read-your-writes is shared for free);
- pairing: `ZarrNode.with_transaction` takes a `Transaction`,
  `AsyncZarrNode.with_transaction` an `AsyncTransaction`, each converting the
  other.

`abc/transactions.py` keeps `Transaction`/`AsyncTransaction` as the ABCs but
**drops the abstract `store` property from the base**; the store-view flavour
moves to a `StoreTransaction` subclass that `BufferedTransaction` extends
(still the fallback for `PathBasedStore`, never a node surface).

### 3.4 Atomicity is answered per instance

`atomic_transactions` cannot be a class constant (§1: it depends on the
kvstore). `SupportsCapabilities.capability` is already documented as overridable
for live state, so `TensorStoreArray.capability` inspects
`self._array.kvstore.spec()`'s driver; where it still cannot know
(`ocdbt`-over-`file`, etc.), it answers "will find out at commit" honestly
rather than `NATIVE`, and a commit-time atomicity `ValueError` is mapped to
`UnsupportedZarrOperation("atomic transaction", driver=…)`. Keep `atomic`, but
do not let `TransactionConflict` promise detection a backend won't do; report
conflict detection separately (tensorstore: none by default; Icechunk: real).

---

## 4. Backend compatibility

| Backend | Async | Transactions | `with_transaction` | Notes |
| --- | --- | --- | --- | --- |
| **tensorstore** | native (`await ts.Future`) | native `ts.Transaction` | **rebind** open handle | atomic per (driver × kvstore), decided at commit; no conflict detection by default |
| **zarr-python** | native (`AsyncArray`) | none of its own | n/a → raises | gains transactions only via an Icechunk store |
| **Icechunk** *(inferred)* | via zarr's async | native `Session` | **re-open** node on `session.store` | `commit(message)`, snapshot isolation, real `ConflictError`; captures writes at the store level, so it wants the re-open contract |
| **zarrista** | native (`AsyncArray`) | none known | raises | bi-colored; no transaction API observed |
| **path-based** | threaded (bagof.paths) | store-level buffered only | raises at node level | the demoted store `BufferedTransaction` lives here |

Icechunk is the reason the contract is "equivalent node whose writes enlist":
it has no operation to bind an already-open `zarr.Array` to a session, so an
Icechunk `with_transaction` re-opens on `session.store` and returns a new node.
Its store-level capture also makes `attrs`/group writes *naturally*
transactional once they route through the store — which abczarr's current
attrs-bypass (§1) blocks, independent of this design.

---

## 5. What lives where

- `abc/node.py` — `with_transaction(txn) -> Self`, `transaction(*, atomic=True)
  -> Transaction`, `_spawn_transaction` (default raises), `_derive` hook;
  node-level meaning for `capability("transactions"/"atomic_transactions")`,
  the latter answerable **per instance**.
- `abc/array.py` + new async twins (`abc/asyncnode.py` or
  `async_array.py`/`async_group.py`) — the coroutine surface
  (`getitem`/`setitem`, `create_array`, …), plus `as_sync`/`as_async`.
- `abc/transactions.py` — reshaped per §3.3: base ABCs lose `store`;
  `_TransactionCore`; `StoreTransaction` holds the buffered fallback.
- `abc/errors.py` — `TransactionClosed`; map tensorstore's commit-time
  atomicity error to `UnsupportedZarrOperation`.
- `_core/asyncutils.py` — bounded dedicated executor, `get_running_loop`,
  default semaphore on fan-out, drop `get_loop`; **no** background-loop
  `sync()` yet.
- `drivers/tensorstore.py` — one shared op layer used by both facades: sync =
  `future.result()`, async = `await future`; `_spawn_transaction` over
  `ts.Transaction`; per-instance `atomic_transactions`.
- `test_parity.py` — a bagof-paths-style parity test so the sync and async
  surfaces cannot drift.

---

## 6. Plan

**Change before writing a line** (vs the first sketch):

1. Drop "synthesize sync-from-async via background loop" from v1; drivers
   implement both colors natively, abczarr synthesizes async-from-sync only.
2. Drop the buffered **node** transaction; node-level `transactions` is `NONE`
   without native support.
3. `with_transaction` contract is "equivalent node whose writes enlist" so
   Icechunk's re-open fits; `atomic_transactions` is per instance.
4. Spawn hook on the node, not `Driver`; add the `PathGroup` derivation seam
   before views exist.
5. Transaction = shared core + sync/async facades; base ABC loses `store`.

**Prototype first — tensorstore only** (`memory`/`ocdbt` for the atomic tests,
`file` for the commit-time-failure test):

- `TensorStoreTransaction` over `ts.Transaction`;
- `with_transaction` returning a rebound `TensorStoreArray`;
- multi-array atomic commit; commit-time atomicity failure → clear error;
  `TransactionClosed` on reuse; group children inheriting the transaction;
- `AsyncTensorStoreArray` with `getitem`/`setitem` awaiting Futures;
- the parity test.

**Second:** `AsyncZarrPythonArray`/`Group` wrapping `zarr.AsyncArray`/
`AsyncGroup` directly (no bridge); zarr-python `with_transaction` raises until
an Icechunk driver exists; a generic thread-synthesized async twin on a bounded
executor for any driver that ships no async surface.

**Defer:** background-loop `sync()` (build when a coroutine-only backend arrives,
with the stricter re-entrancy guard, fork reset and `atexit` cleanup); the
Icechunk driver (design the re-open path then); transactional `attrs` /
group-structure writes via `ts.KvStore.with_transaction`; timeouts on the sync
surface and on `commit()`.

---

## 7. Open questions

- **Timeouts.** tensorstore has `.result(timeout=)` and zarr's `sync()` takes a
  timeout; the abczarr surface has neither. Add optional `timeout=` to the sync
  I/O and to `commit()`?
- **Partial failure** of a non-atomic multi-node commit leaves some chunks
  written with no rollback — the API must say `commit()` on a non-atomic
  transaction can partially apply and closes regardless.
- **Cross-loop stores.** A remote (fsspec) store's aiohttp session is bound to
  one loop; an async twin awaiting on a user's loop while a sync facade drives
  the same store on another is a hazard for remote backends (local is fine).
  Guard or document.
- **`store()` on a view** (atomic dask write) holds the whole array in the
  backend's writeback cache until `commit()` — refuse above a size, or warn?
