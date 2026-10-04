# Design pass: transactions over array and group writes

Status: draft for discussion. Records where transactions stand today, the gap,
and the options for closing it. No code yet.

## Where we are

`Store.transaction(atomic=...)` returns a `Transaction` (`abc/transactions.py`).
A transaction is a *view* of a store: `txn.store` is an ordinary `Store` whose
reads see the transaction's own pending writes, and nothing reaches the
underlying store until `commit()`.

- Backends with real transactions (tensorstore, an Icechunk session) return a
  native transaction.
- Everything else gets `BufferedTransaction`, which buffers `set`/`delete` and
  flushes on commit. It is never atomic, and an atomic transaction is refused
  rather than faked.

This covers **store keys**: `txn.store.set("a/c/0/0", chunk_bytes)`. It does
**not** cover array or group writes: `array[:] = data` and
`group.create_array(...)` do not go through `txn.store`, so they are not part
of any transaction.

## The gap, and why it is not a small patch

The obstacle is structural, not a missing method:

- **`abczarr.open` takes a path, not a store.** There is no way today to open a
  node *onto* a transaction's buffered store view.
- **No driver routes array chunk I/O through abczarr's `Store`.** `ZarrPythonArray`
  writes through `zarr.Array`, `TensorStoreArray` through tensorstore, the new
  `ZarristaArray` through zarrista. Each backend owns its own I/O and its own
  store handle. abczarr's `Store` is used for group metadata and listing (via
  `PathGroup`) and by users directly, but an array write never passes through
  it.

So a chunk write cannot be intercepted by a `BufferedTransaction` the way a
`store.set` can: the bytes never reach the store the transaction is a view of.

Backend transaction support is uneven, too: **tensorstore has `ts.Transaction`**,
**zarr-python has none**.

## The crux

For an array write to be transactional, its chunk I/O must go through a store
the transaction can intercept (buffer or delegate to a native transaction).
There are only two ways to get there:

1. the array does its chunk I/O through *abczarr's own* `Store`, so a buffered
   view sits in front of it; or
2. the backend has its own transaction, and abczarr routes the array's writes
   into it.

zarr-python offers neither today (no native transaction, and it owns its I/O).

## Options

### A. Open a node on a transaction

`abczarr.open` (and `create`) grow the ability to bind to a transaction's store
view, e.g. `node = abczarr.open(location, transaction=txn)` or `txn.open(location)`.
The node's I/O is then store-mediated and buffered.

- Works only for a driver that routes array I/O through abczarr's `Store`. None
  do yet; this needs a **native/store-backed array** (a good fit for a zarrista
  or a from-scratch driver, where abczarr controls the chunk read/write).
- Clean and uniform once such a driver exists: one `Transaction` mechanism, all
  the existing buffered/atomic contract reused unchanged.

### B. Delegate to the backend's native transaction

A `TensorStoreArray` opened in a transaction routes its writes into a
`ts.Transaction`; commit/abort map onto tensorstore's.

- Real atomicity where the backend provides it (tensorstore).
- Per-backend, and only for backends that have transactions. zarr-python is
  left out until it grows one.

### C. A uniform `node.transaction()` that dispatches

`node.transaction(atomic=...)` returns a transactional view of the node,
choosing A or B by capability, and **raising for a backend that supports
neither** — exactly the existing "refuse, do not fake atomic" stance, lifted
from the store level to the node level. `capability("transactions")` /
`capability("atomic_transactions")` already exist to answer the query; they
would gain a node-level meaning.

## StorePath / PathBasedStore, again

This is where the store-vs-location split matters. A transaction is a `Store`
view; opening a node on it means the node's location resolves *through* that
store rather than straight to the filesystem. The recently-renamed
`PathBasedStore` (a store whose keys are paths) is the natural place for a
buffered array driver to live, since it is the one store abczarr fully controls.

## Recommendation (phased)

1. **Lift the capability query to the node.** `node.capability("transactions")`
   answers from the driver, so a caller can tell before trying.
2. **Native path first (option A) on a store-backed driver.** Build array chunk
   I/O over abczarr's `Store` in one driver (zarrista is the candidate, since it
   is pure-Python and we can route its store), and let `open`/`create` bind to a
   transaction there. This proves the uniform mechanism end to end.
3. **Delegate where the backend has it (option B).** Map `TensorStoreArray`
   onto `ts.Transaction`.
4. **Raise, don't fake, elsewhere.** zarr-python and any backend without support
   refuse a node transaction, matching the store-level contract.

This keeps one `Transaction` surface, reuses the atomic/refuse rules, and does
not pretend a backend can do something it cannot.

## Open questions for review

- API shape: `node.transaction()` returning a transactional node view, vs
  `abczarr.open(location, transaction=txn)` binding at open time. (They can
  coexist; which is primary?)
- Is the store-backed native array (step 2) worth building for this, or should
  transactions wait until a native driver exists for other reasons?
- Group writes (`create_array`, `create_group`) inside a transaction: same
  mechanism (metadata writes go through the buffered store), so they come along
  for free on the native path but not through a backend that owns its I/O.
  Confirm that group-metadata transactionality tracks array transactionality.
