# abczarr abc surface — design

The plan for the generic (`abc`) surface, to settle **before** writing driver
code. It answers three questions the roadmap left open:

1. What is the shape of the generic surface, node by node?
2. **Do we also need an abc surface for *stores*?** (Yes — but a specific,
   thin one, mostly synthesized over bagof-paths.)
3. Does that surface map cleanly onto the proposed drivers — zarr-python,
   tensorstore, zarrs, zarrita?

Every backend claim below was checked by running the library, per the repo
rule "find out what they do — by running it": zarr-python 3.1.6 and
tensorstore, introspected directly.

---

## 0. The short version

- **The model is two layers, and every backend agrees.** A **store** is a
  key→bytes map (`get`/`set`/`list`/`delete`); a **node** (array or group) is
  the typed structure built on top of it. zarr-python (`zarr.abc.store.Store`
  under `zarr.Array`) and tensorstore (`ts.KvStore` under `ts.TensorStore`,
  reachable as `array.kvstore`) are both built exactly this way. We keep the
  same two layers.

- **Yes, we need a store abc — but a thin one.** It is the one seam every
  driver, fallback, and format-version reader shares, and the natural place
  the sync/async split is absorbed. Its *default* implementation is
  **synthesized over bagof-paths** (`get` = `read_bytes`, `set` =
  `write_bytes`, `list` = `walk`, `delete` = `unlink`), which gives us local,
  fsspec and cloud stores for free with no per-backend code. A backend's
  *native* store is used only when that backend's driver is active, to keep
  its own strengths (partial reads, sharding-aware I/O).

- **The node surface stays lowest-common-denominator, with two escape
  valves.** `node.native` (the raw `zarr.Array` / `ts.TensorStore`) and
  `driver.supports(capability)` let a caller reach past the shared surface
  without the shared surface having to grow every backend's features.

- **Sync-first, async where earned.** The public surface is synchronous;
  `Async*` twins exist where a backend has a real coroutine I/O path. The
  store seam is where async actually matters, and bagof-paths already carries
  a sync `Path` and an async `AsyncPath` over one spec — so the store abc
  inherits the hard part rather than reinventing it.

- **Three "zarr in Rust" names, three different things — tell them apart.**
  *zarrs-python* is a Rust **codec pipeline that plugs into zarr-python**, not
  a separate array/store API — a configuration under the zarr-python driver,
  not a driver. *zarrita* is the old prototype, absorbed into zarr-python
  3.x; the standalone package is legacy — retire it. *zarrista*
  (developmentseed) is a **full standalone implementation** on the same Rust
  zarrs core, with its own `Array`/`Group` + `AsyncArray`/`AsyncGroup` API —
  a real driver candidate, and the one that fits abczarr's two surfaces best.
  So the driver set is **zarr-python** and **tensorstore** today, with
  **zarrista** the strongest next candidate, zarrs-python a codec option, and
  zarrita retired.

---

## 1. The two-layer model, confirmed against the backends

| Layer | Responsibility | zarr-python 3.1.6 | tensorstore |
|---|---|---|---|
| **Store** | key→bytes: `get`, `set`, `list`, `delete`, `exists` | `zarr.abc.store.Store` (async) | `ts.KvStore`: `read`, `write`, `delete_range`, `list` (futures) |
| **Node** | typed array/group over a store | `zarr.Array` / `zarr.Group` | `ts.TensorStore` (array); groups are just paths |
| **Bridge** | how a node names its store | `array.store`, `array.store_path` | `array.kvstore` |

Both split store from node, and both let you reach the store from the node.
That is the invariant abczarr models. The metadata layer (already built,
`abczarr.metadata` / `abczarr.ome`) is the *content* of the store's
`zarr.json` / `.zarray` / `.zgroup` keys — so the store abc is also what the
version-specific metadata readers sit behind.

### zarr-python's store surface (introspected)

```
abstract : get  set  delete  exists  list  list_dir  list_prefix
           get_partial_values
           supports_writes  supports_deletes  supports_listing
concrete : clear  close  delete_dir  getsize  getsize_prefix  is_empty
           open  set_if_not_exists  with_read_only  read_only
           supports_partial_writes  supports_consolidated_metadata
```

Every I/O method is a coroutine. Capabilities are boolean properties on the
store — the same idea as our `ZarrNode.supports(...)`, and the store abc
should expose the same capability query rather than inventing a second style.

### tensorstore's store surface (introspected)

`ts.KvStore`: `read`, `write`, `delete_range`, `list`, `copy`, `base`,
`path`, `url`, `spec`, `transaction`. Operations return futures (`.result()`
to block). The array (`ts.TensorStore`) carries `chunk_layout`, `codec`,
`domain`, `dtype`, `fill_value`, `oindex`, `vindex`, transforms — far richer
than the shared node surface, which is exactly why `node.native` exists.

---

## 2. The node surface (`abc/node.py`, `array.py`, `group.py`)

This layer is already in good shape and needs only tightening. Keep it as the
**lowest common denominator** — the members every backend can honor — and
push everything backend-specific behind `native` / `supports`.

**`ZarrNode`** (abstract): `metadata`, `attrs`, `zarr_version`; concrete
`store_path`, `native`, `supports(capability)`. Good as is.

**`ZarrArray`** (abstract): `ndim`, `shape`, `dtype`, `chunks`, `shards`,
`__getitem__`, `__setitem__`; concrete `__array__`, `to_dask`. Good — the
lowest-common-denominator array. Richer indexing (tensorstore `oindex` /
`vindex`, label-based domains) is reached through `native`.

**`ZarrGroup`** (abstract): `__getitem__`, `__setitem__`, `__delitem__`,
`create_group`, `create_array`. Good.

### Fixes this surface needs (independent of any new driver)

1. **`metadata` is abstract but no array driver implements it** — both
   drivers currently satisfy it by `__getattr__` delegation, which defeats
   the point of a typed surface. Make each driver return an
   `abczarr.metadata` object.
2. **`create_array` signature drift** — the abc says `config=`, drivers use
   `zarr_config=`. Pick one (`config=`) and hold it with a parity test.
3. **One spec table for the surface**, mirroring bagof-paths' `_spec.py`:
   describe each member once (pure vs I/O, sync-only vs has-async-twin, how a
   returned node is re-wrapped). A `test_parity.py` then keeps the sync and
   async surfaces, and the abc and each driver, from drifting — the same
   mechanism that keeps bagof-paths honest.

---

## 3. Do we need a store abc? — yes, and here is its shape

### Why a store abc at all

The store is the **one seam shared by everything downstream**: every driver,
every metadata-version reader, and every fallback we might synthesize reads
and writes through it. Without it:

- each driver re-implements local/cloud I/O (tensorstore already hand-rolls
  kvstore JSON from a path; zarr-python passes raw strings — two different
  ad-hoc paths for the same job);
- a dependency-free fallback driver (roadmap's deferred item) has nowhere to
  stand;
- the sync/async split has to be solved separately in every driver instead of
  once.

So: **keep `abc/store.py`, but make it a real, minimal ABC** — today it is a
half-built concrete class with methods (`close`, `iter_keys`, `bucket`)
referenced but never defined, and no wiring to actual reads.

### The minimal store contract

Model it on the intersection of zarr's `Store` and tensorstore's `KvStore` —
five primitives and a capability query:

```
get(key)            -> bytes | None          # None = missing
set(key, value)     -> None
delete(key)         -> None
exists(key)         -> bool
list(prefix="")     -> Iterator[str]
supports(capability)-> bool                   # writes, listing, deletes,
                                              # partial_read, partial_write
```

Everything else zarr's store offers (`get_partial_values`, `list_dir`,
`getsize`, `set_if_not_exists`, `clear`, `delete_dir`) is **synthesized** from
these five (delegate → synthesize → raise), or delegated to a native store
that has it and gated behind `supports`. `list_dir` is `list` + a split on
the separator; `clear` is `list` + `delete`; `get_partial_values` is
`partial_read`-gated.

### The default store is bagof-paths, not per-backend code

A Zarr key (`"0/1/2"`, `"zarr.json"`) is a path under the store root. So the
**generic store is a wrapper over a bagof-paths `Path`**:

| store primitive | bagof-paths |
|---|---|
| `get(key)` | `(root / key).read_bytes()` (missing → `None`) |
| `set(key, v)` | `(root / key).write_bytes(v)` (parents created) |
| `delete(key)` | `(root / key).unlink()` |
| `exists(key)` | `(root / key).exists()` |
| `list(prefix)` | `(root / prefix).walk()` → keys relative to root |

Because bagof-paths already dispatches on URL scheme (local, `s3://`,
`gs://`, `az://`, `http://`, any fsspec backend) and carries credentials via
`storage_options`, **one `PathStore` gives every filesystem and cloud store
for free** — no `S3Store` / `GCSStore` / `AzureStore` subclasses with real
bodies. The current empty subclasses in `abc/store.py` collapse into this one
class plus bagof-paths' protocol registry.

This is the same philosophy as the node surface and as bagof-paths itself:
one uniform surface, synthesized from primitives, escape hatch (`store.native`
→ the bagof-paths `Path`, and through it `.wrapped` → the raw driver) for
anything unnamed.

### When a native store is used instead

A backend driver that has its own store (zarr-python's `LocalStore`/`FsspecStore`,
tensorstore's `KvStore`) uses **its** store when that driver is active, so the
backend keeps its own optimized, possibly-partial, possibly-async I/O. The
abc store then *wraps* the native one (delegate to it, `supports` reports what
it can do). `PathStore`-over-bagof-paths is the **default and the fallback**;
a native store is an optimization the driver opts into. Both satisfy the same
five-method ABC, so nothing downstream can tell them apart except through
`supports`.

### async

`Store` gets an `AsyncStore` twin, exactly as bagof-paths has `Path` /
`AsyncPath` (which `abc/path.py`'s `StorePath` / `AsyncStorePath` already
subclass). The sync `PathStore` runs on `Path`; the async `AsyncPathStore`
runs on `AsyncPath` — so the async store is *also* free from bagof-paths,
including its native async cloud path (`AsyncFSPath`). A native async store
(zarr's coroutine `Store`, tensorstore's futures) is awaited through the same
seam. The store is where async earns its keep — chunk read/write is the hot,
blocking, parallel boundary — so this is the one place the async twin is
non-negotiable.

---

## 4. The driver contract

A driver supplies **primitives only**; the abc synthesizes the rest.

```
open(store_or_path, ...)     -> ZarrNode        # sniff array vs group
metadata                     -> NodeMetadata    # typed, per version
read/write a chunk (or region)                  # the I/O the store can't infer
create_group / create_array
list / delete children (group)
_CAPABILITIES                -> frozenset        # what supports() reports
```

Anything richer — `to_dask`, `__array__`, `list_dir`, whole-array
`__getitem__` from per-chunk reads — is **synthesized once** in the abc from
these, so a new driver is small. A driver may *override* a synthesized member
when it can do better (tensorstore reading a whole region in one call), and
declares that reach through `supports`.

One named error for "this backend can't do that": `UnsupportedZarrOperation`
(already in `abc/errors.py`), never a bare `NotImplementedError` — the same
role `UnsupportedPathOperation` plays in bagof-paths.

---

## 5. Compatibility with the proposed drivers

Checked against the real libraries.

### zarr-python — **fits cleanly; the reference driver**

- Store: `zarr.abc.store.Store` maps onto our five primitives directly
  (`get`/`set`/`delete`/`exists`/`list`), and its `supports_*` properties feed
  `supports`. It is async underneath; the sync driver bridges (its own sync
  API also exists).
- Array/group: `zarr.Array` / `zarr.Group` supply every node member. The
  current driver already wraps them; it needs the real `metadata` property,
  a real `fill_value` (currently a TODO), and the `create_array` keyword
  fixed.
- Codecs v2 and v3, sharding, consolidated metadata: all present → advertise
  via `supports`. Our metadata layer already models all three.
- **Verdict:** the lowest-common-denominator surface loses nothing that
  matters here; backend-specific extras stay reachable via `native`.

### tensorstore — **fits, with the widest `native` gap**

- Store: `ts.KvStore` maps on — `read`→`get`, `write`→`set`,
  `delete_range`→`delete`, `list`→`list`. It is future-based; the sync driver
  calls `.result()` (as the current one does), an `AsyncStore` awaits.
- Array: `ts.TensorStore` supplies `shape`/`dtype`/`chunk_layout`/`ndim`; the
  driver derives `chunks`/`shards` from read-vs-write chunk shape (already
  done). Its rich indexing (`oindex`, `vindex`, index transforms, labeled
  domains) is **deliberately not** in the shared surface — reached through
  `native`.
- Groups: tensorstore has no group object; groups are paths with detected
  metadata (already how the driver works). `ZarrGroup.__setitem__` stays
  `UnsupportedZarrOperation` — a documented, named gap, not a crash.
- **Verdict:** fits the surface; the `native` escape hatch is load-bearing
  here and justifies its existence.

### zarrs-python — **not a driver: a codec pipeline under zarr-python**

`zarrs-python` is a Rust reimplementation of the codec pipeline that
registers *into* zarr-python. It has no separate store or array API. So it is
a **configuration of the zarr-python driver** (select the zarrs pipeline),
surfaced — if at all — as a driver *option*, not a peer driver. The roadmap's
"implement the zarrs driver" should become "expose zarrs as a codec-pipeline
option on the zarr-python driver." No new abc surface is required; it changes
only which codecs run, which the metadata layer already describes. Do not
confuse it with **zarrista** (below), which shares the same Rust core but is
a full standalone driver.

### zarrita — **retired into zarr-python 3.x**

zarrita was the prototype that became zarr-python's v3 implementation. The
standalone package is legacy and the in-repo `drivers/zarrita.py` is a
5-line stub that just raises (with a copy-pasted wrong name) and is not even
registered. **Recommendation:** drop it, and drop `"zarrita"` from
`KnownDriver`. Anyone who wanted zarrita wants zarr-python 3.

### zarrista — **a real driver, and the best fit for both surfaces**

zarrista (developmentseed) is a *full standalone* Zarr implementation on the
same Rust `zarrs` core as zarrs-python — but with its own Python API, so
unlike zarrs-python it is a peer driver, not a codec option. Two properties
make it the strongest next candidate after the reference:

- **It splits sync from async at the class level.** `Array` / `Group` (sync,
  local filesystem) and `AsyncArray` / `AsyncGroup` (async, remote) are
  separate types — exactly abczarr's `ZarrArray` / `AsyncZarrArray` shape.
  tensorstore has one future-based class; zarr-python is async underneath a
  sync facade; zarrista is the only proposed backend that earns *both*
  surfaces natively, so it maps on with the least impedance and is the
  cleanest test of the sync/async parity design.
- **Its remote store is [obstore](https://github.com/developmentseed/obstore),
  not fsspec** — a Rust object-store binding (S3/GCS/Azure, ~9× fsspec
  throughput), plus Icechunk. This is the one backend whose store does *not*
  slot into the bagof-paths `PathStore`: for zarrista the abc store wraps its
  **native** obstore-backed store, which is exactly the "native store when a
  driver is active" path the store abc already provides. It validates that
  seam rather than straining it. (A future bagof-paths obstore driver would
  also let `PathStore` reach obstore, but that is not required here.)

Numeric arrays go through a `Tensor` numpy bridge; dtypes lean on
`ml_dtypes`. **Caveat:** zarrista is `v0.1.0-beta.1` — a candidate to design
for and track, not to implement yet. zarr-python stays the reference; when
zarrista stabilizes it is a better *second* driver than tensorstore, because
it earns the async surface honestly.

### The real driver set

| target | status | as a driver? |
|---|---|---|
| zarr-python | present, most complete | **yes** — reference driver |
| tensorstore | present, has store mapping | **yes** — async-capable |
| zarrista | standalone zarrs-core API, native sync+async, obstore/Icechunk | **yes** — strongest next candidate (beta) |
| zarrs-python | Rust codec pipeline for zarr-python | **no** — a codec option under zarr-python |
| zarrita | absorbed into zarr-python 3.x | **no** — retire the stub |
| *(fallback)* | dependency-free store over bagof-paths | **yes, eventually** — `PathStore` + a pure-Python chunk reader |

The bagof-paths `PathStore` is what makes the eventual dependency-free
fallback driver reachable: it already does the I/O, leaving only a
pure-Python chunk codec to synthesize a minimal read path with no backend
installed.

---

## 6. Capability vocabulary

`KNOWN_CAPABILITIES` (in `abc/node.py`) is the contract for `supports`. Align
it with what the backends actually advertise, and let the **store** answer
the I/O-shaped ones:

| capability | asked of | zarr-python | tensorstore | PathStore |
|---|---|---|---|---|
| `sharding` | node | ✓ | ✓ | — |
| `async` | node/store | ✓ (native) | ✓ (futures) | ✓ (AsyncPath) |
| `consolidated_metadata` | node | ✓ | — | — |
| `partial_read` | store | ✓ (`get_partial_values`) | ✓ | scheme-dependent |
| `partial_write` | store | scheme-dependent | ✓ | — |
| `codecs_v2` / `codecs_v3` | node | ✓ / ✓ | v3 | — |
| `listing` | store | ✓ | ✓ | ✓ |

Asking about an unknown capability returns `False` (already the behavior), so
a caller written against a newer vocabulary never crashes an older driver.

---

## 7. Recommended sequence

Design is settled enough to build in this order; each step is small and
testable, and none needs a backend to be installed except where noted.

1. **Turn `abc/store.py` into a real ABC** — the five primitives + `supports`,
   the synthesized members (`list_dir`, `clear`, `getsize`, …), the
   `AsyncStore` twin. Delete the empty `S3Store`/`GCSStore`/… subclasses.
   Fix the dangling `close` / `iter_keys` references.
2. **`PathStore` / `AsyncPathStore` over bagof-paths** — the default store,
   covering local + every fsspec/cloud scheme. Test with a memory path and a
   `tmp_path`; the async one against bagof-paths' in-process async backend.
3. **One surface spec + `test_parity.py`** — lock the node and store surfaces,
   sync vs async, abc vs driver.
4. **Re-fit the zarr-python driver** onto the store abc: real `metadata`,
   real `fill_value`, native `Store` wrapped as an abc store, capabilities
   declared. This is the reference and should be the first green driver.
5. **Re-fit the tensorstore driver**: `KvStore` wrapped, `.result()` in the
   sync path, `native` for rich indexing, groups-as-paths kept.
6. **Retire zarrita**; record zarrs-python as a codec-pipeline option on the
   zarr-python driver, and **zarrista** as a tracked driver candidate — a
   third driver (native sync + async, obstore-backed store) once it leaves
   beta.
7. **(Deferred)** the dependency-free fallback: a pure-Python chunk reader on
   top of `PathStore`.

The through-line: the abc surface is narrow and synthesized; bagof-paths does
the store I/O; each backend contributes primitives and reaches past the
surface through `native` and `supports`. That is what keeps adding a driver
cheap and keeps the shared surface from bloating into the union of every
backend.
