# abczarr development roadmap

Status: **pre-release** (the package advertises "DO NOT USE YET"). Python
floor 3.8. This document is the plan for taking abczarr from its current
mid-refactor state to a coherent, tested, multi-backend Zarr I/O library
with native OME-Zarr support and lossless all-versions metadata.

A designed, shareable version of this plan is committed alongside as
[`roadmap.html`](./roadmap.html).

Findings below were verified against the repository tree, not from memory.

---

## 0. The short version

Five decisions carry the whole plan:

1. **Replace `_core/path.py` with `bagof-paths` — yes, early.** It is a
   2,375-line reimplementation of bagof-paths that is currently broken and
   whose cloud surface was never wired up.
2. **Move attrs → `bagof-magic` — yes, but staged, after the tree is
   stable.** The version lattice hand-rolled in `_core/metadata.py` *is*
   bagof-magic's `polymorphic`.
3. **Own the format, borrow the I/O.** Metadata parsing and cross-version
   conversion is the product; chunk I/O, codecs and compression belong to
   the backends.
4. **Sync-first public surface**, native-async where the backend earns it;
   defer the hard sync-over-async direction. Same shape as bagof-paths.
5. **One coherent abstract layer** with a thin driver contract, a
   first-class `.native` escape hatch, and capability queries — so the
   common denominator never hides a backend's strengths.

---

## 1. Where it stands today

### Blocker: the tree does not import

The `[WIP] metapath` commit renamed the path classes (`S3Path` →
`WrappedS3Path`) but `abc/path.py` still references the old names, and
`__init__.py` imports everything eagerly. So **every `import abczarr…`
raises** — even the passing metadata and OME tests can no longer be
collected. Nothing ships until this is fixed (Phase 0).

### Subsystem status

| Subsystem | What it is | Size | Status |
| --- | --- | --- | --- |
| `metadata/` (v1·v2·v3) | Frozen attrs model of zarr metadata, codecs, dtypes, filters | ~2k ln | parse tested, rest untested |
| `schemas/` (v1·v2·v3) | TypedDict shapes validated by a type-walking `get_validator` | ~0.6k | only OME rc0 exercised |
| `ome/metadata` (v0.1–v0.6) | attrs model of OME-NGFF per spec version | large | `to_version` broken, 0 tests |
| `ome/schemas` (…v0.6rc0) | TypedDict validators per OME version | large | only v0.6rc0 tested |
| `_core/path.py` | Home-grown sync+async wrapper over pathlib/UPath/cloudpathlib | 2,375 | WIP, cloud surface unwired |
| `_core/auto/*` | Type-hint–driven attrs converters/validators/factories | ~3k | works, duplicates bagof-* |
| `abc/` + `_abc.py` | Two parallel ABC layers | ~0.7k | duplicated, import-broken |
| `drivers/zarr_python` | zarr-python backend | 349 | most complete, untested |
| `drivers/tensorstore` | TensorStore backend incl. kvstore↔store mapping | 813 | largest, untested |
| `drivers/zarrita` | — | 5 | stub, raises on import |
| `config`/`api`/`registry` | ZarrConfig, `open_*` façade, driver registry | ~0.6k | plausible, untested |

Two facts drive the plan. **First:** ~5,000 lines of `_core` reimplement
libraries that already exist in the same org (bagof-paths, and the
bagof-converters/validators/factories/magic family). **Second:** the
genuinely valuable, hard-to-buy part — a superset metadata model that reads
every zarr and OME version — is real but almost entirely untested, and its
conversion paths are where the bugs live.

---

## 2. The two big forks

### A · The path layer → bagof-paths — **replace it, Phase 1**

`_core/path.py` and bagof-paths solve the identical problem: one uniform,
driver-pluggable, sync + async surface over pathlib, universal-pathlib and
cloudpathlib, with S3/GCS/Azure/HTTP protocols and a driver registry.
abczarr's version is 2,375 lines, mid-refactor, and its per-protocol cloud
classes are referenced but never defined — so in practice only the local
`Path` subset is exercised today.

- **Why it fits:** bagof-paths ships sync `Path` + async `AsyncPath` over
  one spec table (the paired hierarchy `abc/store.py` needs); a
  native-async fsspec cloud driver with per-loop session handling;
  credentials via `storage_options`; and a `path.wrapped` escape hatch.
- **Migration cost:** low at the call sites (`_core/attributes.py` and the
  drivers use only the vanilla `Path` subset); concentrated at one seam
  (`abc/path.py` / `abc/store.py` subclass the wrapper and use its
  `register_subclass` hooks). Model `StorePath`'s extra `read_only` state as
  a thin subclass or via `with_wrapped` derivation.
- **Before committing:** confirm bagof-paths exposes a *subclassable*
  `Path`/`AsyncPath` and a store-flavoured derivation hook. Its
  `register_protocol` / `register_driver` model is the replacement for
  `register_subclass`.

### B · attrs vs. bagof-magic — **migrate, staged, after Phases 0–3**

- **The case for:** `register_subclass(zarr_format=2, node_type="array")`
  dispatch *is* bagof-magic's `polymorphic=True` +
  `on={"zarr_format":2, "node_type":"array"}`. `_core/auto` derives
  converters/validators/factories from type hints — exactly bagof-magic's
  premise — and is ~3,000 lines maintained untested here.
- **Risks to retire first:** (1) RFC-2119 markers in `_core/rfc2119.py` must
  re-home onto bagof-factories' annotation model; (2) numpy-DType coercion,
  JSON round-trip, and the regex dtype match (`r\d+`) must port to
  bagof-converters or ride as explicit `ConvertTo` callables; (3) the
  TypedDict validation in `schemas/` must be confirmed subsumed by
  bagof-validators (keep the surface until then).
- **Sequencing:** migrate innermost-first (`ZarrConfig` → metadata base →
  array/codec/dtype → OME) behind shim decorators, then delete `_core/auto`
  when nothing imports it. If bagof-magic isn't ready for the DType/RFC-2119
  cases, that is a reason to *delay*, not to keep extending the home-grown
  engine.

---

## 3. Designing the generic "abc" API

Borrow bagof-paths' architecture: describe the surface once, keep drivers
thin, never let the common denominator hide a backend's strengths.

- **Consolidate to one layer.** Delete the old flat `_abc.py`; keep the
  `abc/` package (`node`, `array`, `group`, `store`, `path`). Two parallel
  definitions of `ZarrArray` is how the import break slipped in.
- **One surface spec, like `_spec.py`.** Describe each member once — its
  kind (pure vs. I/O), how its result is wrapped, whether it has an async
  twin — instead of writing sixty method bodies twice.
- **Thin driver contract.** A driver supplies only primitives: `open`,
  `metadata`, read/write chunk, create, list, delete. Everything richer is
  synthesized or capability-gated. That makes a new backend a small file.
- **Metadata first-class on every node:** `.metadata` (rich attrs model),
  `.attrs` (user attributes), `.native` (raw `zarr.Array` / `ts.TensorStore`).

Two contracts keep the abstraction honest:

- **Escape hatch:** `node.native` is a documented, supported contract (not
  an accident of `__getattr__` delegation). Wrapping never costs a capability.
- **Capability query:** `driver.supports("sharding" | "async" |
  "partial-write" | "consolidated-metadata")`, answered from the driver
  *class*, never by touching a live resource.
- **One named error:** when a member can be neither delegated nor
  synthesized, raise a single `UnsupportedZarrOperation` naming the
  operation and driver (the bagof-paths `UnsupportedPathOperation` pattern).
  Never a bare `NotImplementedError`, never a leaked internal name.

---

## 4. Handling sync and async

Backends disagree: tensorstore is async at heart (futures + sync
`.result()`), zarr-python has a real async layer under its sync one,
zarrs/zarrita differ again. Writing each method twice by hand is how the two
surfaces drift. Copy bagof-paths' shape:

| Node ↔ driver | Strategy |
| --- | --- |
| async node, sync driver | bridge into a worker thread (`run_in_executor`, already sketched in `_core/asyncutils.py`) |
| async node, async driver | await the backend's coroutine directly through a native seam (tensorstore/zarr-python fast path) |
| sync node, async driver | **defer** — the sync-over-async "portal" is the hard direction; bagof-paths defers it too |

- **Sync-first public surface.** Consumers are numpy and dask (synchronous);
  make `ZarrArray` sync-primary and offer `AsyncZarrArray` where the backend
  earns it. Enforce parity with a test that fails the moment signatures
  drift (bagof-paths' `test_parity.py`).
- **Async matters most at I/O boundaries** — the store (handled by
  bagof-paths) and chunk read/write. The metadata model stays pure-sync
  data.
- **Fix the loop handling.** `asyncutils` uses the deprecated
  `get_event_loop()`; move to `get_running_loop()` + an explicit executor.

---

## 5. Not disabling a backend's good features

The danger of a lowest-common-denominator ABC: it hides tensorstore's async
concurrency, zarr-python's codec pipeline, sharding, partial reads,
consolidated metadata. Three-tier discipline prevents that:

1. **Delegate.** If the driver implements the member, call it and pass rich
   arguments straight through. Keep `chunk_grid`, `codecs`, sharding and
   dtype config *in the metadata model* so a feature expressed there
   survives even when a convenience API wouldn't surface it.
2. **Synthesize** — only from more-primitive members the driver *does* have:
   `__array__`/`to_dask` from `__getitem__`, group `walk` from `iterdir`,
   whole-array read from chunked reads, pyramid levels from
   `create_array`+`getitem`.
3. **Raise** — one `UnsupportedZarrOperation`; never half-emulate
   compression, a sharding index, or a codec.

Paired with `node.native` and `driver.supports(…)`, there are three ways to
reach a backend strength: it's in the uniform surface, you branch on the
capability, or you drop to `.native`. No feature is stranded by the wrapper.

---

## 6. How many fallbacks to own

Rule of thumb: **own the format logic and the cross-backend glue; borrow I/O
and compute; synthesize only from primitives; otherwise refuse.**

| Responsibility | Stance | Why |
| --- | --- | --- |
| Metadata parse/serialize/**cross-version convert** | own | Backend-independent spec logic — the core value |
| Chunk/shard auto-sizing, pyramid downsampling, chunk math | own | Pure numeric logic (`_core/sharding.py`, `pyramid.py`), driver-independent |
| Path & store abstraction, protocol selection, credentials | delegate to bagof-paths | Solved once already in the org |
| `to_dask`, `__array__`, group `walk`, whole-array read/write | synthesize from primitives | Trivially derived from `__getitem__`/`iterdir` |
| Chunk I/O, compression, codec encode/decode, sharding index | delegate to backend | What zarr-python/tensorstore/zarrs are *for*; reimplementing invites corruption |
| A capability a driver simply lacks | raise, don't emulate | Half-built emulation is worse than an honest refusal |

Litmus test: if a fallback can be written purely in terms of members the
driver already exposes, synthesize it; if it needs to re-encode bytes or
re-implement a compression/sharding format, it belongs to the backend.

---

## 7. All-versions metadata & lossless conversion

The feature you singled out, and the one with the most latent bugs. The
model exists (zarr v1/v2/v3, OME v0.1–v0.6) but the conversion paths are
largely untested.

### What's there, and what's wrong

- **zarr v2↔v3 conversion exists but is untested** — per-leaf `to_version()`
  on codecs/dtypes/filters plus array-level `_to_v2`/`_to_v3`. *(verify)*
- **`v3→v1` looks buggy** — the compressor is double-converted and a
  recomputed dtype is discarded. *(fix)*
- **`v1→v2/v3` is absent** — the v1 array class has no `to_version`.
  *(implement)*
- **OME `to_version` is a broken stub** — it splits `__qualname__` expecting
  a module path that isn't there, targets the wrong subpackage, has no
  callers and no tests. *(replace with explicit per-version migration)*
- **No OME↔zarr-format coupling is encoded.** OME v0.1–v0.4 sit on zarr v2;
  v0.5+ sit on zarr v3 (metadata nested under an `ome` key). Today that
  lives only in which submodules exist. *(add an explicit table)*

### A conversion contract worth committing to

- **Parse into a superset that keeps unknowns.** `FlexibleMetadata`'s
  `extra_items` already retains unnamed fields; lean on it so nothing read
  is silently dropped.
- **Convert by explicit field maps, not reflection.** One converter per
  version pair, hand-written where the spec changed shape (axes,
  transformations, the `ome` nesting at v0.5). Reflection over class names
  can't know that `multiscales` moved.
- **Make "lossy" a choice, not an accident.** A policy flag: `strict`
  (raise if a field can't be represented), `annotate` (stash it in a
  namespaced extra), or `lossy` (warn + drop). Pass a value through, but
  never invent one.
- **Prove it with round-trips.** Property test: for every version pair the
  spec allows, `a → b → a` equals the original.

Keep both the TypedDict `schemas/` (declarative validation of raw JSON) and
the attrs `metadata/` (functional model that converts) — they are
complementary — but add the missing tests (every zarr version, every OME
version, and the negative cases currently commented out).

---

## 8. The sequence

Seven phases, ordered so each unblocks the next. Phase 0 is non-negotiable
and small; the dependency swaps come before the API hardening; the big attrs
migration comes only once the model is stable and tested.

### Phase 0 — Unblock & guard *(blocker)*
- Fix the `abc/path.py` ↔ `_core/path.py` name mismatch (or leapfrog into Phase 1).
- Make `__init__.py` imports lazy/guarded so a missing backend or WIP module can't take down the package.
- Add CI that actually *imports* abczarr and runs the suite on Python 3.8 **and** current.
- **Exit:** green suite, package imports on both interpreters.

### Phase 1 — Adopt bagof-paths
- Rewrite `abc/path.py` + `abc/store.py` onto bagof-paths' `Path`/`AsyncPath` and `register_protocol`.
- Point `_core/attributes.py` and the drivers at the new path type.
- Carry `read_only` as store state; keep `.wrapped` for backend-specific needs.
- **Exit:** 2,375 lines gone; local + one cloud protocol tested end-to-end.

### Phase 2 — Consolidate the abc API
- Remove `_abc.py`; make `abc/` the single source of truth.
- Introduce the surface spec table, the `node.native` contract, `driver.supports(…)`, and `UnsupportedZarrOperation`.
- Settle the sync/async model and add the parity test.
- **Exit:** a documented driver contract a new backend can target.

### Phase 3 — Harden metadata & conversion *(core value)*
- Fix `v3→v1`; implement `v1→v2/v3`; add the strict/annotate/lossy policy.
- Replace OME `to_version` with explicit per-version migrations; add the OME↔zarr-format table.
- Round-trip property tests across every version pair; re-enable the negative validation tests.
- **Exit:** every advertised version parses, converts, and round-trips under test.

### Phase 4 — attrs → bagof-magic *(staged)*
- Verify RFC-2119 markers, DType/JSON converters, and TypedDict validation against the bags first.
- Migrate innermost-first behind shim decorators; delete `_core/auto` when nothing imports it.
- Express the version lattice with `polymorphic` + `on={…}`.
- **Exit:** ~3k lines of home-grown engine replaced; behavior unchanged under the Phase-3 tests.

### Phase 5 — Finish the drivers
- Complete zarr-python (`fill_value`, groups) and test tensorstore against the contract.
- Implement the zarrita and zarrs drivers on the thin contract from Phase 2.
- Publish a capability matrix (driver × feature) driven by `supports()`.
- **Exit:** the same test suite passes against every installed backend.

### Phase 6 — Docs, polish, release
- Runnable `pycon` docstring tests on the 3.8 floor (the bagof house convention).
- A "lead with what the user writes" guide; the capability matrix; a conversion-fidelity page.
- Drop the "DO NOT USE YET" once Phases 0–5 hold.
- **Exit:** a first tagged release.

---

## 9. Open questions to settle first

- **Does bagof-paths expose a subclassable store seam?** If `Path`/`AsyncPath`
  can't be subclassed for `StorePath`, model the store as a wrapper holding a
  path instead. Settles Phase-1 shape.
- **Is bagof-magic ready for the DType & RFC-2119 cases?** If not, Phase 4
  waits — and that's fine. Freeze `_core/auto` in the meantime; don't extend
  it.
- **Which OME versions are in scope for v1?** Full v0.1–v0.6 round-tripping is
  a large test matrix. A defensible cut: v0.4 & v0.5 lossless, older ones
  read-only.
- **Sync-primary or async-primary internals?** This plan recommends
  sync-primary (numpy/dask consumers). Revisit only if the dominant use case
  is high-concurrency cloud I/O.
