# abczarr backlog

The durable task list. Committed and pushed so it survives session compaction
and the ephemeral container. Update it as work lands. Conventions at the
bottom are standing rules, not tasks.

Legend: `[ ]` open, `[~]` in progress, `[x]` done.

## In flight / recently landed

- [x] Path-based group recovered and generalized as `PathGroup` (abc/group.py),
  with `node_type_at` detection in the metadata layer and `TensorStoreGroup`
  as its first consumer. PR #43 (branch `claude/abczarr-roadmap-fo0yzr`).
- [x] Creation-path defects #1 to #3 fixed (sharding loops, itemsize budget,
  v3 zstd). PR #44. Defect #4 (fill value) moved to the config phase, below.
- [x] OME metadata documentation (usage-first overview + base + v0.5),
  Sonnet-written. PR #45. Nav wiring pending the docs restructure.
- [x] Config-based creation API: designed (`design/config-api.md`, Fable),
  approved, and landed across PR #46 (increments 1+2) and PR #47
  (write-through attrs).
- [x] bagof-hints adoption analysis filed as bagofseeds/bagof-hints#18
  (backburner).

### Session of 2026-09-02 (branch `claude/abczarr-roadmap-fo0yzr`)

- [x] Per-driver node bases: `ZarrPythonNode` (real dedup of the shared
  metadata/attrs/version accessors) and `TensorStoreNode` (common type so
  `open` returns one node kind). PR #48.
- [x] Dead code removed: the `zarrita` stub, the unused `_core/constants.py`
  tuples, and `"zarrita"` dropped from the `KnownDriver` Literal. PR #49.
  (`GeneralConfig` was already gone.)
- [x] Errors consolidated in `abc/errors.py` with top-level re-exports;
  `UnsupportedConversion` moved there, `report_loss` imports it lazily to
  avoid the abc<->metadata cycle. PR #50.
- [x] Capability query renamed `support()` -> `capability()` (returns
  `Support`); `supports()` (returns `bool`) unchanged. PR #52.
- [x] `ZarrArray.store(dask_array)`: a uniform, driver-independent dask
  write path (block by block via `da.store`), since `da.to_zarr` rejects
  the wrapper and only accepts a `zarr.Array` native. PR #53.
- [x] Docs nav restructure (zensical, not mkdocs) + tutorial + registry
  page; OME page wired into nav. Sonnet-written. PR #51.
- [x] `PathStore` -> `PathBasedStore` (and `AsyncPathBasedStore`); `StorePath`
  kept (matches zarr-python). README fixed too. PR #54.
- [x] `report_loss` / `node_at` / `node_type_at` made private. PR #55.
- [x] Metadata docs render class attributes now (`show_if_no_docstring = true`)
  + v2/v3 codecs, v2 filters, v3 extensions pages added to the nav. PR #56.
- [x] `to_dask(chunks=...)` (align to "chunks"/"shards"/explicit) and
  `store(lock="auto")` (locks only when the source's Dask blocks don't fall
  on whole chunks). PR #58.
- [x] `config` and `registry` moved under an `api/` package
  (`abczarr.api.config` / `abczarr.api.registry`); `api/__init__` is fully
  lazy (`__getattr__`) to keep the abc<->config graph cycle-free, with a
  TYPE_CHECKING block for the reference builder. PR #59.
- [~] OME `downsample_array` / `create_pyramid` draft (dask coarsen), no OME
  metadata write yet. PR #57 (DRAFT, awaiting review: free functions vs group
  methods, level-naming convention, then the OME multiscales metadata write).
- [x] Doc audit (Sonnet, read-only): findings applied -- README em dashes,
  the wrong top-level package docstring, and the OMEZarrConfig contributor
  note (in PRs #54/#56/#59). Report at scratchpad/doc-audit.md.

Standing note learned this session: the **Documentation** workflow runs only
on pushes to `main`, not on PRs, so build docs locally (`zensical build
--clean`) before merging anything that touches a docstring or a docs page.

## Config API design plan (all landed)

The full plan from `design/config-api.md` shipped across PR #46 and #47:
`ArrayConfig`/`GroupConfig`/`ArrayOptions`; the `create(location, config)`
surface returning array-or-group by config type; the `create_from_metadata`
driver primitive (also accepting a raw metadata object, not only a config);
`resolve()`/`to_metadata()` with the auto semantics and `-1`/`None` for a
whole axis; `config=` accepting a dict plus `**`-unpack; `keys()`/
`__getitem__` on configs; and `open` staying access-only. See the "Config
API" increments section below for the as-built record.

## Defects found while probing the config design (real bugs, fix independently)

- [x] `auto_shard`/`auto_chunk` never terminate when one auto axis saturates
  while another is capped. Fixed in PR #44.
- [x] `auto_shard` drops the chunk byte budget (ignored the real itemsize).
  Fixed in PR #44.
- [x] `CompressorTypeV3` rejects `"zstd"`. Fixed in PR #44.
- [x] `fill_value=None` reaching v3 metadata as JSON null: closed by the
  auto-fill resolution in the config phase, and verified (an unspecified
  fill resolves to a concrete dtype zero, e.g. `0.0`, not null).
- [x] `GeneralConfig.set_default_name` bug: moot, `GeneralConfig` is gone.

## Documentation (Sonnet-written, no em dashes, no history/impl detail)

- [x] Nav: "API Reference" to "Reference"; "Opening" to "API". PR #51.
- [x] "Nodes" to "ABC/Nodes", all node types on one page; "Stores" to
  "ABC/Stores". PR #51.
- [x] Metadata section: base, v1, v2, v3 as sub-pages (v2/v3 with a dtypes
  sub-page; codecs/filters/extensions have no class docstrings, so no stub
  pages). PR #51.
- [x] OME metadata page written (PR #45) and wired into the nav (PR #51).
- [ ] Icons on Home and Reference menu items.
- [ ] A more Zarr-relevant logo (cube / voxel). (Needs an image asset.)
- [x] Tutorial page. PR #51.
- [x] Document the `registry` module. PR #51.

## PR #43 review (path-based group)

- [x] #4 needless string annotation on `node_type_at` removed.
- [x] #5 detection is version-aware (`node_at` returns type + version, no
  guess) and a `PathGroup` only sees children of its own version.
- [x] #2 tensorstore `_create_array` -> done in config increment 2 (PR #46);
  tensorstore creates arrays via native `ts.open(create=True)`.
- [x] #1 write-through `attrs`: wired the write-through `Attributes` into the
  node contract; zarr-python delegates to its live attrs, tensorstore uses
  the metadata-file mapping. PR #47.
- [x] #3 per-driver node bases: `ZarrPythonNode` and `TensorStoreNode`; `open`
  returns the driver node type. PR #48.

## Config API

- [x] Increment 1: ArrayConfig/GroupConfig/ArrayOptions merge, resolve() /
  to_metadata with the auto semantics and dict/** unpack.
- [x] Increment 2: `create(location, config)`, the `Driver.create(metadata)`
  primitive, `ZarrGroup.create_array` concrete over `_create_array(metadata)`,
  and tensorstore array creation. The whole slice is PR #46 (green, 305 tests).
- [x] Serialization-conformance fixes surfaced by making metadata
  backend-writable: nested metadata now serializes through its own `to_dict`
  (core v3 data_type is a bare string), and v3 omits unset `dimension_names` /
  `storage_transformers`. Both in PR #46.
- [x] Defect #4 (v3 fill value) closed by the auto-fill resolution; verified.

## Architecture / clarity

- [x] `errors` placement: DECIDED (abc/errors.py + top-level re-exports) and
  landed. `UnsupportedConversion` consolidated there. PR #50.
- [x] Rename the `support()`/`supports()` pair: DECIDED and landed as
  `capability()` (returns `Support`) / `supports()` (returns `bool`). PR #52.
- [ ] Clarify `StorePath`/`AsyncStorePath` (locations) vs `PathStore`/
  `AsyncPathStore` (stores): docs, and possibly a rename. (User priority #1 of
  the remaining larger items; do after the docs restructure so the docs part
  does not conflict.)
- [ ] Capabilities vocabulary: consolidate the coarse `KNOWN_CAPABILITIES` and
  the generated feature keys into one defined, validated, documented registry
  (a queried capability is currently unvalidated and silently returns NONE).
- [ ] Module layout: an `api/` package (private `_open.py` / `_create.py` /
  `_registry.py`), per the config design. `config.py` stays top level;
  `open.py` as a public name collides with the `open()` function. (User
  priority #2. Note: moving `registry.py` would break the new registry doc
  page from PR #51, so update that page in the same change.)

## Features

- [ ] OME helpers in the ABC + pyramid construction. Write OME metadata as soon
  as the first level array exists; the next-level generator edits the existing
  OME metadata to append the new levels. (User priority #3.)
- [ ] Transactions extended to array/group writes (today they are store-key only;
  a user cannot write array chunks in a transaction). (User priority #4;
  entangled with the StorePath/PathStore clarity item.)
- [ ] Implement a real `zarrista` driver. (The broken `zarrita` stub and its
  dead `DRIVERS`/`KnownDriver` references were removed in PR #49.)
- [x] Dask write interop verified and closed: `da.store(darr, array)` works for
  every driver; `da.to_zarr` only accepts a `zarr.Array` native (zarr-python
  only). Added `ZarrArray.store(dask_array)` as the uniform path. PR #53.

## Cleanup

- [x] Dead code removed (PR #49): the `zarrita` stub and the unused
  `_core/constants.py` tuples. `GeneralConfig` was already gone. Note:
  `OMEZarrConfig` is NOT dead; it gets wired by the OME work.

## Typing

- [ ] Adopt bagof-hints for the generic hints, only if we intend to consume it
  more broadly (its protocols, array types). Clean drop-ins: `OneOrIter`,
  `OneOrSeq`, `BuiltinSequence`. Gaps tracked in bagof-hints#18.

## Standing conventions (not tasks)

- One branch and PR per feature: `claude/<type>/<desc>`.
- Documentation is written by a Sonnet agent, not inline. No em dashes.
- Fix the upstream module, not the caller (no if-case hacks).
- The driver registry lives in `registry.py`; driver methods mirror the `api`
  surface names.
- Python 3.8 floor; `import typing_extensions as tx` only.
