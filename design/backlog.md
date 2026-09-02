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
- [~] Config-based creation API design: `design/config-api.md` (Fable), branch
  `claude/design/config-api`. Awaiting review/approval before implementation.
- [x] bagof-hints adoption analysis filed as bagofseeds/bagof-hints#18
  (backburner).

## Config API (rests on approving design/config-api.md)

- [ ] Merge the two `ZarrArrayConfig`: a rich config object (`ArrayConfig`) plus
  a per-call options mapping (`ArrayOptions`, a `TypedDict`). `abc/array.py`
  stops defining a config. This merge is the crux and depends on the design.
- [ ] `ZarrConfig` base with `GroupConfig` / `ArrayConfig`; frozen, keyword-only.
- [ ] Creation surface: `create_array(name, shape, dtype, *, config=None,
  **options)`, effective config = `replace(config or ArrayConfig(), **options)`,
  keyword wins; a keyword contradicting a fact the target fixes is a `ValueError`.
- [ ] `from_config` becomes `abczarr.create(location, config)`, returning a group
  or an array by the node type the config lowers to (typing overloads).
- [ ] Driver contract: `Driver.create(location, metadata, *, overwrite)` +
  `can_create`; `ZarrGroup.create_array` becomes concrete, drivers implement
  `_create_array(name, metadata)`. Unblocks tensorstore array creation.
- [ ] Coarse-to-fine: `resolve()` turns "auto" into concrete values per version;
  `to_metadata()` builds v3 then `to_version(n, policy="strict")`. `-1` for a
  whole axis replaces `0`. Configs expose `plan()` (OME hook).
- [ ] `config=` also accepts the plain `dict` form of a config (zarr-python
  style), alongside a config object.
- [ ] Config objects implement `keys()` + `__getitem__` so `create(..., **config)`
  works. Lean: `keys()` yields only explicitly-set fields (a clean override),
  not every field. Naming guard: never add a config field named `keys`,
  `values`, or `items` (they would shadow the mapping protocol).
- [ ] Resolve an unspecified v3 fill value to a concrete dtype zero during
  creation (defect #4): the metadata model keeps `None` for reads and
  conversions, so the default belongs in `resolve()`/`to_metadata`, not the
  model.
- [ ] Open question settled by the design: `open` stays access-only (`mode`,
  `driver`); config attributes belong to `create`, not `open`.

## Defects found while probing the config design (real bugs, fix independently)

- [x] `auto_shard`/`auto_chunk` never terminate when one auto axis saturates
  while another is capped. Fixed in PR #44.
- [x] `auto_shard` drops the chunk byte budget (ignored the real itemsize).
  Fixed in PR #44.
- [x] `CompressorTypeV3` rejects `"zstd"`. Fixed in PR #44.
- [ ] `fill_value=None` reaches v3 metadata as JSON null. Moved to the config
  phase (resolve a concrete fill at creation; the model keeps None on purpose).
- [ ] `GeneralConfig.set_default_name` references a nonexistent `self.variant`
  (moot once `GeneralConfig` is removed).

## Documentation (Sonnet-written, no em dashes, no history/impl detail)

- [ ] Nav: "API Reference" to "Reference"; "Opening" to "API".
- [ ] "Nodes" to "ABC/Nodes", all node types on one page; "Stores" to
  "ABC/Stores".
- [ ] Metadata section: base, v1, v2, v3 as sub-pages, each with array/codecs/etc
  as sub-pages.
- [~] OME metadata page (missing today).
- [ ] Icons on Home and Reference menu items.
- [ ] A more Zarr-relevant logo (cube / voxel).
- [ ] Tutorial page.
- [ ] Document the `registry` module (exported and tested, no page today).

## Architecture / clarity

- [ ] `errors` placement: Fable recommends keeping `errors.py` in `abc/` (bottom
  of the import graph) with top-level re-exports; my earlier lean was a
  top-level `errors.py`. Decide, and consolidate `UnsupportedConversion` (now in
  metadata/base.py) with the rest.
- [ ] Rename the `support()`/`supports()` pair so the enum-vs-bool split is
  obvious (e.g. `capability()` returns `Support`, `supports()` returns `bool`).
- [ ] Clarify `StorePath`/`AsyncStorePath` (locations) vs `PathStore`/
  `AsyncPathStore` (stores): docs, and possibly a rename.
- [ ] Capabilities vocabulary: consolidate the coarse `KNOWN_CAPABILITIES` and
  the generated feature keys into one defined, validated, documented registry
  (a queried capability is currently unvalidated and silently returns NONE).
- [ ] Module layout: an `api/` package (private `_open.py` / `_create.py` /
  `_registry.py`), per the config design. `config.py` stays top level;
  `open.py` as a public name collides with the `open()` function.

## Features

- [ ] OME helpers in the ABC + pyramid construction. Write OME metadata as soon
  as the first level array exists; the next-level generator edits the existing
  OME metadata to append the new levels.
- [ ] Transactions extended to array/group writes (today they are store-key only;
  a user cannot write array chunks in a transaction).
- [ ] Remove the broken `zarrita` stub and its `KnownDriver` / `DRIVERS`
  references; implement a real `zarrista` driver.
- [ ] Verify and optimize dask `to_zarr` / `to_store` interop with the array
  classes.

## Cleanup

- [ ] Dead code: the `zarrita` stub, `config.GeneralConfig`, unused
  `_core/constants.py` constants (`DRIVERS`, and probably `FILE_MODES`,
  `LOG_LEVELS`, `COMPRESSORS_V2`, `COMPRESSORS_V3`, `ZARR_VERSIONS`,
  `OME_VERSIONS`). Note: `config.ZarrArrayConfig`, `OMEZarrConfig` are NOT dead;
  they get wired by the config redesign / OME work.

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
