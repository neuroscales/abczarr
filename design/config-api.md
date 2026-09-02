# The config-based creation API

A design proposal, not an implementation. It answers the five questions the
maintainer asked about wiring the config classes into creation, and it
records what the neighbouring libraries do, checked by running them.

Versions checked, in a fresh venv: zarr-python 3.1.6, tensorstore 0.1.85,
xarray 2026.7.0, dask 2026.8.0, attrs 26.1.0, pydantic 2.13.5. Every claim
about a neighbour below was verified by calling the library, not recalled.

## 0. The short version

1. **Two creatable configs, one base.** `ZarrConfig` becomes the abstract
   base holding the store-level choices (`zarr_version`, `overwrite`,
   `driver`, `attributes`). `GroupConfig(ZarrConfig)` and
   `ArrayConfig(ZarrConfig)` are what a user builds. Frozen, keyword-only.
2. **A config lowers to metadata, and drivers create from metadata.** Each
   config has `to_metadata()`, which returns the version-correct
   `GroupMetadata` or `ArrayMetadata`. The driver primitive becomes
   `Driver.create(location, metadata, *, overwrite)`, and the group hook
   becomes `_create_array(name, metadata)`. Both backends can create from a
   full metadata document (verified for tensorstore; zarr-python has
   `AsyncArray.from_dict` and a legacy `create(codecs=...)`). This is what
   unblocks array creation on tensorstore, which raises today.
3. **`from_config` becomes `abczarr.create(location, config)`.** There is no
   isinstance ladder: the config lowers to metadata, and the metadata's
   `node_type` says whether a group or an array is created. Typing overloads
   give `ArrayConfig -> ZarrArray`, `GroupConfig -> ZarrGroup`.
4. **Call surface: config object plus keyword overrides, keyword wins.**
   `create_array(name, shape, dtype, *, config=None, **options)`. The
   effective config is `replace(config or ArrayConfig(), **options)`. The
   keywords are exactly the config's fields, typed through a TypedDict that
   mirrors the config and is held in step by a test. A keyword that
   contradicts a fact fixed by the target (a v3 group asked for a v2 array)
   is a `ValueError`, not an override.
5. **`open*` does not grow creation keywords.** Opening describes access
   (`mode`, `driver`); creation describes content. They stay separate
   functions.
6. **The name collision is resolved by concept.** The rich object is
   `ArrayConfig`; the per-call keyword mapping is `ArrayOptions`. Both are
   defined in `abczarr/config.py`, beside each other, so they cannot drift
   apart unnoticed. `abc/array.py` stops defining a config.
7. **Layout: an `api/` package of private modules; `config.py` stays at the
   top level; `errors.py` stays in `abc/`.** `api/_open.py`, `api/_create.py`,
   `api/_registry.py`, re-exported from `abczarr.api` and `abczarr`. Config
   cannot live under `api/` because `abc/group.py` must import it. Errors
   stay at the bottom of the import graph, where every layer already reaches
   them; "broadly reachable" is served by the top-level re-exports.
8. **The smart part is `resolve()`.** It turns `"auto"` chunks, shards,
   compressor, fill value and separator into concrete values from shape,
   dtype, a byte budget and the target Zarr version. OME pyramids plug in as
   `OMEImageConfig`, whose `plan()` yields one group plus one array config
   per level, each resolved by the same code.

## 1. What exists today, and what is wrong with it

| Piece | State |
|---|---|
| `config.ZarrConfig` | Consumed only by `api.from_config`, which creates a group. |
| `config.ZarrArrayConfig` | Rich object with `finalize()` and `to_metadata()`. Wired to nothing. |
| `abc.array.ZarrArrayConfig` | A `TypedDict` of seven per-call keys. This is what `ZarrGroup.create_array(config=..., **kwargs)` threads through, and what the zarr-python driver maps to `zarr` keywords. |
| `config.OMEZarrConfig`, `config.GeneralConfig` | Unwired. |
| `Driver.create_group` | The only driver-level creation primitive. Takes `zarr_version` and `overwrite`, not metadata. |
| tensorstore driver | `PathGroup._create_array` raises. |

Two objects called `ZarrArrayConfig` with different shapes, different key
spellings (`chunk` versus `chunks`, `compressor_options` versus
`compressor_opt`) and different reachability. The one users can reach from
`abczarr.config` does nothing; the one that works is a mapping hidden in
`abc`.

Defects found while probing the rich config, all in the resolver it depends
on. None of them is this design's job, but the design assumes they are
fixed:

- `_core/sharding.py`: `auto_shard` never terminates for
  `shape=(100, 2048, 2048)`, `dtype="uint16"`, `chunks=(1, "auto", "auto")`,
  `shards="auto"`, `max_shard_bytes=64 MiB`. Once one `"auto"` axis has
  reached the array extent while another is capped by the byte budget, the
  no-op doubling of the full axis still sets `improved`, and the loop never
  breaks. `auto_chunk` has the same loop shape and the same hazard.
- `auto_shard` calls `auto_chunk` without `itemsize` or `maxsize`, so under
  sharding the chunk budget is silently the defaults (4 bytes, 8 MiB).
- `tz.CompressorTypeV3` is `Literal["blosc", "gzip", "none"]`, so
  `ZarrArrayConfig(compressor="zstd")` raises, while the TypedDict path
  accepts `"zstd"` and zarr-python 3 writes zstd by default.
- `fill_value=None` lowers to v3 metadata with `"fill_value": null`, which
  the v3 spec forbids.
- `GeneralConfig.set_default_name` reads `self.variant`, an attribute that
  does not exist.

## 2. What the neighbours do

Read this table before any of the judgement calls below; each row was
produced by running the library.

| Library | Creation surface | Config object and keywords | "auto" sizing |
|---|---|---|---|
| zarr-python 3.1.6 | `create_array(store, *, shape, dtype, chunks="auto", shards=None, filters="auto", compressors="auto", serializer="auto", fill_value, order, zarr_format=3, attributes, chunk_key_encoding, dimension_names, overwrite, config=None, ...)`. All keyword-only after `store`. | `config=` is `ArrayConfig | dict` and holds **runtime** choices only (`order`, `write_empty_chunks`), filled from the global `zarr.config`. Metadata choices are keywords. Passing `order=` both ways warns and deprecates the keyword; the object wins. | `"auto"` is a per-knob sentinel. `_guess_chunks` (from h5py) targets 128 KiB to 64 MiB; with shards, chunks aim at 1 MiB and the shard budget comes from the global `array.target_shard_size_bytes`. `zarr.open(store, mode="a", shape=...)` creates an array; without `shape` it creates a group. |
| tensorstore 0.1.85 | `ts.open(spec, *, create, open, delete_existing, dtype, shape, chunk_layout, codec, fill_value, schema, ...)`. | Spec (JSON or `ts.Spec`) plus keywords. A keyword that **conflicts** with the spec raises `ValueError: Specified dtype (float32) does not match existing value (uint8)`; keywords add constraints, never override. `ts.open({"driver": "zarr3", "kvstore": ..., "metadata": <full zarr.json>}, create=True)` creates from a complete metadata document (verified). | `ChunkLayout(chunk_elements=2**20, chunk_aspect_ratio=[1, 1, 0.25])` is a soft constraint resolved to a concrete grid ((161, 161, 40) for uint16); `write_chunk_shape` plus `read_chunk_shape` yields `sharding_indexed`. |
| xarray 2026.7 | `Dataset.to_zarr(store, mode, group, encoding={var: {...}}, zarr_format, ...)`; `open_zarr(store, ..., **kwargs)`. | Keywords, plus one mapping per variable. `encoding=` passed to the call overrides `variable.encoding` stored on the object (verified: stored `(5, 5)`, call `(2, 2)`, written `(2, 2)`). | Delegates to dask. |
| dask 2026.8 | `normalize_chunks(chunks, shape, limit=None, dtype=None)` | Global default `array.chunk-size` (128 MiB), per-call `limit`. | `"auto"` per axis, `-1`/`None` for the whole axis, a mapping `{axis: spec}`. |
| attrs 26.1 | `@define(frozen=..., kw_only=...)` class options; `evolve(inst, **changes)`; `asdict`. | Options are class-level; instances are updated by building a new one. | n/a |
| pydantic 2.13 | `model_config = ConfigDict(...)`; `model_copy(update={...})`; `model_validate(dict)`; `model_dump(exclude_unset=True)`. | Same shape as attrs, plus tracking of which fields were set explicitly. | n/a |

Three things follow from the table and are used below.

- **Object plus keyword override, with the keyword winning, is the
  consensus** for a reusable object: xarray's `encoding=`, `attrs.evolve`,
  `model_copy(update=)`. zarr-python's warn-and-deprecate is what happens
  when two spellings of one choice have no stated rule; the lesson is to
  state one rule, once.
- **Conflict-raises is right for facts, not for templates.** tensorstore's
  rule protects a Spec that describes an existing store. The same rule
  applies here to a fact the target already fixes (a group's format
  version, its driver), and not to a template's defaults.
- **"auto" resolved from a byte budget is standard**, and each library has a
  different budget. Ours is a field on the config with a sane default, not a
  constant.

## 3. The type hierarchy and the names

### 3.1 The name collision

Two legitimate concepts, two names, one module.

| Concept | Name | Kind | Lives in |
|---|---|---|---|
| A reusable, validated, resolvable description of an array to create | `ArrayConfig` | frozen attrs class | `abczarr/config.py` |
| The keyword arguments `create_array` accepts, for typing `**options` | `ArrayOptions` | `TypedDict(total=False)` | `abczarr/config.py` |
| Same pair for a group | `GroupConfig`, `GroupOptions` | | `abczarr/config.py` |
| The store-level fields both share | `ZarrConfig`, `ZarrOptions` | | `abczarr/config.py` |

The `Zarr` prefix is dropped from the two concrete names because they are
reached as `abczarr.ArrayConfig` and `abczarr.GroupConfig`; `ZarrConfig`
keeps its name because it is the shared base and the name the tests already
use. "Options" is the word zarr-python (`storage_options`), xarray
(`backend_kwargs`) and tensorstore (`ts.open(**kwargs)`) use for the loose
keyword bag; "config" is what all of them call the object.

`abc/array.py` stops defining any config. It imports `ArrayConfig` and
`ArrayOptions` from `abczarr.config` for the `create_array` signature. The
TypedDict and the class are declared next to each other, and a test asserts
that `ArrayOptions.__annotations__` equals the config's field names minus
`shape` and `dtype`, so the two cannot drift.

The package is pre-release ("not ready for use yet"), so the renames need no
deprecation shims: update the two test files that name `ZarrConfig` and the
TypedDict keys, and move on.

### 3.2 The hierarchy

```
ZarrConfig                 (base; not created on its own)
  zarr_version : 1 | 2 | 3 = 3
  overwrite    : bool = False
  driver       : str | None = None       (None: pick one that can write this)
  attributes   : Mapping[str, JSON] = {}

GroupConfig(ZarrConfig)    -> GroupMetadata
  (nothing more)

ArrayConfig(ZarrConfig)    -> ArrayMetadata
  shape              : Shape | None = None        (required by the time it is lowered)
  dtype              : DTypeLike | None = None    (same)
  dimension_names    : tuple[str | None, ...] | None = None
  chunks             : ChunkSpec = "auto"
  shards             : ChunkSpec | None = None
  max_chunk_bytes    : int = 8 MiB
  max_shard_bytes    : int = 2 GiB
  compression_ratio  : float = 1.8
  compressor         : str | Mapping | None | "auto" = "auto"
  compressor_options : Mapping[str, JSON] = {}
  filters            : tuple[Mapping, ...] = ()
  fill_value         : Number | None | "auto" = "auto"
  order              : "C" | "F" = "C"
  dimension_separator: "/" | "." | "auto" = "auto"

OMEImageConfig(ArrayConfig)   (later; see section 6.4)
```

Decisions inside the table, and why:

- **Frozen and keyword-only.** `@autofrozen(kw_only=True)` today; the same
  declaration ports to `class ArrayConfig(ZarrConfig, frozen=True,
  kw_only=True)` under bagof-magic when the roadmap's migration lands. A
  config is a template shared across calls; `resolve()` and the keyword
  merge return new instances (`evolve` today, `replace` under magic), so a
  call can never mutate the template it was handed. Metadata classes are
  already `autofrozen`; the config follows them.
- **`shape` and `dtype` are optional on the object and required at
  lowering.** A template like `ArrayConfig(compressor="zstd",
  max_chunk_bytes=4 * MiB)` has no shape yet. `to_metadata()` raises
  `ValueError("ArrayConfig.shape is required to create an array")` when
  either is missing. zarr-python's `create_array` has the same rule
  (`shape=None, dtype=None, data=None`: give `data`, or both of the others).
- **`driver` defaults to `None`, not `"zarr-python"`.** `None` means: lower
  the config to metadata, then pick the first driver whose `can_create`
  verdict passes. That is `select_driver` reused for creation, and it is how
  `open` already chooses. A hardcoded default silently fails on an install
  that only has tensorstore.
- **`"auto"` means "the target version's convention".** `compressor="auto"`
  is zstd (zarr-python's default for both v2 and v3 in 3.x; tensorstore
  reads it), `fill_value="auto"` is the dtype's zero on v3 and null on v2,
  `dimension_separator="auto"` is `/` on v3 and `.` on v2 (zarr-python and
  tensorstore both write `.` for v2). One config then lowers correctly to
  either version. `compressor=None` means no compressor (the `"none"`
  string is accepted and normalised to `None`). A mapping
  (`{"name": "zstd", "configuration": {...}}`) passes through untouched, for
  a codec the coarse vocabulary does not name.
- **`chunks` per-axis vocabulary follows dask.** An int, `"auto"`, or `-1`
  for the whole axis, in a sequence or in a mapping keyed by dimension name.
  The current code uses `0` for "whole axis"; `0` is a valid-looking chunk
  size and a silent trap, and dask, the only neighbour with a per-axis
  spelling, uses `-1`. A short sequence is right-padded by repeating its
  last entry, as today.
- **`names` becomes `dimension_names`**, the v3 metadata field and
  zarr-python's keyword. It also lets a chunk mapping be keyed by name.
- **`compressor_opt` becomes `compressor_options`**, the spelling the
  working TypedDict already uses.
- **`filters` is a passthrough.** v2 filters, or v3 array-to-array codecs,
  as codec mappings. No coarse vocabulary is offered for them yet.
- **`compression_ratio` is exposed** because it is an assumption the byte
  budget depends on and the user often knows better (a label volume
  compresses far more than 1.8x). It was a hidden constant.
- **Byte-budget defaults stay at 8 MiB and 2 GiB.** zarr-python's chunk
  cap is 64 MiB (1 MiB under sharding), dask's is 128 MiB; 8 MiB sits where
  a single cloud GET is cheap and a partial read is not wasteful. They are
  fields, so a caller who disagrees does not need a global.

### 3.3 The options mirror

```python
class ZarrOptions(tx.TypedDict, total=False):
    zarr_version: tz.ZarrVersion
    overwrite: bool
    driver: tx.Optional[str]
    attributes: tx.Mapping[str, tz.JSON]

class GroupOptions(ZarrOptions, total=False):
    pass

class ArrayOptions(ZarrOptions, total=False):
    dimension_names: ...
    chunks: ChunkSpec
    shards: tx.Optional[ChunkSpec]
    max_chunk_bytes: int
    ...                      # every ArrayConfig field except shape and dtype
```

`shape` and `dtype` are positional on `create_array` and so cannot also be
keys of the `Unpack`ed TypedDict without a duplicate-parameter complaint
from type checkers. They are the "what"; the options are the "how".

### 3.4 `GeneralConfig` and `OMEZarrConfig`

`GeneralConfig` (output name, `max_load`, log level) is a conversion
process's settings, not a creation config, and it is broken. Remove it from
`config.py`; the tool that needs it owns it. `OMEZarrConfig` moves to
`abczarr/ome/config.py` as `OMEImageConfig(ArrayConfig)` and is filled in
when pyramids land (section 6.4).

## 4. The call surface

### 4.1 What the user writes

Before:

```python
from abczarr import from_config
from abczarr.config import ZarrConfig

group = from_config("out.zarr", ZarrConfig(zarr_version=3, overwrite=True))
img = group.create_array(
    "img", (100, 2048, 2048), "uint16",
    chunk=(1, 512, 512), compressor="zstd", compressor_options={"level": 3},
)
# No array creation at a location. No "give me 8 MiB chunks". No sharding
# by budget. The rich ZarrArrayConfig is unreachable from any call.
```

After:

```python
import abczarr
from abczarr import ArrayConfig, GroupConfig

MiB = 1024 ** 2

# 1. Keywords only: the simplest spelling.
group = abczarr.create_group("out.zarr", overwrite=True)
img = group.create_array(
    "img", (100, 2048, 2048), "uint16",
    chunks=(1, "auto", "auto"), max_chunk_bytes=8 * MiB, compressor="zstd",
)

# 2. A reusable config, overridden per call. The keyword wins.
cfg = ArrayConfig(
    chunks=(1, "auto", "auto"), shards="auto",
    compressor="zstd", compressor_options={"level": 3},
)
raw = group.create_array("raw", (100, 2048, 2048), "uint16", config=cfg)
mask = group.create_array("mask", (100, 2048, 2048), "bool", config=cfg,
                          compressor=None)

# 3. One generic entry point that returns what the config describes.
g = abczarr.create("v2.zarr", GroupConfig(zarr_version=2))            # ZarrGroup
a = abczarr.create("a.zarr", ArrayConfig(shape=(4, 4), dtype="f4"))   # ZarrArray

# 4. See what a config will write before writing it.
cfg.resolve(shape=(100, 2048, 2048), dtype="uint16").chunks   # (1, 1024, 512)
cfg.resolve(shape=(100, 2048, 2048), dtype="uint16").to_metadata()
# ArrayMetadata(zarr_format=3, chunk_grid=RegularChunkGrid(...), codecs=(...))
```

### 4.2 The functions

```python
# abczarr.api

@tx.overload
def create(location, config: ArrayConfig, **options: tx.Unpack[ArrayOptions]) -> ZarrArray: ...
@tx.overload
def create(location, config: GroupConfig, **options: tx.Unpack[GroupOptions]) -> ZarrGroup: ...
def create(location, config: ZarrConfig, **options) -> ZarrNode: ...

def create_array(location, shape, dtype, *, config: tx.Optional[ArrayConfig] = None,
                 **options: tx.Unpack[ArrayOptions]) -> ZarrArray: ...

def create_group(location, *, config: tx.Optional[GroupConfig] = None,
                 **options: tx.Unpack[GroupOptions]) -> ZarrGroup: ...

# abczarr.abc.group.ZarrGroup

def create_array(self, name, shape, dtype, *, config: tx.Optional[ArrayConfig] = None,
                 **options: tx.Unpack[ArrayOptions]) -> ZarrArray: ...
def create_group(self, name, *, config: tx.Optional[GroupConfig] = None,
                 **options: tx.Unpack[GroupOptions]) -> ZarrGroup: ...
```

`from_config` is renamed to `create`; the two tests that call it change.
The `ZarrGroup.create_group(name, overwrite=False)` signature becomes the
config-plus-options form, and `overwrite` is still spelled
`create_group("sub", overwrite=True)`.

### 4.3 The merge rule, stated once

> A keyword passed with a config overrides that config's field. A keyword or
> config field that contradicts a fact the target already fixes is an error.

Concretely, `create_array` does:

```python
effective = replace(config or ArrayConfig(), shape=shape, dtype=dtype, **options)
```

and then, when called on a group, checks the fixed facts:

- `effective.zarr_version` must equal `self.zarr_version` (a v2 array cannot
  live in a v3 group). If the config left it at the default and the group is
  v2, the group's version is used: a template with the default `3` is not a
  contradiction, an explicit `zarr_version=3` keyword against a v2 group is.
  To tell the two apart, the base config records which fields were given
  explicitly (`_given: frozenset[str]`, set in `__attrs_post_init__` from
  the constructor call; pydantic calls this `model_fields_set`). Only an
  explicitly given value can contradict.
- `effective.driver` must be `None` or the group's driver name. A group
  created through zarr-python cannot create its child through tensorstore.
- `overwrite` applies to the child being created, as today.

Rejected alternatives:

- **Kwargs only.** Simplest to type, but there is no way to reuse a policy
  across twenty arrays without a helper dict, and a dict is an untyped,
  unvalidated `ArrayOptions` with none of `resolve()`.
- **Config only.** `group.create_array("mask", ArrayConfig(shape=..., dtype=...,
  compressor=None))` for every array is heavier than every neighbour, and it
  makes the one-off case (a shape, a dtype, one compressor) the clumsy one.
- **Both, with the config winning** (what zarr-python ended up with for
  `order`). It reads backwards: the keyword is the thing written closest to
  the call, and every other neighbour lets the closest thing win.
- **Both, with any overlap an error** (tensorstore). Correct for a Spec that
  describes an existing store; wrong for a template, whose whole purpose is
  to be overridden.

### 4.4 `open` stays as it is

`open`, `open_array`, `open_group` keep `(path, mode="a", *, driver=None)`.
They do not accept `config=` or the config's fields:

- The only field they share with the config is `driver`, and it is already
  a keyword.
- zarr-python's `open(mode="a", shape=...)` infers "create an array" from
  the presence of `shape`, and its `**kwargs` passthrough is the least typed
  part of its API. abczarr has a typed config; it does not need the
  inference.
- Opening and creating differ in what they promise: `open` returns what is
  there, `create` writes what you asked for. One function that does either
  depending on which keywords were passed is the thing `open_array` versus
  `open_group` already exists to avoid.

What `open(path, mode="w")` does today (the driver creates an empty group)
is left alone by this design, and is worth restricting later so that
creation has exactly one spelling.

## 5. `create` and the dispatch

### 5.1 No dispatch on the config's class

`create` does not test `isinstance(config, ArrayConfig)`. It lowers the
config and lets the metadata say what it is:

```python
def create(location, config, **options):
    config = replace(config, **options)
    driver = _driver_for(config)              # config.driver, else select_driver(...)
    root = None
    for relpath, metadata in config.plan():   # ("", metadata) for a plain config
        node = driver.create(_join(location, relpath), metadata,
                             overwrite=config.overwrite)
        root = root or node
    return root
```

`plan()` is the one method that varies:

```python
class ZarrConfig:
    def to_metadata(self) -> NodeMetadata: ...           # abstract
    def plan(self) -> tx.Iterator[tx.Tuple[str, NodeMetadata]]:
        yield "", self.to_metadata()                     # one node, at the root
```

A `GroupConfig` lowers to `GroupMetadataV2` or `GroupMetadataV3` (the table
`abc/group.py` already keeps). An `ArrayConfig` lowers to the
version-specific `ArrayMetadata`. Whether `driver.create` writes a group or
an array is decided by `metadata.node_type`, a field the metadata model has
carried from the start. An `OMEImageConfig` overrides `plan()` to yield the
root group and then one array per level (section 6.4), and nothing in
`create` changes.

The return type is narrowed statically by the overloads in 4.2, and
dynamically by an `isinstance` check that raises `UnsupportedZarrOperation`
if a driver hands back the wrong kind of node, as `from_config` does today.

### 5.2 The driver contract

`Driver.create_group(location, *, zarr_version, overwrite)` is replaced by
one primitive that takes metadata:

```python
class Driver:
    def can_create(self, metadata: NodeMetadata) -> Verdict:
        """can_open(metadata) plus the "writes" capability."""
    def create(self, location, metadata: NodeMetadata, *, overwrite: bool = False) -> ZarrNode:
        raise UnsupportedZarrOperation("create", self.name or None)
```

and `ZarrGroup.create_array` / `create_group` become **concrete** in the
ABC. They merge, check the fixed facts, resolve, lower, and call two
abstract hooks:

```python
class ZarrGroup(ZarrNode):
    def create_array(self, name, shape, dtype, *, config=None, **options) -> ZarrArray:
        effective = self._effective(config or ArrayConfig(), shape=shape, dtype=dtype, **options)
        return self._create_array(name, effective.to_metadata(), overwrite=effective.overwrite)

    @abstractmethod
    def _create_array(self, name: str, metadata: ArrayMetadata, *, overwrite: bool) -> ZarrArray: ...
    @abstractmethod
    def _create_group(self, name: str, metadata: GroupMetadata, *, overwrite: bool) -> tx.Self: ...
```

This is the template-method shape `PathGroup` already has (`_open_array`,
`_create_array`), promoted to the ABC, and it is the roadmap's "thin driver
contract: a driver supplies only primitives". The public signature the
tests and the abc-surface design pin down is unchanged; what changes is
which method a driver writes.

How each backend consumes metadata, checked against the installed libraries:

| Backend | Group | Array |
|---|---|---|
| zarr-python 3.1.6 | `zarr.open_group(location, mode="w" / "w-", zarr_format=metadata.zarr_format)`, then `attrs.update(metadata.attributes)`. | Map the metadata to `Group.create_array` keywords: v3 codec pipeline splits into `filters` (array-to-array), `serializer` (array-to-bytes), `compressors` (bytes-to-bytes); `sharding_indexed` becomes `shards=` plus its inner pipeline; `chunk_grid` becomes `chunks=`; `chunk_key_encoding`, `fill_value`, `dimension_names`, `attributes` pass straight through. v2 metadata maps to `chunks`, `compressor`, `filters`, `fill_value`, `order`, `dimension_separator`. The fallback, if a pipeline cannot be expressed in keywords, is what `PathGroup.create_group` does for groups: write the document with `metadata.to_file` and open it. |
| tensorstore 0.1.85 | `PathGroup.create_group` writes the group document; already works. | `ts.open({"driver": "zarr3", "kvstore": ..., "metadata": metadata.to_dict()}, create=True, delete_existing=overwrite)`. Verified: a complete v3 document with a zstd codec and attributes creates the array, and `ts.open` reports the chunk grid and codecs from it. `TensorStoreGroup._create_array` becomes a ten-line method. |

The driver maps *metadata*, not the config's keywords. That is the point of
lowering: `_create_kwargs` in the zarr-python driver, which maps the
TypedDict keys today, is rewritten once to map `ArrayMetadata`, and never
has to learn a new coarse keyword again.

### 5.3 Driver selection for creation

`_driver_for(config)` mirrors `_resolve_drivers` plus `_choose` in
`api.py` today: an explicit `driver` (name or `Driver`) is used as is; with
`None`, `select_driver(metadata, available_drivers())` is called with
`can_create` in place of `can_open`. The metadata's `required_features()`
already names the codecs and grids the array needs, so an array asking for
a codec only one installed backend can write is routed to that backend, and
an array nobody can write fails up front with the same "X lacks
v3:codec:..." message opening gives. A group requires no features, so it
goes to the first available driver.

## 6. Coarse to fine: how a config becomes version-correct metadata

### 6.1 Three stages

```
ArrayConfig                    "auto" chunks, shards, compressor, fill_value, separator;
   |  resolve()                shape/dtype may still be None on a template
   v
ArrayConfig (resolved)         every field concrete; shape and dtype present;
   |  to_metadata()            chunks and shards are tuples; compressor is a mapping or None
   v
ArrayMetadata (v3)             chunk_grid, codecs (with sharding_indexed when sharded),
   |  to_version(n, "strict")  chunk_key_encoding, data_type, fill_value, dimension_names
   v
ArrayMetadata (v1 / v2 / v3)   what the driver creates from
```

`resolve()` is today's `finalize()`, renamed because "finalize" suggests a
lifecycle and this is a pure function. It accepts `data=` (an array-like
whose shape and dtype fill the blanks) or explicit `shape=`, `dtype=`,
`dimension_names=`, and returns a new config with no `"auto"` left.
`to_metadata()` calls `resolve()` itself, so a user never has to.

Resolution rules, in order:

1. `shape` and `dtype` must be known; else `ValueError` naming the field.
2. `chunks` and `shards` are broadcast to the rank (repeat the last entry,
   or look each dimension name up in a mapping), `-1` becomes the axis
   extent, and `"auto"` axes are grown by doubling until the next doubling
   would exceed `max_chunk_bytes * compression_ratio / itemsize` elements
   (the algorithm in `_core/sharding.py`, with its termination bug fixed).
   When `shards` is set, the shard grid is grown the same way against
   `max_shard_bytes`, then chunks are grown inside the shard against
   `max_chunk_bytes` (passing the budget through, which today's
   `auto_shard` forgets), and each shard extent is rounded up to a multiple
   of the chunk extent.
3. `compressor="auto"` becomes zstd; `"none"` becomes `None`; a string
   becomes `{"name": <str>, "configuration": compressor_options}`; a
   mapping is kept.
4. `fill_value="auto"` becomes the dtype's zero when `zarr_version == 3` and
   stays null for v2 and v1.
5. `dimension_separator="auto"` becomes `/` for v3 and `.` for v2 and v1.
6. Contradictions with the target version raise here, with the field named:
   `shards` on v2 or v1, `dimension_names` on v1, a `fill_value=None` on v3.

### 6.2 Lowering, one path

`to_metadata()` builds the v3 document (the richest of the three) and calls
`ArrayMetadata.to_version(zarr_version, policy="strict")`. This is what the
existing `to_metadata` does, minus the strict policy, and it was verified
today: the same config lowers to a v3 `ArrayMetadata` with a regular chunk
grid and a `bytes` plus `blosc` pipeline, and to a v2 `ArrayMetadata` with
`chunks=(1, 1024, 512)`, a numcodecs `BloscCodec`, `dimension_separator="/"`
and `order="C"`.

The reason for one path rather than one builder per version is that the
metadata layer already owns, and tests, the correspondence between a v3
pipeline and v2's `compressor` plus `filters` plus byte-order-bearing dtype.
Writing a second v2 builder in `config.py` would be a second copy of that
table. `"strict"` turns a request v2 cannot hold (a sharding codec, a
rectilinear grid, dimension names on v1) into an `UnsupportedConversion`
naming the field, instead of dropping it silently and writing a smaller
array than the user asked for.

Per-version outcome for one resolved config:

| Resolved field | v3 document | v2 document (`to_version(2)`) |
|---|---|---|
| `chunks=(1, 1024, 512)`, no shards | `chunk_grid: regular, chunk_shape [1, 1024, 512]` | `chunks: [1, 1024, 512]` |
| `chunks=(1, 512, 512)`, `shards=(8, 2048, 2048)` | `chunk_grid: regular [8, 2048, 2048]`; `codecs: [sharding_indexed {chunk_shape [1, 512, 512], codecs [bytes, zstd], index_codecs [bytes, crc32c]}]` | `UnsupportedConversion("codecs.sharding_indexed", 2)` |
| `compressor={"name": "zstd", "configuration": {"level": 3}}` | `codecs: [bytes(little), zstd(level 3)]` | `compressor: {"id": "zstd", "level": 3}` |
| `filters=({"name": "transpose", ...},)` | array-to-array codec ahead of `bytes` | `filters: [...]` when numcodecs has it, else `UnsupportedConversion` |
| `order="F"` | a `transpose` codec | `order: "F"` |
| `fill_value="auto"` | `0` (dtype's zero) | `null` |
| `dimension_separator="auto"` | `chunk_key_encoding: default, separator "/"` | `dimension_separator: "."` |
| `dimension_names=("z", "y", "x")` | `dimension_names` | dropped: policy `"strict"` raises; `"warn"` keeps going |

### 6.3 A worked example, with today's resolver

```python
cfg = ArrayConfig(
    shape=(100, 2048, 2048), dtype="uint16",
    chunks=(1, "auto", "auto"), max_chunk_bytes=1 * MiB,
    compressor="blosc", compressor_options={"cname": "zstd", "clevel": 3},
)
cfg.resolve().chunks
# (1, 1024, 512)
#   1 MiB * 1.8 / 2 bytes = 943,718 elements; (1, 1024, 512) = 524,288 fits,
#   the next doubling on either axis (1,048,576) does not.
sorted(cfg.to_metadata().required_features())
# ['v3:chunk_grid:regular', 'v3:chunk_key_encoding:default',
#  'v3:codec:blosc', 'v3:codec:bytes', 'v3:data_type:uint16']
```

Those are real outputs from the current `finalize()` and `to_metadata()`,
so the numbers above hold once the renames are done. The feature set is
what `select_driver` uses to route the array to a backend that can write it.

### 6.4 Where OME pyramids plug in

An OME-Zarr image is one group whose attributes carry the `multiscales`
document, plus one array per resolution level. `OMEImageConfig` is an
`ArrayConfig` (the level-0 array's config) with the OME-specific choices:

```python
class OMEImageConfig(ArrayConfig):
    ome_version     : OMEVersion | "auto" = "auto"    # derived from zarr_version, or the reverse
    axes            : tuple[Axis, ...]                # name, type, unit per dimension
    levels          : int = -1                        # -1: until the coarsest level fits one chunk
    no_pyramid_axis : SpatialAxisName | None = None
    chunk_channels  : bool = False                    # sugar for chunks["c"] = 1
    chunk_time      : bool = True                     # sugar for chunks["t"] = 1
    shard_channels  : bool = False
    shard_time      : bool = False
    scale, translation ...                            # coordinate transforms per level
```

It overrides two methods and nothing else:

- `plan()` yields `("", GroupMetadata(attributes=<multiscales built from
  abczarr.ome.metadata>))`, then `(f"{level}", level_config.to_metadata())`
  for each level, where `level_config = replace(self, shape=next_shape)` and
  `next_shape` comes from `_core/pyramid.next_level_shape`. Each level is
  therefore resolved by the same `resolve()`, with the same budget, so a
  coarse level gets smaller chunks or shards where that is the right thing
  and nothing is special-cased.
- A version table: OME 0.1 to 0.4 sit on Zarr v2, 0.5 and later on v3. If
  `ome_version` and `zarr_version` are both given and disagree, `ValueError`;
  if one is given, the other is derived.

`chunk_time` and friends are resolved in `resolve()` into the chunk mapping
before the generic rules run, which is why the mapping form of `chunks` is
keyed by dimension name. Writing the downsampled data is not creation and
is not this API; a later `write_pyramid(image, data)` fills the levels
`create` laid out.

## 7. Module layout

### 7.1 The recommendation

```
src/abczarr/
  __init__.py          # open, open_array, open_group, create, create_array,
                       #   create_group, ZarrConfig, GroupConfig, ArrayConfig,
                       #   ArrayOptions, GroupOptions, nodes, errors, register_driver
  config.py            # ZarrConfig, GroupConfig, ArrayConfig + the *Options mirrors
  api/
    __init__.py        # re-exports the façade; the documented path stays abczarr.api
    _open.py           # open, open_array, open_group, _choose, _peek_array_metadata
    _create.py         # create, create_array, create_group, _driver_for
    _registry.py       # register_driver, available_drivers, select_driver
  abc/                 # unchanged; errors.py stays here
  drivers/             # Driver.create / can_create; _create_array / _create_group hooks
  metadata/            # unchanged
  ome/
    config.py          # OMEImageConfig, when pyramids land
```

Import direction, bottom to top, with no cycles:

```
_core  <-  metadata  <-  config  <-  abc  <-  drivers  <-  api
```

### 7.2 Why each piece is where it is

- **`api/` is a package because there are now two verbs.** `open` and
  `create` each carry a few helpers, and the driver-selection code they
  share (`_driver_for`, `_choose`) has a home. The registry is façade code:
  its callers are `api` and third parties registering a driver, and no
  driver imports it.
- **The modules are private (`_open.py`), not `open.py`.** A public module
  named after the function it exports collides: `abczarr.api.open` would be
  both the function and the module, `import abczarr.api.open` would replace
  the function attribute with the module object, and griffe builds the
  reference from dotted paths, so mkdocstrings would render the wrong one
  (bagof-magic hit exactly this with `fields.py` and renamed it to
  `_fields.py`). Any public name a user should reach is re-exported from
  `abczarr.api` and `abczarr`; the module names are free to change.
- **`config.py` stays at the top level, not under `api/`.** `abc/group.py`
  names `ArrayConfig` in `create_array`'s signature and calls
  `to_metadata()` in the concrete implementation, so `config` must sit
  below `abc`. Putting it in `api/` would make the abstract layer import
  the façade. Putting it in `abc/` was considered and rejected: `abc/` holds
  the interfaces, and these are concrete data classes a user instantiates.
  One caution on the name: `zarr.config` is a global runtime-settings object
  (donfig). If abczarr ever grows global defaults, they should not be
  called `config`; `abczarr.defaults` would be the place.
- **`errors.py` stays in `abc/`.** The maintainer's instinct, that errors
  are raised everywhere and should be reachable everywhere, is right, and it
  argues for leaving them where they are: `abc` is the bottom of the import
  graph, so every layer (drivers, api, registry, metadata's callers) already
  reaches `abc/errors.py` without a cycle. Moving it under `api/` would
  invert that. "Broadly reachable" for a *user* means `abczarr.
  UnsupportedZarrOperation` and `abczarr.TransactionConflict`, which the
  top-level `__init__` already exports; the docs page (`::: abczarr.abc.
  errors`) and its place in the nav are unchanged. No alias module
  (`abczarr/errors.py`) is added: two dotted paths for one class is the
  duplication mkdocstrings and griffe punish.
- **No new error class.** A config that contradicts its target raises
  `ValueError` naming the field (as the `__attrs_post_init__` checks do
  today); a request the version cannot hold raises the metadata layer's
  `UnsupportedConversion`; a driver that cannot create raises
  `UnsupportedZarrOperation`. Three existing errors cover the three kinds
  of failure.

### 7.3 Docs

`docs/api/open.md` keeps `::: abczarr.api` filtered to the `open*` members;
a new `docs/api/create.md` renders the `create*` members; `docs/api/
config.md` keeps `::: abczarr.config`; the "Opening" nav group becomes
"Opening and creating" with the three pages. `docs/api/drivers.md` gains
the registry functions, which now live under `abczarr.api`.

## 8. Implementation notes and order

Not part of the design, but the order in which it should land so each
step is testable on its own.

1. Fix `_core/sharding.py`: the non-terminating loops (a doubling that
   changes nothing must not count as improvement) and the dropped budget
   in `auto_shard`; add `-1` for the whole axis; tests with a shape where
   one auto axis saturates.
2. `config.py`: the hierarchy in 3.2, `resolve()`, `to_metadata()` with
   `policy="strict"`, `plan()`, the `_given` set, the `*Options` mirrors,
   and the drift test between mirror and class. Widen
   `tz.CompressorTypeV3` (or stop using a Literal for it; the codec
   registry is the source of truth).
3. `abc/group.py`: concrete `create_array` / `create_group`, abstract
   `_create_array` / `_create_group`; remove the TypedDict from
   `abc/array.py`.
4. `drivers/base.py`: `can_create`, `create(location, metadata, *,
   overwrite)`; zarr-python: metadata to keywords; tensorstore: `ts.open`
   with the metadata document. `test_zarr_python_create.py` runs unchanged
   except for the renamed keys; the tensorstore creation tests that are
   skipped today start passing.
5. `api/`: the package split, `create*`, the overloads; update `__init__`
   and the docs nav.
6. `ome/config.py` when pyramid writing is scheduled.

Open questions the maintainer should settle, with the recommended answer
first:

- Should `create_array` at the top level accept `data=` (shape and dtype
  from an array-like, then write it), as zarr-python's `create_array(data=)`
  and `from_array` do? Recommended: yes, later, as `abczarr.from_array(
  location, data, *, config=None, **options)`; it is `resolve(data=)` plus
  one `__setitem__`.
- Should `open(path, mode="w")` keep creating an empty group? Recommended:
  no; restrict `mode` on `open*` to the reading modes once `create_group`
  exists, so creation has one spelling.
