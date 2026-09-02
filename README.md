# abczarr

**ABC...Z(arr)** — one interface for reading and writing Zarr arrays and
groups, no matter which backend or storage location holds them.

> **Status:** pre-release — under active development, not ready for use yet.

## What it does

abczarr gives you a single, uniform API for Zarr — `ZarrArray`, `ZarrGroup`,
and a key→bytes `Store` — the way a pathlib-style wrapper gives you one API
over many filesystems. Write your code against that surface once, and swap
what's underneath without touching it:

- **One surface, many backends.** `ZarrArray` and `ZarrGroup` behave the
  same whether the data lives behind zarr-python or tensorstore; the right
  driver is chosen automatically for what an array needs.
- **One surface, many stores.** The default store is built on
  [bagof-paths](https://github.com/bagofseeds/bagof-paths), so a local
  directory and any fsspec or cloud URL (`s3://`, `gs://`, ...) work with no
  extra code.
- **Honest capabilities.** Ask a store or a driver what it supports — and
  whether that support is native to the backend or built up from simpler
  operations — before you rely on it.
- **Clear failures.** When something is genuinely unsupported, the error
  names exactly what is missing, instead of failing deep in someone else's
  internals.
- **Typed, versioned metadata.** A single metadata model spans Zarr formats
  v1, v2 and v3, with lossless-where-possible conversion between them, and
  first-class support for OME-Zarr.

```python
from abczarr.abc.store import PathStore

store = PathStore("s3://my-bucket/dataset.zarr")
group = open_group(store)  # same code, any backend or scheme
array = group["images"]
data = array[:100, :100]
```

## Learn more

Full documentation lives at
[neuroscales.github.io/abczarr](https://neuroscales.github.io/abczarr/).

abczarr is part of the **bagof** ecosystem of small, focused packages —
see [bagof-paths](https://github.com/bagofseeds/bagof-paths) for the
storage layer and [bagof-magic](https://github.com/bagofseeds/bagof-magic)
for the type-hint-driven data classes abczarr's metadata model is built on.
