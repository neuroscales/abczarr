# abczarr

abczarr is one interface for reading and writing Zarr arrays and
groups, no matter which backend or storage location holds them.
Write against a single API, and swap zarr-python, tensorstore, a
local directory, or a cloud bucket underneath it without touching
your code.

## What you get

- **One surface, many backends.** `ZarrArray` and `ZarrGroup` work
  the same way whether the data lives behind zarr-python or
  tensorstore.
- **One surface, many stores.** The default store is built on
  [bagof-paths](https://neuroscales.github.io/bagof-paths/), so a
  local path and any fsspec or cloud URL (`s3://`, `gs://`, ...)
  work with no extra code.
- **Honest capabilities.** Ask a store or a driver what it supports
  -- and whether that support is native to the backend or built up
  from simpler operations -- before you rely on it.
- **Clear failures.** When a backend can't open an array or perform
  an operation, the error names exactly what is missing.

```python
from abczarr.abc.store import PathBasedStore

store = PathBasedStore("s3://my-bucket/dataset.zarr")
group = open_group(store)  # same code, any backend or scheme
array = group["images"]
data = array[:100, :100]
```

New here? The [tutorial](tutorial.md) walks through creating an
array, reading and writing it, and navigating a group. For the full
surface, see the [reference](api/index.md).
