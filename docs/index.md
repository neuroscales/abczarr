# abczarr

abczarr is one interface for reading and writing Zarr arrays and
groups -- synchronous or asynchronous -- no matter which backend or
storage location holds them. Write against a single API, and swap
zarr-python, tensorstore or zarrista, a local directory or a cloud
bucket, underneath it without touching your code.

## What you get

- **One API, sync or async.** Write against `ZarrArray` /
  `ZarrGroup` once; every node has an `await`-able twin -- native
  where the backend offers async I/O, transparently threaded where
  it doesn't.
- **Three interchangeable drivers, auto-selected.** zarr-python,
  tensorstore and zarrista sit behind the same interface. Name one
  explicitly, or let abczarr pick the one that supports what an
  array actually needs. Local paths and cloud URLs (`s3://`,
  `gs://`, ...) work out of the box.
- **Honest about what works.** Ask a driver what it supports -- and
  whether that support is native to the backend or synthesized from
  simpler operations -- before you rely on it. When something
  genuinely isn't supported, the error names the missing feature
  instead of failing deep in a backend's internals.
- **Typed, versioned metadata with conversion.** One frozen,
  validated model spans Zarr v1, v2 and v3 and converts between
  them -- keeping equivalent options where the target format allows,
  and telling you what it drops when it can't (`lossy`, `warn` or
  `strict`).
- **OME-Zarr, first class.** The versioned OME-NGFF metadata gets
  the same typed model, offline JSON-Schema validation, and
  cross-version conversion as the core Zarr metadata.

```python
from abczarr.api import open

# any backend, any location -- the driver is chosen for the array
group = open("s3://my-bucket/dataset.zarr")
array = group["images"]
data = array[:100, :100]

# the same call is await-able -- native async where the backend has it
agroup = await open("s3://my-bucket/dataset.zarr", asynchronous=True)
aimages = await agroup.getitem("images")
tile = await aimages.getitem((slice(100), slice(100)))
```

New here? The [tutorial](tutorial.md) walks through creating an
array, reading and writing it, and navigating a group. For the full
surface, see the [reference](api/index.md).
