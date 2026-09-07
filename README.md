# abczarr

<img src="https://neuroscales.github.io/abczarr/images/logo_title_color.svg" style="display: block; margin: 0 auto; width: 75%; height: auto;" alt="abczarr logo" />

One interface for reading and writing Zarr arrays and groups, synchronously
or asynchronously, regardless of which backend or storage location holds
them.

## What it does

Zarr has several capable Python implementations
([`zarr-python`](https://github.com/zarr-developers/zarr-python),
[`tensorstore`](https://github.com/google/tensorstore/),
[`zarrista`](https://github.com/developmentseed/zarrista)), and they
differ in what they support and in the API each one exposes. abczarr sits
above these implementations and exposes a single API, `ZarrArray` and
`ZarrGroup`, and selects whichever backend actually supports the requested
operation. A backend can also be named explicitly when a specific one is
required. Local paths and cloud URLs such as `s3://` or `gs://` are handled
the same way, regardless of which backend is in use.

Every node exists in both a synchronous and an asynchronous form, and the
two behave identically regardless of what the underlying backend supports.
When a backend provides real asynchronous I/O, abczarr uses it directly.
When a backend does not, the call runs in a background thread instead.
Either form returns an object with the same behavior, and the asynchronous
form can be awaited.

Backends genuinely differ in what they support. abczarr exposes a backend's
capabilities so that they can be checked before an operation runs, instead
of discovering a gap partway through a write. When an operation is not
supported, the resulting error names what is missing, instead of
surfacing as a stack trace from inside a driver's internals.

Zarr's array metadata has changed shape across versions. Versions v1, v2,
and v3 each describe an array a little differently. abczarr models all
three as one typed, validated object that converts between versions. When
a conversion can carry an option across cleanly, it does. When it cannot,
the missing option can be configured to be dropped silently, to raise a
warning, or to raise a hard error. OME-Zarr metadata receives the same
treatment: a typed model, schema validation that runs offline, and
conversion between OME-NGFF versions.

```python
from abczarr import open

# opens a Zarr node at any location, with the backend chosen automatically
group = open("s3://my-bucket/dataset.zarr")
array = group["images"]
data = array[:100, :100]

# a backend can also be named explicitly
volume = open("data.zarr", driver="tensorstore")

# the asynchronous form awaits open() and each subsequent method call
agroup = await open("s3://my-bucket/dataset.zarr", asynchronous=True)
aimages = await agroup.getitem("images")
tile = await aimages.getitem((slice(100), slice(100)))
```

## Install

```sh
pip install abczarr
```

The core installs no backend of its own. The driver needed for reading
and writing, along with any required storage or dtype support, is added
through extras:

| extra | enables |
| --- | --- |
| `abczarr[zarr-py]` | the zarr-python driver |
| `abczarr[tensorstore]` | the TensorStore driver |
| `abczarr[zarrista]` | the zarrista driver |
| `abczarr[upath]` | fsspec and cloud URLs (`s3://`, `gs://`, ...) via universal-pathlib |
| `abczarr[anypath]` | cloud paths via cloudpathlib |
| `abczarr[ml-dtypes]` | exotic v3 float dtypes (`bfloat16`, `float8_*`, ...) |

Extras can be combined, for example `pip install "abczarr[zarr-py,upath]"`.

## Learn more

Full documentation lives at
[neuroscales.github.io/abczarr](https://neuroscales.github.io/abczarr/).
