# abczarr

<img src="docs/images/logo_title_color.svg" style="display: block; margin: 0 auto; width: 75%; height: auto;" alt="abczarr logo" />

One interface for reading and writing Zarr arrays and groups -- sync or
async -- no matter which backend or storage location holds them.

> [!WARNING]
> **Status:** pre-release -- under active development, not ready for use yet.

## What it does

Zarr has several good Python implementations -- zarr-python, tensorstore,
zarrista -- and they don't all support the same things or speak the same API.
abczarr sits on top of them and gives you one API to write against,
`ZarrArray` and `ZarrGroup`, and picks whichever backend actually supports
what you're asking for. You can also name a backend yourself if you care
which one runs. Local paths and cloud URLs like `s3://` or `gs://` work the
same way either way.

Every node comes in a synchronous and an asynchronous form, and you don't
have to think about which backend makes that easy. Where the backend has
real async I/O, abczarr uses it directly; where it doesn't, the call quietly
runs in a background thread instead. Either way you get an `await`-able
array or group with the same behavior.

Because backends genuinely differ in what they can do, abczarr lets you
check a backend's capabilities before you rely on them, rather than finding
out partway through a write. And when an operation really isn't supported,
you get an error that says what's missing -- not a stack trace from deep
inside someone else's driver.

Zarr's metadata has changed shape across versions -- v1, v2, and v3 all
describe an array a little differently -- and abczarr models all of them as
one typed, validated object that knows how to convert between versions. When
a conversion can carry an option across cleanly, it does; when it can't, you
choose whether that's silently dropped, a warning, or a hard error. OME-Zarr
metadata gets the same treatment: a typed model, schema validation you can
run offline, and conversion between OME-NGFF versions.

```python
from abczarr.api import open

# open a Zarr node anywhere; the right backend is chosen for you
group = open("s3://my-bucket/dataset.zarr")
array = group["images"]
data = array[:100, :100]

# the async version awaits the open call and uses method calls instead of indexing
agroup = await open("s3://my-bucket/dataset.zarr", asynchronous=True)
aimages = await agroup.getitem("images")
tile = await aimages.getitem((slice(100), slice(100)))
```

## Learn more

Full documentation lives at
[neuroscales.github.io/abczarr](https://neuroscales.github.io/abczarr/).

abczarr is part of the **bagof** ecosystem of small, focused packages --
see [bagof-paths](https://github.com/bagofseeds/bagof-paths) for the
storage layer and [bagof-magic](https://github.com/bagofseeds/bagof-magic)
for the type-hint-driven data classes abczarr's metadata model is built on.
