# abczarr

Zarr has several good Python implementations -- zarr-python, tensorstore,
zarrista -- and they don't all support the same things or speak the same
API. abczarr sits on top of them and gives you one interface to write
against, `ZarrArray` and `ZarrGroup`, whether the data lives on a local
disk or behind a cloud URL like `s3://` or `gs://`. abczarr picks a backend
that supports what you're asking for, or you can name one yourself.

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

New here? The [tutorial](tutorial.md) walks through creating an
array, reading and writing it, and navigating a group. For the full
surface, see the [reference](api/index.md).
