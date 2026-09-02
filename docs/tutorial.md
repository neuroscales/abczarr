# Tutorial

A short walkthrough of the four things you do most: create a group and
an array, read and write data, create an array from a reusable config,
and navigate a group's members. It runs against a plain local
directory, so you can follow along without any cloud storage or extra
setup beyond `pip install abczarr[zarr-py]`.

## Create a group and an array

[create_group][abczarr.api.create_group] makes a Zarr group at a
path. Call [create_array][abczarr.abc.group.ZarrGroup.create_array] on
it to add an array inside:

```pycon
>>> import numpy as np
>>> import abczarr

>>> group = abczarr.create_group("weather.zarr", overwrite=True)
>>> isinstance(group, abczarr.ZarrGroup)
True

>>> temperature = group.create_array(
...     "temperature", shape=(4, 4), dtype="float32", chunks=(2, 2)
... )
>>> temperature.shape
(4, 4)
>>> str(temperature.dtype)
'float32'
>>> temperature.chunks
(2, 2)

```

`overwrite=True` clears anything already at that path, which keeps
this tutorial repeatable; drop it once you are working with real data.

## Read and write

An array reads and writes like NumPy, with NumPy-style selections:

```pycon
>>> temperature[:] = np.arange(16, dtype="float32").reshape(4, 4)
>>> temperature[1, :]
array([4., 5., 6., 7.], dtype=float32)

```

## Create via a config

[ArrayConfig][abczarr.config.ArrayConfig] describes an array once,
chunking, sharding, compression, fill value, so you can reuse the same
shape of description across arrays. Pass it to
[create][abczarr.api.create]:

```pycon
>>> from abczarr.config import ArrayConfig

>>> counts = abczarr.create(
...     "weather.zarr/counts",
...     ArrayConfig(shape=(10,), dtype="int16", chunks=4),
... )
>>> counts[:] = np.arange(10)
>>> np.asarray(counts)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int16)

```

Fields left as `"auto"`, chunk size, compressor, fill value, are
resolved from the shape, dtype, and Zarr format version, so a config
this small already produces sensible chunking and compression.

## Navigate a group

A group behaves like a mapping of names to nodes. Nest another group
inside it, then list, test, and index members the way you would a
dict:

```pycon
>>> stations = group.create_group("stations")
>>> _ = stations.create_array(
...     "station-a", shape=(3,), dtype="uint8", chunks=(3,)
... )
>>> sorted(group.keys())
['counts', 'stations', 'temperature']
>>> "stations" in group
True
>>> sorted(group["stations"].keys())
['station-a']

```

## Reopen it later

[open][abczarr.api.open] reads whatever is at a path, group or array,
and hands back the same uniform surface, no matter which backend wrote
it:

```pycon
>>> reopened = abczarr.open("weather.zarr", mode="r")
>>> isinstance(reopened, abczarr.ZarrGroup)
True
>>> sorted(reopened.keys())
['counts', 'stations', 'temperature']
>>> np.asarray(reopened["temperature"])
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.],
       [12., 13., 14., 15.]], dtype=float32)

```

The same code opens `"s3://my-bucket/dataset.zarr"` or any other
fsspec URL; only the path changes. From here, the [Reference](api/index.md)
covers the full surface, and [Metadata](api/metadata.md) covers the typed,
versioned model behind it.
