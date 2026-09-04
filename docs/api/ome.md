# OME-Zarr metadata

OME-Zarr (the NGFF spec) is a metadata convention for bioimaging data
stored in Zarr: multiscale image pyramids, high-content screening
plates, segmentation labels, and rendering settings, all described by
JSON attached to a Zarr group. abczarr models that metadata as typed
classes under `abczarr.ome.metadata`. There is one package per NGFF
version, `v0_1` through `v0_5`, plus a `v0_6dev4` preview of the next
release. The examples below target 0.5, the latest stable version; the
[Reference](#reference) documents every version.

## Describing a multiscale image

A multiscale image is a pyramid of resolution levels, each a Zarr
array, described by
[Multiscale][abczarr.ome.metadata.v0_5.images.Multiscale]. Build it
from a plain dict shaped like the JSON the spec defines:

```pycon
>>> from abczarr.ome.metadata import v0_5
>>> multiscale = v0_5.Multiscale.from_dict({
...     "name": "nucleus-stain",
...     "type": "gaussian",
...     "axes": [
...         {"name": "c", "type": "channel"},
...         {"name": "y", "type": "space", "unit": "micrometer"},
...         {"name": "x", "type": "space", "unit": "micrometer"},
...     ],
...     "datasets": [
...         {
...             "path": "0",
...             "coordinateTransformations": [
...                 {"type": "scale", "scale": [1.0, 0.325, 0.325]}
...             ],
...         },
...         {
...             "path": "1",
...             "coordinateTransformations": [
...                 {"type": "scale", "scale": [1.0, 0.65, 0.65]}
...             ],
...         },
...     ],
... })
>>> [axis.name for axis in multiscale.axes]
['c', 'y', 'x']
>>> [dataset.path for dataset in multiscale.datasets]
['0', '1']

```

Each axis in `axes` becomes an
[Axis][abczarr.ome.metadata.v0_5.axes.Axis]. Here that's a
[SpaceAxis][abczarr.ome.metadata.v0_5.axes.SpaceAxis], since each one
carries `type="space"`. Each entry in `datasets` becomes a
[Dataset][abczarr.ome.metadata.v0_5.images.Dataset]. It names an array
and carries the [Scale][abczarr.ome.metadata.v0_5.transformations.Scale]
that places it in physical space, one value per axis, in that axis's
unit.

A `Multiscale` describes the pyramid, not the whole group. Wrap it in
[OMEImage][abczarr.ome.metadata.v0_5.ome.OMEImage] to get the metadata
an image group actually carries, optionally alongside
[Omero][abczarr.ome.metadata.v0_5.omero.Omero] rendering settings:

```python
from abczarr.ome.metadata import v0_5

image = v0_5.OMEImage(
    version="0.5",
    multiscales=[multiscale],
    omero=v0_5.Omero.from_dict({
        "channels": [
            {
                "color": "00FF00",
                "window": {"start": 0, "end": 1500, "min": 0, "max": 65535},
            },
        ],
    }),
)
```

The same shape covers the other kinds of OME-Zarr group:
[ImageLabel][abczarr.ome.metadata.v0_5.labels.ImageLabel] for a
segmentation, and
[Plate][abczarr.ome.metadata.v0_5.plates.Plate] /
[Well][abczarr.ome.metadata.v0_5.wells.Well] for a high-content
screen, wrapped in
[OMEImageLabel][abczarr.ome.metadata.v0_5.ome.OMEImageLabel],
[OMEPlate][abczarr.ome.metadata.v0_5.ome.OMEPlate] and
[OMEWell][abczarr.ome.metadata.v0_5.ome.OMEWell].

## Reading and writing OME metadata on a group

OME-Zarr metadata lives in a group's user attributes, the same
`attrs` mapping any Zarr group exposes. NGFF 0.5 nests it all under
one `"ome"` key, so an object round-trips through
[to_dict][abczarr._core.metadata.Metadata.to_dict] and
[from_dict][abczarr._core.metadata.Metadata.from_dict] like this:

```python
group.attrs["ome"] = image.to_dict()

loaded = v0_5.OMEImage.from_dict(group.attrs["ome"])
loaded.multiscales[0].axes[0].name  # "c"
```

Earlier NGFF versions (0.4 and before) write the same fields directly
at the top level of `attrs` instead of nesting them under `"ome"`:
`group.attrs.update(image.to_dict())`.

## Converting between NGFF versions

[OMEMetadata.to_version][abczarr.ome.metadata.base.OMEMetadata.to_version]
converts a piece of OME metadata, built against one NGFF version, to
another. It works on the top-level container and on any nested piece
of metadata alike:

```pycon
>>> from abczarr.ome.metadata import v0_4
>>> old = v0_4.Multiscale.from_dict({
...     "version": "0.4",
...     "axes": [
...         {"name": "y", "type": "space"},
...         {"name": "x", "type": "space"},
...     ],
...     "datasets": [
...         {
...             "path": "0",
...             "coordinateTransformations": [
...                 {"type": "scale", "scale": [1.0, 1.0]}
...             ],
...         }
...     ],
... })
>>> new = old.to_version("0.5")
>>> type(new).__module__
'abczarr.ome.metadata.v0_5.images'
>>> new.to_version("0.4") == old
True

```

Fields both versions share carry over unchanged. Converting forward,
a field only the newer version has gets a reasonable default. Axes,
for example, gained a `type` when NGFF 0.4 introduced them, defaulted
from the axis's name. Converting back, that field is dropped.
Converting to a version that would need information the source does
not carry raises `ValueError` rather than guessing.

!!! example
    ```pycon
    >>> from abczarr.ome.metadata import v0_2
    >>> untyped = v0_2.Multiscale.from_dict({
    ...     "version": "0.2",
    ...     "name": "x",
    ...     "datasets": [{"path": "0"}],
    ... })
    >>> untyped.to_version("0.3")
    Traceback (most recent call last):
        ...
    ValueError: cannot convert Multiscale from OME 0.2 to 0.3: the target requires information OME 0.2 does not carry

    ```

## Reference

The classes for every NGFF version are documented below, grouped by
version. Only 0.5 carries hand-written prose today; the other versions
render from their signatures, fields and annotations.

### `abczarr.ome.metadata.base`

::: abczarr.ome.metadata.base

<!-- NGFF 0.1 -->

### `abczarr.ome.metadata.v0_1.ome`

::: abczarr.ome.metadata.v0_1.ome

### `abczarr.ome.metadata.v0_1.images`

::: abczarr.ome.metadata.v0_1.images

### `abczarr.ome.metadata.v0_1.plates`

::: abczarr.ome.metadata.v0_1.plates

### `abczarr.ome.metadata.v0_1.wells`

::: abczarr.ome.metadata.v0_1.wells

### `abczarr.ome.metadata.v0_1.labels`

::: abczarr.ome.metadata.v0_1.labels

### `abczarr.ome.metadata.v0_1.omero`

::: abczarr.ome.metadata.v0_1.omero

### `abczarr.ome.metadata.v0_1.version`

::: abczarr.ome.metadata.v0_1.version

<!-- NGFF 0.2 -->

### `abczarr.ome.metadata.v0_2.ome`

::: abczarr.ome.metadata.v0_2.ome

### `abczarr.ome.metadata.v0_2.images`

::: abczarr.ome.metadata.v0_2.images

### `abczarr.ome.metadata.v0_2.plates`

::: abczarr.ome.metadata.v0_2.plates

### `abczarr.ome.metadata.v0_2.wells`

::: abczarr.ome.metadata.v0_2.wells

### `abczarr.ome.metadata.v0_2.labels`

::: abczarr.ome.metadata.v0_2.labels

### `abczarr.ome.metadata.v0_2.omero`

::: abczarr.ome.metadata.v0_2.omero

### `abczarr.ome.metadata.v0_2.version`

::: abczarr.ome.metadata.v0_2.version

<!-- NGFF 0.3 -->

### `abczarr.ome.metadata.v0_3.ome`

::: abczarr.ome.metadata.v0_3.ome

### `abczarr.ome.metadata.v0_3.images`

::: abczarr.ome.metadata.v0_3.images

### `abczarr.ome.metadata.v0_3.plates`

::: abczarr.ome.metadata.v0_3.plates

### `abczarr.ome.metadata.v0_3.wells`

::: abczarr.ome.metadata.v0_3.wells

### `abczarr.ome.metadata.v0_3.labels`

::: abczarr.ome.metadata.v0_3.labels

### `abczarr.ome.metadata.v0_3.omero`

::: abczarr.ome.metadata.v0_3.omero

### `abczarr.ome.metadata.v0_3.version`

::: abczarr.ome.metadata.v0_3.version

<!-- NGFF 0.4 -->

### `abczarr.ome.metadata.v0_4.ome`

::: abczarr.ome.metadata.v0_4.ome

### `abczarr.ome.metadata.v0_4.images`

::: abczarr.ome.metadata.v0_4.images

### `abczarr.ome.metadata.v0_4.axes`

::: abczarr.ome.metadata.v0_4.axes

### `abczarr.ome.metadata.v0_4.transformations`

::: abczarr.ome.metadata.v0_4.transformations

### `abczarr.ome.metadata.v0_4.plates`

::: abczarr.ome.metadata.v0_4.plates

### `abczarr.ome.metadata.v0_4.wells`

::: abczarr.ome.metadata.v0_4.wells

### `abczarr.ome.metadata.v0_4.labels`

::: abczarr.ome.metadata.v0_4.labels

### `abczarr.ome.metadata.v0_4.omero`

::: abczarr.ome.metadata.v0_4.omero

### `abczarr.ome.metadata.v0_4.version`

::: abczarr.ome.metadata.v0_4.version

<!-- NGFF 0.5 -->

### `abczarr.ome.metadata.v0_5.ome`

::: abczarr.ome.metadata.v0_5.ome

### `abczarr.ome.metadata.v0_5.images`

::: abczarr.ome.metadata.v0_5.images

### `abczarr.ome.metadata.v0_5.axes`

::: abczarr.ome.metadata.v0_5.axes

### `abczarr.ome.metadata.v0_5.transformations`

::: abczarr.ome.metadata.v0_5.transformations

### `abczarr.ome.metadata.v0_5.plates`

::: abczarr.ome.metadata.v0_5.plates

### `abczarr.ome.metadata.v0_5.wells`

::: abczarr.ome.metadata.v0_5.wells

### `abczarr.ome.metadata.v0_5.labels`

::: abczarr.ome.metadata.v0_5.labels

### `abczarr.ome.metadata.v0_5.omero`

::: abczarr.ome.metadata.v0_5.omero

### `abczarr.ome.metadata.v0_5.version`

::: abczarr.ome.metadata.v0_5.version

<!-- NGFF 0.6 (dev4 preview) -->

### `abczarr.ome.metadata.v0_6dev4.ome`

::: abczarr.ome.metadata.v0_6dev4.ome

### `abczarr.ome.metadata.v0_6dev4.images`

::: abczarr.ome.metadata.v0_6dev4.images

### `abczarr.ome.metadata.v0_6dev4.systems`

::: abczarr.ome.metadata.v0_6dev4.systems

### `abczarr.ome.metadata.v0_6dev4.transformations`

::: abczarr.ome.metadata.v0_6dev4.transformations

### `abczarr.ome.metadata.v0_6dev4.scenes`

::: abczarr.ome.metadata.v0_6dev4.scenes

### `abczarr.ome.metadata.v0_6dev4.plates`

::: abczarr.ome.metadata.v0_6dev4.plates

### `abczarr.ome.metadata.v0_6dev4.wells`

::: abczarr.ome.metadata.v0_6dev4.wells

### `abczarr.ome.metadata.v0_6dev4.labels`

::: abczarr.ome.metadata.v0_6dev4.labels

### `abczarr.ome.metadata.v0_6dev4.omero`

::: abczarr.ome.metadata.v0_6dev4.omero

### `abczarr.ome.metadata.v0_6dev4.version`

::: abczarr.ome.metadata.v0_6dev4.version
