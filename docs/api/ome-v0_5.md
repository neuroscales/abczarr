# OME-Zarr metadata — NGFF 0.5

## Changes from NGFF 0.4

- The per-object `version` field is dropped from every payload carrier —
  [Multiscale][abczarr.ome.v0_5.images.Multiscale],
  [ImageLabel][abczarr.ome.v0_5.labels.ImageLabel],
  [Omero][abczarr.ome.v0_5.omero.Omero],
  [Plate][abczarr.ome.v0_5.plates.Plate] and
  [Well][abczarr.ome.v0_5.wells.Well]. The version is now
  recorded once, on the top-level
  [OME][abczarr.ome.v0_5.ome.OME] container, where it
  discriminates the metadata version.
- At the storage layer, 0.5 nests all OME metadata under a single
  `"ome"` key in a group's attributes; 0.4 and earlier wrote the same
  fields at the top level of `attrs`.

## `abczarr.ome.v0_5.ome`

::: abczarr.ome.v0_5.ome

## `abczarr.ome.v0_5.images`

::: abczarr.ome.v0_5.images

## `abczarr.ome.v0_5.axes`

::: abczarr.ome.v0_5.axes

## `abczarr.ome.v0_5.transformations`

::: abczarr.ome.v0_5.transformations

## `abczarr.ome.v0_5.plates`

::: abczarr.ome.v0_5.plates

## `abczarr.ome.v0_5.wells`

::: abczarr.ome.v0_5.wells

## `abczarr.ome.v0_5.labels`

::: abczarr.ome.v0_5.labels

## `abczarr.ome.v0_5.omero`

::: abczarr.ome.v0_5.omero

## `abczarr.ome.v0_5.version`

::: abczarr.ome.v0_5.version
