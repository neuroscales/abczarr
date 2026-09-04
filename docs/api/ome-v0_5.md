# OME-Zarr metadata — NGFF 0.5

## Changes from NGFF 0.4

- The per-object `version` field is dropped from every payload carrier —
  [Multiscale][abczarr.ome.metadata.v0_5.images.Multiscale],
  [ImageLabel][abczarr.ome.metadata.v0_5.labels.ImageLabel],
  [Omero][abczarr.ome.metadata.v0_5.omero.Omero],
  [Plate][abczarr.ome.metadata.v0_5.plates.Plate] and
  [Well][abczarr.ome.metadata.v0_5.wells.Well]. The version is now
  recorded once, on the top-level
  [OME][abczarr.ome.metadata.v0_5.ome.OME] container, where it
  discriminates the metadata version.
- At the storage layer, 0.5 nests all OME metadata under a single
  `"ome"` key in a group's attributes; 0.4 and earlier wrote the same
  fields at the top level of `attrs`.

## `abczarr.ome.metadata.v0_5.ome`

::: abczarr.ome.metadata.v0_5.ome

## `abczarr.ome.metadata.v0_5.images`

::: abczarr.ome.metadata.v0_5.images

## `abczarr.ome.metadata.v0_5.axes`

::: abczarr.ome.metadata.v0_5.axes

## `abczarr.ome.metadata.v0_5.transformations`

::: abczarr.ome.metadata.v0_5.transformations

## `abczarr.ome.metadata.v0_5.plates`

::: abczarr.ome.metadata.v0_5.plates

## `abczarr.ome.metadata.v0_5.wells`

::: abczarr.ome.metadata.v0_5.wells

## `abczarr.ome.metadata.v0_5.labels`

::: abczarr.ome.metadata.v0_5.labels

## `abczarr.ome.metadata.v0_5.omero`

::: abczarr.ome.metadata.v0_5.omero

## `abczarr.ome.metadata.v0_5.version`

::: abczarr.ome.metadata.v0_5.version
