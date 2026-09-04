# OME-Zarr metadata — NGFF 0.3

## Changes from NGFF 0.2

- [Multiscale][abczarr.ome.metadata.v0_3.images.Multiscale] gains a
  required `axes` field. A pyramid's dimensions are now listed
  explicitly and in order, instead of being left implicit in the array
  shape. The axis names are still bare string literals (`x`, `y`, `z`,
  `t`, `c`); typed axis objects arrive in 0.4.
- `version` is promoted from recommended to required on every carrier
  that records it:
  [Multiscale][abczarr.ome.metadata.v0_3.images.Multiscale],
  [ImageLabel][abczarr.ome.metadata.v0_3.labels.ImageLabel],
  [Omero][abczarr.ome.metadata.v0_3.omero.Omero],
  [Plate][abczarr.ome.metadata.v0_3.plates.Plate] and
  [Well][abczarr.ome.metadata.v0_3.wells.Well].

## `abczarr.ome.metadata.v0_3.ome`

::: abczarr.ome.metadata.v0_3.ome

## `abczarr.ome.metadata.v0_3.images`

::: abczarr.ome.metadata.v0_3.images

## `abczarr.ome.metadata.v0_3.plates`

::: abczarr.ome.metadata.v0_3.plates

## `abczarr.ome.metadata.v0_3.wells`

::: abczarr.ome.metadata.v0_3.wells

## `abczarr.ome.metadata.v0_3.labels`

::: abczarr.ome.metadata.v0_3.labels

## `abczarr.ome.metadata.v0_3.omero`

::: abczarr.ome.metadata.v0_3.omero

## `abczarr.ome.metadata.v0_3.version`

::: abczarr.ome.metadata.v0_3.version
