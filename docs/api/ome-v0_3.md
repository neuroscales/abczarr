# OME-Zarr metadata — NGFF 0.3

## Changes from NGFF 0.2

- [Multiscale][abczarr.ome.v0_3.images.Multiscale] gains a
  required `axes` field. A pyramid's dimensions are now listed
  explicitly and in order, instead of being left implicit in the array
  shape. The axis names are still bare string literals (`x`, `y`, `z`,
  `t`, `c`); typed axis objects arrive in 0.4.
- `version` is promoted from recommended to required on every carrier
  that records it:
  [Multiscale][abczarr.ome.v0_3.images.Multiscale],
  [ImageLabel][abczarr.ome.v0_3.labels.ImageLabel],
  [Omero][abczarr.ome.v0_3.omero.Omero],
  [Plate][abczarr.ome.v0_3.plates.Plate] and
  [Well][abczarr.ome.v0_3.wells.Well].

## `abczarr.ome.v0_3.ome`

::: abczarr.ome.v0_3.ome

## `abczarr.ome.v0_3.images`

::: abczarr.ome.v0_3.images

## `abczarr.ome.v0_3.plates`

::: abczarr.ome.v0_3.plates

## `abczarr.ome.v0_3.wells`

::: abczarr.ome.v0_3.wells

## `abczarr.ome.v0_3.labels`

::: abczarr.ome.v0_3.labels

## `abczarr.ome.v0_3.omero`

::: abczarr.ome.v0_3.omero

## `abczarr.ome.v0_3.version`

::: abczarr.ome.v0_3.version
