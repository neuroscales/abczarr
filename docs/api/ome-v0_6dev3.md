# OME-Zarr metadata — NGFF 0.6.dev3

A 0.6 preview carrying the reworked coordinate-systems model; the model may still change.

## Changes from NGFF 0.6.dev2

- Each child of a
  [ByDimension][abczarr.ome.v0_6dev3.transformations.ByDimension]
  is wrapped in an object that carries `transformation`, `input_axes`
  and `output_axes`. The `input_axes` and `output_axes` fields hold
  axis indices. In 0.6.dev2, these fields held axis names.
- The `inverseOf` transformation is removed.
- Scenes are introduced. A new `scenes` module adds
  [Scene][abczarr.ome.v0_6dev3.scenes.Scene], and a new
  top-level [OMEScene][abczarr.ome.v0_6dev3.ome.OMEScene]
  container carries one.

## `abczarr.ome.v0_6dev3.ome`

::: abczarr.ome.v0_6dev3.ome

## `abczarr.ome.v0_6dev3.images`

::: abczarr.ome.v0_6dev3.images

## `abczarr.ome.v0_6dev3.systems`

::: abczarr.ome.v0_6dev3.systems

## `abczarr.ome.v0_6dev3.transformations`

::: abczarr.ome.v0_6dev3.transformations

## `abczarr.ome.v0_6dev3.scenes`

::: abczarr.ome.v0_6dev3.scenes

## `abczarr.ome.v0_6dev3.plates`

::: abczarr.ome.v0_6dev3.plates

## `abczarr.ome.v0_6dev3.wells`

::: abczarr.ome.v0_6dev3.wells

## `abczarr.ome.v0_6dev3.labels`

::: abczarr.ome.v0_6dev3.labels

## `abczarr.ome.v0_6dev3.omero`

::: abczarr.ome.v0_6dev3.omero

## `abczarr.ome.v0_6dev3.version`

::: abczarr.ome.v0_6dev3.version
