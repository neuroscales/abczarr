# OME-Zarr metadata — NGFF 0.6rc0

The 0.6 release candidate; the model may still change.

## Changes from NGFF 0.6.dev4

- A new
  [ProjectAxis][abczarr.ome.metadata.v0_6rc0.transformations.ProjectAxis]
  transformation is added (`type: "projectAxis"`, carrying
  `createdOutputs` and `droppedInputs`), for adding or dropping axes.
- [ByDimension][abczarr.ome.metadata.v0_6rc0.transformations.ByDimension]'s
  inner per-dimension axis keys are re-spelled to camelCase: `input_axes`
  becomes `inputAxes` and `output_axes` becomes `outputAxes`.

## `abczarr.ome.metadata.v0_6rc0.ome`

::: abczarr.ome.metadata.v0_6rc0.ome

## `abczarr.ome.metadata.v0_6rc0.images`

::: abczarr.ome.metadata.v0_6rc0.images

## `abczarr.ome.metadata.v0_6rc0.systems`

::: abczarr.ome.metadata.v0_6rc0.systems

## `abczarr.ome.metadata.v0_6rc0.transformations`

::: abczarr.ome.metadata.v0_6rc0.transformations

## `abczarr.ome.metadata.v0_6rc0.scenes`

::: abczarr.ome.metadata.v0_6rc0.scenes

## `abczarr.ome.metadata.v0_6rc0.plates`

::: abczarr.ome.metadata.v0_6rc0.plates

## `abczarr.ome.metadata.v0_6rc0.wells`

::: abczarr.ome.metadata.v0_6rc0.wells

## `abczarr.ome.metadata.v0_6rc0.labels`

::: abczarr.ome.metadata.v0_6rc0.labels

## `abczarr.ome.metadata.v0_6rc0.omero`

::: abczarr.ome.metadata.v0_6rc0.omero

## `abczarr.ome.metadata.v0_6rc0.version`

::: abczarr.ome.metadata.v0_6rc0.version
