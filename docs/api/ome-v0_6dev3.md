# OME-Zarr metadata — NGFF 0.6.dev3

A 0.6 preview carrying the reworked coordinate-systems model; the model may still change.

## Changes from NGFF 0.6.dev2

- The transformation model is reworked.
  [MapAxis][abczarr.ome.metadata.v0_6dev3.transformations.MapAxis]'s
  `mapAxis` becomes an index list (`List[int]`) instead of a
  name-to-name mapping.
  [ByDimension][abczarr.ome.metadata.v0_6dev3.transformations.ByDimension]
  now wraps each per-dimension entry in a nested `Transformation`
  (carrying `transformation`, `input_axes` and `output_axes`) rather
  than a flat transformation list. The `inverseOf` transformation is
  dropped.
- Scenes are introduced. A new `scenes` module adds
  [Scene][abczarr.ome.metadata.v0_6dev3.scenes.Scene], and a new
  top-level [OMEScene][abczarr.ome.metadata.v0_6dev3.ome.OMEScene]
  container carries one.

## `abczarr.ome.metadata.v0_6dev3.ome`

::: abczarr.ome.metadata.v0_6dev3.ome

## `abczarr.ome.metadata.v0_6dev3.images`

::: abczarr.ome.metadata.v0_6dev3.images

## `abczarr.ome.metadata.v0_6dev3.systems`

::: abczarr.ome.metadata.v0_6dev3.systems

## `abczarr.ome.metadata.v0_6dev3.transformations`

::: abczarr.ome.metadata.v0_6dev3.transformations

## `abczarr.ome.metadata.v0_6dev3.scenes`

::: abczarr.ome.metadata.v0_6dev3.scenes

## `abczarr.ome.metadata.v0_6dev3.plates`

::: abczarr.ome.metadata.v0_6dev3.plates

## `abczarr.ome.metadata.v0_6dev3.wells`

::: abczarr.ome.metadata.v0_6dev3.wells

## `abczarr.ome.metadata.v0_6dev3.labels`

::: abczarr.ome.metadata.v0_6dev3.labels

## `abczarr.ome.metadata.v0_6dev3.omero`

::: abczarr.ome.metadata.v0_6dev3.omero

## `abczarr.ome.metadata.v0_6dev3.version`

::: abczarr.ome.metadata.v0_6dev3.version
