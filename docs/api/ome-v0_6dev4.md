# OME-Zarr metadata — NGFF 0.6.dev4

A preview of the next NGFF release; the model may still change.

## Changes from NGFF 0.6.dev3

- The transform input/output model is overhauled. A
  [CoordinateTransformation][abczarr.ome.metadata.v0_6dev4.transformations.CoordinateTransformation]'s
  `input` and `output` are no longer free-form coordinate-system name
  strings but a structured
  [Space][abczarr.ome.metadata.v0_6dev4.transformations.Space] object (a
  `name` and/or a `path`), and are now recommended fields.
- Alongside that, the matrix transforms are given concrete shapes:
  [Affine][abczarr.ome.metadata.v0_6dev4.transformations.Affine]'s
  `affine` and
  [Rotation][abczarr.ome.metadata.v0_6dev4.transformations.Rotation]'s
  `rotation` become nested float lists, and `path` becomes required on
  [Displacements][abczarr.ome.metadata.v0_6dev4.transformations.Displacements]
  and
  [Coordinates][abczarr.ome.metadata.v0_6dev4.transformations.Coordinates].

## `abczarr.ome.metadata.v0_6dev4.ome`

::: abczarr.ome.metadata.v0_6dev4.ome

## `abczarr.ome.metadata.v0_6dev4.images`

::: abczarr.ome.metadata.v0_6dev4.images

## `abczarr.ome.metadata.v0_6dev4.systems`

::: abczarr.ome.metadata.v0_6dev4.systems

## `abczarr.ome.metadata.v0_6dev4.transformations`

::: abczarr.ome.metadata.v0_6dev4.transformations

## `abczarr.ome.metadata.v0_6dev4.scenes`

::: abczarr.ome.metadata.v0_6dev4.scenes

## `abczarr.ome.metadata.v0_6dev4.plates`

::: abczarr.ome.metadata.v0_6dev4.plates

## `abczarr.ome.metadata.v0_6dev4.wells`

::: abczarr.ome.metadata.v0_6dev4.wells

## `abczarr.ome.metadata.v0_6dev4.labels`

::: abczarr.ome.metadata.v0_6dev4.labels

## `abczarr.ome.metadata.v0_6dev4.omero`

::: abczarr.ome.metadata.v0_6dev4.omero

## `abczarr.ome.metadata.v0_6dev4.version`

::: abczarr.ome.metadata.v0_6dev4.version
