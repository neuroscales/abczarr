# OME-Zarr metadata — NGFF 0.4

## Changes from NGFF 0.3

- Axes become typed objects. A new `axes` module introduces
  [Axis][abczarr.ome.metadata.v0_4.axes.Axis] and its kinds —
  [SpaceAxis][abczarr.ome.metadata.v0_4.axes.SpaceAxis],
  [TimeAxis][abczarr.ome.metadata.v0_4.axes.TimeAxis] and
  [ChannelAxis][abczarr.ome.metadata.v0_4.axes.ChannelAxis], each with a
  `name`, a `type` and (for space and time) a `unit`. The bare-string
  axis names of 0.3 are gone;
  [Multiscale][abczarr.ome.metadata.v0_4.images.Multiscale]'s `axes` is
  now a list of these.
- Coordinate transformations are introduced. A new `transformations`
  module adds [Scale][abczarr.ome.metadata.v0_4.transformations.Scale]
  and [Translation][abczarr.ome.metadata.v0_4.transformations.Translation].
  Each [Dataset][abczarr.ome.metadata.v0_4.images.Dataset] now carries a
  required `coordinateTransformations` — a `Scale`, optionally followed
  by a `Translation`, one value per axis — that places the level in
  physical space.
  [Multiscale][abczarr.ome.metadata.v0_4.images.Multiscale] gains an
  optional pyramid-wide `coordinateTransformations`, applied before each
  level's own.

## `abczarr.ome.metadata.v0_4.ome`

::: abczarr.ome.metadata.v0_4.ome

## `abczarr.ome.metadata.v0_4.images`

::: abczarr.ome.metadata.v0_4.images

## `abczarr.ome.metadata.v0_4.axes`

::: abczarr.ome.metadata.v0_4.axes

## `abczarr.ome.metadata.v0_4.transformations`

::: abczarr.ome.metadata.v0_4.transformations

## `abczarr.ome.metadata.v0_4.plates`

::: abczarr.ome.metadata.v0_4.plates

## `abczarr.ome.metadata.v0_4.wells`

::: abczarr.ome.metadata.v0_4.wells

## `abczarr.ome.metadata.v0_4.labels`

::: abczarr.ome.metadata.v0_4.labels

## `abczarr.ome.metadata.v0_4.omero`

::: abczarr.ome.metadata.v0_4.omero

## `abczarr.ome.metadata.v0_4.version`

::: abczarr.ome.metadata.v0_4.version
