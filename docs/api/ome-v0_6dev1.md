# OME-Zarr metadata — NGFF 0.6.dev1

An early 0.6 preview, before the coordinate-systems overhaul; the model may still change.

## Changes from NGFF 0.5

0.6 is a redesign, not an incremental step, so there is no automated
conversion from 0.5 — it begins its own line of previews. The headline
change is that coordinate systems and transformations become
first-class and general.

- Named coordinate systems.
  [Multiscale][abczarr.ome.metadata.v0_6dev1.images.Multiscale] gains a
  required `coordinateSystems`: a list of named
  [CoordinateSystem][abczarr.ome.metadata.v0_6dev1.systems.CoordinateSystem]s,
  each a named set of axes. Axes move into a new `systems` module and
  grow beyond space/time/channel to include
  [ArrayAxis][abczarr.ome.metadata.v0_6dev1.systems.ArrayAxis],
  [DisplacementAxis][abczarr.ome.metadata.v0_6dev1.systems.DisplacementAxis]
  and
  [CoordinateAxis][abczarr.ome.metadata.v0_6dev1.systems.CoordinateAxis],
  with explicit space- and time-unit vocabularies.
- A general transformation model. The fixed scale/translation pair of
  0.5 is replaced by an open list of
  [CoordinateTransformation][abczarr.ome.metadata.v0_6dev1.transformations.CoordinateTransformation]s,
  each naming its `input` and `output` coordinate system. Besides
  [Scale][abczarr.ome.metadata.v0_6dev1.transformations.Scale] and
  [Translation][abczarr.ome.metadata.v0_6dev1.transformations.Translation],
  the module adds
  [Identity][abczarr.ome.metadata.v0_6dev1.transformations.Identity],
  [MapAxis][abczarr.ome.metadata.v0_6dev1.transformations.MapAxis],
  [Affine][abczarr.ome.metadata.v0_6dev1.transformations.Affine],
  [Rotation][abczarr.ome.metadata.v0_6dev1.transformations.Rotation],
  [Sequence][abczarr.ome.metadata.v0_6dev1.transformations.Sequence],
  [ByDimension][abczarr.ome.metadata.v0_6dev1.transformations.ByDimension],
  [Bijection][abczarr.ome.metadata.v0_6dev1.transformations.Bijection],
  [InverseOf][abczarr.ome.metadata.v0_6dev1.transformations.InverseOf],
  [Displacements][abczarr.ome.metadata.v0_6dev1.transformations.Displacements]
  and
  [Coordinates][abczarr.ome.metadata.v0_6dev1.transformations.Coordinates].
- Consequently
  [Dataset][abczarr.ome.metadata.v0_6dev1.images.Dataset]'s
  `coordinateTransformations` is now a required list of these general
  transformations rather than the scale-then-translation pair of 0.4/0.5.

## `abczarr.ome.metadata.v0_6dev1.ome`

::: abczarr.ome.metadata.v0_6dev1.ome

## `abczarr.ome.metadata.v0_6dev1.images`

::: abczarr.ome.metadata.v0_6dev1.images

## `abczarr.ome.metadata.v0_6dev1.systems`

::: abczarr.ome.metadata.v0_6dev1.systems

## `abczarr.ome.metadata.v0_6dev1.transformations`

::: abczarr.ome.metadata.v0_6dev1.transformations

## `abczarr.ome.metadata.v0_6dev1.plates`

::: abczarr.ome.metadata.v0_6dev1.plates

## `abczarr.ome.metadata.v0_6dev1.wells`

::: abczarr.ome.metadata.v0_6dev1.wells

## `abczarr.ome.metadata.v0_6dev1.labels`

::: abczarr.ome.metadata.v0_6dev1.labels

## `abczarr.ome.metadata.v0_6dev1.omero`

::: abczarr.ome.metadata.v0_6dev1.omero

## `abczarr.ome.metadata.v0_6dev1.version`

::: abczarr.ome.metadata.v0_6dev1.version
