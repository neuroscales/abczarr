# OME-Zarr metadata — NGFF 0.6.dev2

An early 0.6 preview. The model may still change.

## Changes from NGFF 0.6.dev1

- [MapAxis][abczarr.ome.v0_6dev2.transformations.MapAxis] addresses
  axes by index. Its `mapAxis` field is a list of integers, and entry
  `k` gives the input axis that feeds output axis `k`. The list must be
  a permutation. In 0.6.dev1, `mapAxis` mapped output axis names to
  input axis names.
- Each child of a
  [ByDimension][abczarr.ome.v0_6dev2.transformations.ByDimension]
  names its axes in `input_axes` and `output_axes`. These fields still
  hold axis names. In 0.6.dev1, the same fields were called `input` and
  `output`.

## `abczarr.ome.v0_6dev2.ome`

::: abczarr.ome.v0_6dev2.ome

## `abczarr.ome.v0_6dev2.images`

::: abczarr.ome.v0_6dev2.images

## `abczarr.ome.v0_6dev2.systems`

::: abczarr.ome.v0_6dev2.systems

## `abczarr.ome.v0_6dev2.transformations`

::: abczarr.ome.v0_6dev2.transformations

## `abczarr.ome.v0_6dev2.plates`

::: abczarr.ome.v0_6dev2.plates

## `abczarr.ome.v0_6dev2.wells`

::: abczarr.ome.v0_6dev2.wells

## `abczarr.ome.v0_6dev2.labels`

::: abczarr.ome.v0_6dev2.labels

## `abczarr.ome.v0_6dev2.omero`

::: abczarr.ome.v0_6dev2.omero

## `abczarr.ome.v0_6dev2.version`

::: abczarr.ome.v0_6dev2.version
