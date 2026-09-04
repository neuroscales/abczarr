# OME-Zarr metadata — NGFF 0.1

## The first NGFF metadata version

0.1 is where the OME-Zarr vocabulary starts. It establishes the
multiscale image and the container objects that carry it; every later
version is a change against this baseline.

- A multiscale image is a
  [Multiscale][abczarr.ome.v0_1.images.Multiscale]: an ordered
  list of [Dataset][abczarr.ome.v0_1.images.Dataset]
  resolution levels, full resolution first, each naming a Zarr array by
  `path`.
- Dimensions are implicit. There is no axes list and no
  coordinate-transformation model yet; axis names are single-letter
  string literals (`t`, `c`, `z`, `y`, `x`).
- The top-level container is
  [OME][abczarr.ome.v0_1.ome.OME], discriminated by the
  payload it carries:
  [OMEImage][abczarr.ome.v0_1.ome.OMEImage] (multiscales,
  optionally with [Omero][abczarr.ome.v0_1.omero.Omero]
  rendering settings),
  [OMEImageLabel][abczarr.ome.v0_1.ome.OMEImageLabel] for
  segmentation [ImageLabel][abczarr.ome.v0_1.labels.ImageLabel]s,
  and [OMEPlate][abczarr.ome.v0_1.ome.OMEPlate] /
  [OMEWell][abczarr.ome.v0_1.ome.OMEWell] for high-content
  screens ([Plate][abczarr.ome.v0_1.plates.Plate] /
  [Well][abczarr.ome.v0_1.wells.Well]), alongside
  [OMESeries][abczarr.ome.v0_1.ome.OMESeries],
  [OMELabels][abczarr.ome.v0_1.ome.OMELabels] and
  [OMEBioformats2Raw][abczarr.ome.v0_1.ome.OMEBioformats2Raw].

## `abczarr.ome.v0_1.ome`

::: abczarr.ome.v0_1.ome

## `abczarr.ome.v0_1.images`

::: abczarr.ome.v0_1.images

## `abczarr.ome.v0_1.plates`

::: abczarr.ome.v0_1.plates

## `abczarr.ome.v0_1.wells`

::: abczarr.ome.v0_1.wells

## `abczarr.ome.v0_1.labels`

::: abczarr.ome.v0_1.labels

## `abczarr.ome.v0_1.omero`

::: abczarr.ome.v0_1.omero

## `abczarr.ome.v0_1.version`

::: abczarr.ome.v0_1.version
