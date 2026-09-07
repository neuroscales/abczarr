# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

__all__ = [
    "OME",
    "OMESeries",
    "OMEImage",
    "OMEImageLabel",
    "OMELabels",
    "OMEPlate",
    "OMEWell",
    "OMEBioformats2Raw",
]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Optional, Required

from ..base import OME as OMEBase
from .images import Multiscale
from .labels import ImageLabel
from .omero import Omero
from .plates import Plate
from .version import VERSION, Version
from .wells import Well


@register_subclass(version=VERSION)
@autodefine
class OME(OMEBase):
    """This class represents the OME-Zarr metadata attached to a Zarr group.

    An image, a plate, a well, and a set of labels are each
    represented by a different subclass that adds the fields for
    what it holds. See [OMEImage][abczarr.ome.v0_5.ome.OMEImage],
    [OMEPlate][abczarr.ome.v0_5.ome.OMEPlate], and
    [OMEWell][abczarr.ome.v0_5.ome.OMEWell] for examples. The
    `version` field records which OME-NGFF version the metadata is
    written against.
    """

    version: Required[Version]


@register_subclass(series=tx.Any)
@autodefine
class OMESeries(OME):
    """An `OMESeries` object holds the metadata for a bioformats2raw
    dataset's set of image series.

    The `series` field names each series' subgroup. The subgroup
    names appear in the same order as the corresponding series in
    the accompanying OME-XML document.
    """

    series: tx.Optional[tx.List[str]] = None


@register_subclass(multiscales=tx.Any)
@autodefine
class OMEImage(OME):
    """An `OMEImage` object represents an image group.

    The group holds one or more multiscale pyramids and, optionally,
    settings for how the image should be rendered. The `multiscales`
    field lists the group's resolution pyramids. Each pyramid is a
    [Multiscale][abczarr.ome.v0_5.images.Multiscale] object, and an
    image almost always has exactly one. The `omero` field, when
    present, suggests how a viewer should display the image's
    channels. See [Omero][abczarr.ome.v0_5.omero.Omero] for the
    fields it carries.
    """

    multiscales: Required[tx.List[Multiscale]]
    omero: Optional[Omero]


@register_subclass(image_label=tx.Any)
@autodefine
class OMEImageLabel(OMEImage):
    """An `OMEImageLabel` object represents a label image.

    A label image is an [OMEImage][abczarr.ome.v0_5.ome.OMEImage]
    whose pixel values name segments. The `image_label` field
    carries the [ImageLabel][abczarr.ome.v0_5.labels.ImageLabel]
    metadata that describes those segments, including their display
    colors and any per-label properties.
    """

    image_label: Required[ImageLabel] = field(json="image-label")


@register_subclass(labels=tx.Any)
@autodefine
class OMELabels(OME):
    """An `OMELabels` object represents a group that collects a set
    of label images.

    The `labels` field names the subgroups that hold the label
    images. Each of those subgroups is in turn represented by an
    [OMEImageLabel][abczarr.ome.v0_5.ome.OMEImageLabel] object.
    """

    labels: Required[tx.List[str]]


@register_subclass(plate=tx.Any)
@autodefine
class OMEPlate(OME):
    """An `OMEPlate` object represents a high-content screening
    plate group.

    The `plate` field carries the
    [Plate][abczarr.ome.v0_5.plates.Plate] metadata, which describes
    the plate's rows, columns, wells, and imaging runs.
    """

    plate: Required[Plate]


@register_subclass(well=tx.Any)
@autodefine
class OMEWell(OME):
    """An `OMEWell` object represents a well group.

    A well group holds the images captured at one position of a
    plate. The `well` field carries the
    [Well][abczarr.ome.v0_5.wells.Well] metadata, which describes
    the group's fields of view and, for each field of view, the
    acquisition run it belongs to.
    """

    well: Required[Well]


@register_subclass(bioformats2raw_layout=3)
@autodefine
class OMEBioformats2Raw(OME):
    """An `OMEBioformats2Raw` object represents the root group of a
    bioformats2raw-converted dataset.

    The `bioformats2raw_layout` field marks the group as following
    that layout. Its value is always `3`. When the converted
    document describes a screening plate, the `plate` field carries
    the [Plate][abczarr.ome.v0_5.plates.Plate] metadata for that
    plate.
    """

    bioformats2raw_layout: Required[tx.Literal[3]] = field(
        json="bioformats2raw.layout"
    )
    plate: Required[Plate]
