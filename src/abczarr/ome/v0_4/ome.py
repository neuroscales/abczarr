# Generated from v0_1 by tools/gen_ome_metadata.py -- do not edit

"""The top-level metadata attached to an OME-Zarr group.

[OME][abczarr.ome.v0_4.ome.OME] and its subclasses each describe what
kind of group carries them, an image, a plate, a well, a collection of
labels, and so on.
"""

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
    """OME-Zarr metadata attached to a Zarr group.

    An image, a plate, a well, and a set of labels are each represented
    by a different subclass that adds the fields for what it holds. See
    [OMEImage][abczarr.ome.v0_4.ome.OMEImage],
    [OMEPlate][abczarr.ome.v0_4.ome.OMEPlate], and
    [OMEWell][abczarr.ome.v0_4.ome.OMEWell] for examples.
    """

    version: Required[Version] = field(kw_only=True)
    """The OME-NGFF version the metadata is written against."""


@register_subclass(series=tx.Any)
@autodefine
class OMESeries(OME):
    """Metadata for a bioformats2raw dataset's set of image series."""

    series: tx.Optional[tx.List[str]] = None
    """The path of each series' subgroup, in the same order as the
    corresponding series in the accompanying OME-XML document.
    Optional."""


@register_subclass(multiscales=tx.Any)
@autodefine
class OMEImage(OME):
    """Metadata for an image group.

    The group holds one or more multiscale pyramids and, optionally,
    settings for how the image should be rendered.
    """

    multiscales: Required[tx.List[Multiscale]]
    """The group's resolution pyramids, each a
    [Multiscale][abczarr.ome.v0_4.images.Multiscale] object. An image
    almost always has exactly one."""
    omero: Optional[Omero]
    """Rendering settings suggesting how a viewer should display the
    image's channels. See [Omero][abczarr.ome.v0_4.omero.Omero] for the
    fields it carries. Optional."""


@register_subclass(image_label=tx.Any)
@autodefine
class OMEImageLabel(OMEImage):
    """Metadata for a label image.

    A label image is an [OMEImage][abczarr.ome.v0_4.ome.OMEImage] whose
    pixel values name segments.
    """

    image_label: Required[ImageLabel] = field(json="image-label")
    """The [ImageLabel][abczarr.ome.v0_4.labels.ImageLabel] metadata
    describing those segments, including their display colors and any
    per-label properties."""


@register_subclass(labels=tx.Any)
@autodefine
class OMELabels(OME):
    """Metadata for a group that collects a set of label images."""

    labels: Required[tx.List[str]]
    """The path of each subgroup that holds a label image. Each such
    subgroup is in turn represented by an
    [OMEImageLabel][abczarr.ome.v0_4.ome.OMEImageLabel] object."""


@register_subclass(plate=tx.Any)
@autodefine
class OMEPlate(OME):
    """Metadata for a high-content screening plate group."""

    plate: Required[Plate]
    """The [Plate][abczarr.ome.v0_4.plates.Plate] metadata describing
    the plate's rows, columns, wells, and imaging runs."""


@register_subclass(well=tx.Any)
@autodefine
class OMEWell(OME):
    """Metadata for a well group.

    A well group holds the images captured at one position of a plate.
    """

    well: Required[Well]
    """The [Well][abczarr.ome.v0_4.wells.Well] metadata describing the
    group's fields of view. For each field of view, that metadata also
    names the acquisition run it belongs to."""


@register_subclass(bioformats2raw_layout=3)
@autodefine
class OMEBioformats2Raw(OME):
    """Metadata for the root group of a bioformats2raw-converted dataset."""

    bioformats2raw_layout: Required[tx.Literal[3]] = field(
        json="bioformats2raw.layout"
    )
    """Marks the group as following the bioformats2raw layout. Its
    value is always `3`."""
    plate: Required[Plate]
    """The [Plate][abczarr.ome.v0_4.plates.Plate] metadata, present when
    the converted dataset describes a screening plate. This field is
    optional in OME-NGFF 0.1, and required from 0.2 on."""
