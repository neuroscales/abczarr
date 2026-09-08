# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

"""The top-level metadata attached to an OME-Zarr group.

[OME][abczarr.ome.v0_6rc0.ome.OME] and its subclasses each describe what
kind of group carries them, an image, a plate, a well, a scene, a
collection of labels, and so on.
"""

__all__ = [
    "OME",
    "OMESeries",
    "OMEImage",
    "OMEImageLabel",
    "OMELabels",
    "OMEPlate",
    "OMEWell",
    "OMEScene",
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
from .scenes import Scene
from .version import VERSION, Version
from .wells import Well


@register_subclass(version=VERSION)
@autodefine
class OME(OMEBase):
    """The version-tagged root of an OME-Zarr group's metadata.

    Every more specific container, such as
    [OMEImage][abczarr.ome.v0_6rc0.ome.OMEImage] or
    [OMEPlate][abczarr.ome.v0_6rc0.ome.OMEPlate], is an `OME` object that
    carries one further field identifying what kind of group it
    describes.
    """

    version: Required[Version] = field(kw_only=True)
    """The OME-NGFF version the metadata is written against."""


@register_subclass(series=tx.Any)
@autodefine
class OMESeries(OME):
    """Marks an OME-Zarr group that collects other groups as a named
    series.
    """

    series: tx.Optional[tx.List[str]] = None
    """The path of each member group, in order, when the group states
    them explicitly. Optional."""


@register_subclass(multiscales=tx.Any)
@autodefine
class OMEImage(OME):
    """Holds an image group's metadata: its multiscale pyramids and
    rendering settings.
    """

    multiscales: Required[tx.List[Multiscale]]
    """The group's
    [Multiscale][abczarr.ome.v0_6rc0.images.Multiscale] pyramids."""
    omero: Optional[Omero]
    """Rendering settings suggesting how to render the pyramids.
    Optional."""


@register_subclass(image_label=tx.Any)
@autodefine
class OMEImageLabel(OMEImage):
    """Marks an image group that is itself a segmentation label image.

    Extends [OMEImage][abczarr.ome.v0_6rc0.ome.OMEImage] with the
    metadata describing the segments the image's pixel values name.
    """

    image_label: Required[ImageLabel] = field(json="image-label")
    """The [ImageLabel][abczarr.ome.v0_6rc0.labels.ImageLabel] metadata
    for the segments."""


@register_subclass(labels=tx.Any)
@autodefine
class OMELabels(OME):
    """Marks a group that collects the label images derived from an
    intensity image.
    """

    labels: Required[tx.List[str]]
    """The path of each label image group."""


@register_subclass(plate=tx.Any)
@autodefine
class OMEPlate(OME):
    """Holds a high-content screening plate group's metadata."""

    plate: Required[Plate]
    """The [Plate][abczarr.ome.v0_6rc0.plates.Plate] metadata describing
    the plate's rows, columns, and wells."""


@register_subclass(well=tx.Any)
@autodefine
class OMEWell(OME):
    """Holds a well group's metadata: the images acquired at one position
    of a plate.
    """

    well: Required[Well]
    """The [Well][abczarr.ome.v0_6rc0.wells.Well] metadata listing the
    well's fields of view."""


@register_subclass(scene=tx.Any)
@autodefine
class OMEScene(OME):
    """Marks a group whose metadata is a scene."""

    scene: Required[Scene]
    """The [Scene][abczarr.ome.v0_6rc0.scenes.Scene] the group carries."""


@register_subclass(bioformats2raw_layout=3)
@autodefine
class OMEBioformats2Raw(OME):
    """The root group of a Bio-Formats2Raw-layout conversion."""

    bioformats2raw_layout: Required[tx.Literal[3]] = field(
        json="bioformats2raw.layout"
    )
    """Marks the layout version. Its value is always `3`."""
    plate: Required[Plate]
    """The [Plate][abczarr.ome.v0_6rc0.plates.Plate] metadata for the
    screening plate the converted data belongs to."""
