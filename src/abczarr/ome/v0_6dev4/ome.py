# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

"""The top-level metadata attached to an OME-Zarr group.

[OME][abczarr.ome.v0_6dev4.ome.OME] and its subclasses each describe what
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
    [OMEImage][abczarr.ome.v0_6dev4.ome.OMEImage] or
    [OMEPlate][abczarr.ome.v0_6dev4.ome.OMEPlate], is an `OME` object that
    carries one further field identifying what kind of group it
    describes.

    Parameters
    ----------
    version : Version
        The OME-NGFF version the metadata is written against.
    """

    version: Required[Version] = field(kw_only=True)


@register_subclass(series=tx.Any)
@autodefine
class OMESeries(OME):
    """Marks an OME-Zarr group that collects other groups as a named series.

    Parameters
    ----------
    series : list of str
        The path of each member group, in order, when the group states
        them explicitly. Optional.
    """

    series: tx.Optional[tx.List[str]] = None


@register_subclass(multiscales=tx.Any)
@autodefine
class OMEImage(OME):
    """Holds an image group's metadata: its multiscale pyramids and
    rendering settings.

    Parameters
    ----------
    multiscales : list of Multiscale
        The group's
        [Multiscale][abczarr.ome.v0_6dev4.images.Multiscale] pyramids.
    omero : Omero
        Rendering settings suggesting how to render the pyramids.
        Optional.
    """

    multiscales: Required[tx.List[Multiscale]]
    omero: Optional[Omero]


@register_subclass(image_label=tx.Any)
@autodefine
class OMEImageLabel(OMEImage):
    """Marks an image group that is itself a segmentation label image.

    Extends [OMEImage][abczarr.ome.v0_6dev4.ome.OMEImage] with the
    metadata describing the segments the image's pixel values name.

    Parameters
    ----------
    image_label : ImageLabel
        The [ImageLabel][abczarr.ome.v0_6dev4.labels.ImageLabel] metadata
        for the segments.
    """

    image_label: Required[ImageLabel] = field(json="image-label")


@register_subclass(labels=tx.Any)
@autodefine
class OMELabels(OME):
    """Marks a group that collects the label images derived from an
    intensity image.

    Parameters
    ----------
    labels : list of str
        The path of each label image group.
    """

    labels: Required[tx.List[str]]


@register_subclass(plate=tx.Any)
@autodefine
class OMEPlate(OME):
    """Holds a high-content screening plate group's metadata.

    Parameters
    ----------
    plate : Plate
        The [Plate][abczarr.ome.v0_6dev4.plates.Plate] metadata describing
        the plate's rows, columns, and wells.
    """

    plate: Required[Plate]


@register_subclass(well=tx.Any)
@autodefine
class OMEWell(OME):
    """Holds a well group's metadata: the images acquired at one position
    of a plate.

    Parameters
    ----------
    well : Well
        The [Well][abczarr.ome.v0_6dev4.wells.Well] metadata listing the
        well's fields of view.
    """

    well: Required[Well]


@register_subclass(scene=tx.Any)
@autodefine
class OMEScene(OME):
    """Marks a group whose metadata is a scene.

    Parameters
    ----------
    scene : Scene
        The [Scene][abczarr.ome.v0_6dev4.scenes.Scene] the group carries.
    """

    scene: Required[Scene]


@register_subclass(bioformats2raw_layout=3)
@autodefine
class OMEBioformats2Raw(OME):
    """The root group of a Bio-Formats2Raw-layout conversion.

    Parameters
    ----------
    bioformats2raw_layout : int
        Marks the layout version. Its value is always `3`.
    plate : Plate
        The [Plate][abczarr.ome.v0_6dev4.plates.Plate] metadata for the
        screening plate the converted data belongs to.
    """

    bioformats2raw_layout: Required[tx.Literal[3]] = field(
        json="bioformats2raw.layout"
    )
    plate: Required[Plate]
