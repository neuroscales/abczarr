# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

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
    """An `OME` object is the version-tagged root of an OME-Zarr group's
    metadata.

    Every more specific container, such as
    [OMEImage][abczarr.ome.v0_6dev4.ome.OMEImage] or
    [OMEPlate][abczarr.ome.v0_6dev4.ome.OMEPlate], is an `OME` object that
    carries one further field identifying what kind of group it describes.
    """

    version: Required[Version]


@register_subclass(series=tx.Any)
@autodefine
class OMESeries(OME):
    """An `OMESeries` object marks an OME-Zarr group that collects other groups
    as a named series.

    `series` lists the paths of the member groups, in order, when the group
    states them explicitly.
    """

    series: tx.Optional[tx.List[str]] = None


@register_subclass(multiscales=tx.Any)
@autodefine
class OMEImage(OME):
    """An `OMEImage` object holds an image group's metadata: its multiscale
    pyramids and rendering settings.

    `multiscales` lists the group's
    [Multiscale][abczarr.ome.v0_6dev4.images.Multiscale] pyramids. `omero`,
    when given, suggests how to render them.
    """

    multiscales: Required[tx.List[Multiscale]]
    omero: Optional[Omero]


@register_subclass(image_label=tx.Any)
@autodefine
class OMEImageLabel(OMEImage):
    """An `OMEImageLabel` object marks an image group that is itself a
    segmentation label image.

    It extends [OMEImage][abczarr.ome.v0_6dev4.ome.OMEImage] with
    `image_label`, the [ImageLabel][abczarr.ome.v0_6dev4.labels.ImageLabel]
    that describes the segments its pixel values name.
    """

    image_label: Required[ImageLabel] = field(json="image-label")


@register_subclass(labels=tx.Any)
@autodefine
class OMELabels(OME):
    """An `OMELabels` object marks a group that collects the label images
    derived from an intensity image.

    `labels` lists the paths of those label image groups.
    """

    labels: Required[tx.List[str]]


@register_subclass(plate=tx.Any)
@autodefine
class OMEPlate(OME):
    """An `OMEPlate` object holds a high-content screening plate group's
    metadata.

    `plate` is the [Plate][abczarr.ome.v0_6dev4.plates.Plate] that describes
    the plate's rows, columns, and wells.
    """

    plate: Required[Plate]


@register_subclass(well=tx.Any)
@autodefine
class OMEWell(OME):
    """An `OMEWell` object holds a well group's metadata: the images acquired
    at one position of a plate.

    `well` is the [Well][abczarr.ome.v0_6dev4.wells.Well] that lists the
    well's fields of view.
    """

    well: Required[Well]


@register_subclass(scene=tx.Any)
@autodefine
class OMEScene(OME):
    """An `OMEScene` object marks a group whose metadata is a scene.

    `scene` is the [Scene][abczarr.ome.v0_6dev4.scenes.Scene] that it
    carries.
    """

    scene: Required[Scene]


@register_subclass(bioformats2raw_layout=3)
@autodefine
class OMEBioformats2Raw(OME):
    """An `OMEBioformats2Raw` object is the root group of a
    Bio-Formats2Raw-layout conversion.

    `bioformats2raw_layout` marks the layout version. `plate` is the
    [Plate][abczarr.ome.v0_6dev4.plates.Plate] the converted data belongs
    to.
    """

    bioformats2raw_layout: Required[tx.Literal[3]] = field(
        json="bioformats2raw.layout"
    )
    plate: Required[Plate]
