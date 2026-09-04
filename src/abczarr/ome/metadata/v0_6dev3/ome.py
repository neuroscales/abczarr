__all__ = [
    "OME",
    "OMESeries", "OMEImage", "OMEImageLabel", "OMELabels",
    "OMEPlate", "OMEWell", "OMEScene", "OMEBioformats2Raw"
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Required

# locals
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
    version: Required[Version]


@register_subclass(series=tx.Any)
@autodefine
class OMESeries(OME):
    series: tx.Optional[tx.List[str]] = None


@register_subclass(multiscales=tx.Any)
@autodefine
class OMEImage(OME):
    multiscales: Required[tx.List[Multiscale]]
    omero: tx.Optional[Omero]


@register_subclass(image_label=tx.Any)
@autodefine
class OMEImageLabel(OMEImage):
    image_labels: Required[ImageLabel]


@register_subclass(labels=tx.Any)
@autodefine
class OMELabels(OME):
    labels: Required[tx.List[str]]


@register_subclass(plate=tx.Any)
@autodefine
class OMEPlate(OME):
    plate: Required[Plate]


@register_subclass(well=tx.Any)
@autodefine
class OMEWell(OME):
    well: Required[Well]


@register_subclass(scene=tx.Any)
@autodefine
class OMEScene(OME):
    scene: Required[Scene]


@register_subclass(bioformats2raw_layout=3)
@autodefine
class OMEBioformats2Raw(OME):
    bioformats2raw_layout: Required[tx.Literal[3]]
    plate: Required[Plate]
