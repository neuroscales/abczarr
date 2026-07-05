__all__ = [
    "OMESeries", "OMEImage", "OMEImageLabel", "OMELabels",
    "OMEPlate", "OMEWell", "OMEScene", "OMEBioformats2Raw",
    "OME", "OMEAttributes"
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OME as OMEBase, OMEAttributes as OMEAttributesBase
from .images import Multiscale
from .omero import Omero
from .labels import ImageLabel
from .plates import Plate
from .wells import Well
from .scenes import Scene
from .version import Version

# typing
Required = RequirementForTypedDict.Required
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple


class OMEVersion(OMEBase):
    version: Required[Version]


class OMESeries(OMEVersion):
    series: tx.Optional[List[str]] = None


class OMEImage(OMEVersion):
    multiscales: Required[List[Multiscale]]
    omero: Optional[Omero]


class OMEImageLabel(OMEImage):
    __annotations__ = {
        "image-label": Required[List[ImageLabel]],
    }


class OMELabels(OMEVersion):
    labels: Required[List[str]]


class OMEPlate(OMEVersion):
    plate: Required[Plate]


class OMEWell(OMEVersion):
    well: Required[Well]


class OMEScene(OMEVersion):
    scene: Required[Scene]


class OMEBioformats2Raw(OMEVersion):
    bioformats2raw_layout: Required[tx.Literal[3]]
    plate: Optional[Plate]


OME = tx.Union[
    OMESeries,
    OMEImage,
    OMEImageLabel,
    OMELabels,
    OMEPlate,
    OMEWell,
    OMEScene,
    OMEBioformats2Raw
]


class OMEAttributes(OMEAttributesBase):
    ome: OME
