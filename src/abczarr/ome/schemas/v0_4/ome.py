__all__ = [
    "OMESeries", "OMEImage", "OMEImageLabel", "OMELabels",
    "OMEPlate", "OMEWell", "OMEBioformats2Raw",
    "OME", "OMEAttributes"
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OME as OMEBase
from .images import Multiscale
from .omero import Omero
from .labels import ImageLabel
from .plates import Plate
from .wells import Well

# typing
Required = RequirementForTypedDict.Required
Optional = RequirementForTypedDict.Optional
List = tz.BuiltinSequence  # list | tuple


class OMESeries(OMEBase):
    series: tx.Optional[List[str]] = None


class OMEImage(OMEBase):
    multiscales: Required[List[Multiscale]]
    omero: Optional[Omero]


class OMEImageLabel(OMEImage):
    __annotations__ = {
        "image-label": Required[List[ImageLabel]],
    }


class OMELabels(OMEBase):
    labels: Required[List[str]]


class OMEPlate(OMEBase):
    plate: Required[Plate]


class OMEWell(OMEBase):
    well: Required[Well]


class OMEBioformats2Raw(OMEBase):
    bioformats2raw_layout: Required[tx.Literal[3]]
    plate: Optional[Plate]


OMEAttributes = OME = tx.Union[
    OMESeries,
    OMEImage,
    OMEImageLabel,
    OMELabels,
    OMEPlate,
    OMEWell,
    OMEBioformats2Raw
]
