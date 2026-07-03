__all__ = [
    "OME",
    "OMESeries", "OMEImage", "OMEImageLabel", "OMELabels",
    "OMEPlate", "OMEWell", "OMEScene", "OMEBioformats2Raw"
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.metadata import FlexibleMetadata, register_subclass

# locals
from .image import Multiscale
from .omero import Omero
from .labels import ImageLabel
from .plates import Plate
from .wells import Well
from .scenes import Scene


@autodefine
class OME(FlexibleMetadata):
    version: tx.Literal["0.6.dev4"]


@register_subclass(series=tx.Any)
@autodefine
class OMESeries(OME):
    series: tx.Optional[tx.List[str]] = None


@register_subclass(multiscales=tx.Any)
@autodefine
class OMEImage(OME):
    multiscales: tx.List[Multiscale]
    omero: tx.Optional[Omero]


@register_subclass(image_label=tx.Any)
@autodefine
class OMEImageLabel(OMEImage):
    image_labels: ImageLabel = field(alias="image-labels")


@register_subclass(labels=tx.Any)
@autodefine
class OMELabels(OME):
    labels: tx.List[str]


@register_subclass(plate=tx.Any)
@autodefine
class OMEPlate(OME):
    plate: Plate


@register_subclass(well=tx.Any)
@autodefine
class OMEWell(OME):
    well: Well


@register_subclass(scene=tx.Any)
@autodefine
class OMEScene(OME):
    scene: Scene


@register_subclass(bioformats2raw_layout=3)
@autodefine
class OMEBioformats2Raw(OME):
    bioformats2raw_layout: tx.Literal[3]
    plate: Plate
