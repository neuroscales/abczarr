__all__ = [
    "CoordinateTransformation",
    "Translation",
    "Scale",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Required

# locals
from ..base import OMEMetadata


@autodefine
class CoordinateTransformation(OMEMetadata):
    type: Required[str] = field(factory=False)


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    type: Required[tx.Literal["translation"]]
    translation: Required[tx.List[float]]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    type: Required[tx.Literal["scale"]]
    scale: Required[tx.List[float]]
