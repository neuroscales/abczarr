__all__ = [
    "CoordinateTransformation",
    "Identity",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Rotation",
    "InverseOf",
    "Bijection",
    "Sequence",
    "ByDimension",
    "Displacements",
    "Coordinates",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Optional, Required

# locals
from ..base import OMEMetadata

# typing
Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]


@autodefine
class CoordinateTransformation(OMEMetadata):
    type: Required[str] = field(factory=False)
    input: Optional[tz.Json]
    output: Optional[tz.Json]
    name: Optional[str]


@register_subclass(type="identity")
@autodefine
class Identity(CoordinateTransformation):
    type: Required[tx.Literal["identity"]]


@register_subclass(type="mapAxis")
@autodefine
class MapAxis(CoordinateTransformation):
    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[tx.Dict[str, str]]


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    type: Required[tx.Literal["translation"]]
    translation: Optional[tx.List[float]]
    path: Optional[str]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    type: Required[tx.Literal["scale"]]
    scale: Optional[tx.List[float]]
    path: Optional[str]


@register_subclass(type="affine")
@autodefine
class Affine(CoordinateTransformation):
    type: Required[tx.Literal["affine"]]
    affine: Optional[tz.Json]
    path: Optional[str]


@register_subclass(type="rotation")
@autodefine
class Rotation(CoordinateTransformation):
    type: Required[tx.Literal["rotation"]]
    rotation: Optional[tz.Json]
    path: Optional[str]


@register_subclass(type="inverseOf")
@autodefine
class InverseOf(CoordinateTransformation):
    type: Required[tx.Literal["inverseOf"]]
    transformation: Required[CoordinateTransformation]


@register_subclass(type="bijection")
@autodefine
class Bijection(CoordinateTransformation):
    type: Required[tx.Literal["bijection"]]
    forward: Required[CoordinateTransformation]
    inverse: Required[CoordinateTransformation]


@register_subclass(type="sequence")
@autodefine
class Sequence(CoordinateTransformation):
    type: Required[tx.Literal["sequence"]]
    transformations: Required[tx.List[CoordinateTransformation]]


@register_subclass(type="byDimension")
@autodefine
class ByDimension(CoordinateTransformation):
    type: Required[tx.Literal["byDimension"]]
    transformations: Required[tx.List[CoordinateTransformation]]


@register_subclass(type="displacements")
@autodefine
class Displacements(CoordinateTransformation):
    type: Required[tx.Literal["displacements"]]
    path: Optional[str]
    interpolation: Optional[Interpolation]


@register_subclass(type="coordinates")
@autodefine
class Coordinates(CoordinateTransformation):
    type: Required[tx.Literal["coordinates"]]
    path: Optional[str]
    interpolation: Optional[Interpolation]
