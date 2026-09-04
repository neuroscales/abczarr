__all__ = [
    "Space",
    "CoordinateTransformation",
    "Identity",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Rotation",
    "Sequence",
    "Displacements",
    "Coordinates",
    "Bijection",
    "ByDimension",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Optional, Recommended, Required

# locals
from ..base import OMEMetadata

# typing
Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]


@autodefine
class Space(OMEMetadata):
    name: Optional[str]
    path: Optional[str]


@autodefine
class CoordinateTransformation(OMEMetadata):
    type: Required[str] = field(factory=False)
    output: Recommended[Space]
    input: Recommended[Space]
    name: Optional[str]


@register_subclass(type="identity")
@autodefine
class Identity(CoordinateTransformation):
    type: Required[tx.Literal["identity"]]


@register_subclass(type="mapAxis")
@autodefine
class MapAxis(CoordinateTransformation):
    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[tx.List[int]]


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
    affine: Optional[tx.List[tx.List[float]]]
    path: Optional[str]


@register_subclass(type="rotation")
@autodefine
class Rotation(CoordinateTransformation):
    type: Required[tx.Literal["rotation"]]
    rotation: Optional[tx.List[tx.List[float]]]
    path: Optional[str]


@register_subclass(type="sequence")
@autodefine
class Sequence(CoordinateTransformation):
    type: Required[tx.Literal["sequence"]]
    transformations: Required[tx.List[CoordinateTransformation]]


@register_subclass(type="displacements")
@autodefine
class Displacements(CoordinateTransformation):
    type: Required[tx.Literal["displacements"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


@register_subclass(type="coordinates")
@autodefine
class Coordinates(CoordinateTransformation):
    type: Required[tx.Literal["coordinates"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


@register_subclass(type="bijection")
@autodefine
class Bijection(CoordinateTransformation):
    type: Required[tx.Literal["bijection"]]
    forward: Required[CoordinateTransformation]
    inverse: Required[CoordinateTransformation]


@register_subclass(type="byDimension")
@autodefine
class ByDimension(CoordinateTransformation):

    @autodefine
    class Transformation(OMEMetadata):
        transformation: Optional[CoordinateTransformation]
        input_axes: Optional[tx.List[int]]
        output_axes: Optional[tx.List[int]]

    type: Required[tx.Literal["byDimension"]]
    transformations: Required[tx.List[Transformation]]
