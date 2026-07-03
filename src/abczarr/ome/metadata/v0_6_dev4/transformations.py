__all__ = [
    "Space",
    "CoordinateTransformation",
    "Identity",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine", "AffineMatrix", "AffinePath",
    "Rotation", "RotationMatrix", "RotationPath",
    "Sequence",
    "Displacements",
    "Coordinates",
    "Bijection",
    "ByDimension"
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.metadata import FlexibleMetadata, register_subclass

# locals
from ..rfc2119 import Required, Optional

# typing
Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]


@autodefine
class Space(FlexibleMetadata):
    name: Required[str]
    path: Optional[str]


@autodefine
class CoordinateTransformation(FlexibleMetadata):
    type: Required[str] = field(factory=False)
    output: Required[Space]
    input: Required[Space]
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
    translation: Required[tx.List[float]]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    type: Required[tx.Literal["scale"]]
    scale: Required[tx.List[float]]


@register_subclass(type="affine")
@autodefine
class Affine(CoordinateTransformation):
    type: Required[tx.Literal["affine"]]


@register_subclass(type="affine", affine=tx.Any)
@autodefine
class AffineMatrix(Affine):
    affine: Required[tx.List[tx.List[float]]]


@register_subclass(type="affine", path=tx.Any)
@autodefine
class AffinePath(Affine):
    path: Required[str]


@register_subclass(type="rotation")
@autodefine
class Rotation(CoordinateTransformation):
    type: Required[tx.Literal["rotation"]]


@register_subclass(type="rotation", affine=tx.Any)
@autodefine
class RotationMatrix(Rotation):
    rotation: Required[tx.List[tx.List[float]]]


@register_subclass(type="rotation", path=tx.Any)
@autodefine
class RotationPath(Rotation):
    path: Required[str]


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
    type: Required[tx.Literal["byDimension"]]
    transformations: Required[tx.List[CoordinateTransformation]]
