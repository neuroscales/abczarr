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
from abczarr._core import typing as tz
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]
List = tz.BuiltinSequence  # list | tuple


class Space(OMESchemaItem):
    name: Required[str]
    path: Optional[str]


class CoordinateTransformationBase(OMESchemaItem):
    type: Required[str]
    output: Recommended[Space]  # Optional if wrapped. Else required.
    input: Recommended[Space]   # Optional if wrapped. Else required.
    name: Optional[str]


class Identity(CoordinateTransformationBase):
    type: Required[tx.Literal["identity"]]


class MapAxis(CoordinateTransformationBase):
    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[List[int]]


class Translation(CoordinateTransformationBase):
    type: Required[tx.Literal["translation"]]
    translation: Required[List[float]]


class Scale(CoordinateTransformationBase):
    type: Required[tx.Literal["scale"]]
    scale: Required[List[float]]


class Affine(CoordinateTransformationBase):
    type: Required[tx.Literal["affine"]]


class AffineMatrix(Affine):
    affine: Required[List[List[float]]]


class AffinePath(Affine):
    path: Required[str]


class Rotation(CoordinateTransformationBase):
    type: Required[tx.Literal["rotation"]]


class RotationMatrix(Rotation):
    rotation: Required[List[List[float]]]


class RotationPath(Rotation):
    path: Required[str]


class Sequence(CoordinateTransformationBase):
    type: Required[tx.Literal["sequence"]]
    transformations: Required[List["CoordinateTransformation"]]


class Displacements(CoordinateTransformationBase):
    type: Required[tx.Literal["displacements"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


class Coordinates(CoordinateTransformationBase):
    type: Required[tx.Literal["coordinates"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


class Bijection(CoordinateTransformationBase):
    type: Required[tx.Literal["bijection"]]
    forward: Required["CoordinateTransformation"]
    inverse: Required["CoordinateTransformation"]


class ByDimension(CoordinateTransformationBase):

    class Transformation(OMESchemaItem):
        inputAxes: Required[List[int]]
        outputAxes: Required[List[int]]
        transformation: Required["CoordinateTransformation"]

    type: Required[tx.Literal["byDimension"]]
    transformations: Required[List[Transformation]]


CoordinateTransformation = tx.Union[
    Identity,
    MapAxis,
    Translation,
    Scale,
    AffineMatrix, AffinePath,
    RotationMatrix, RotationPath,
    Sequence,
    Displacements,
    Coordinates,
    Bijection,
    ByDimension
]
