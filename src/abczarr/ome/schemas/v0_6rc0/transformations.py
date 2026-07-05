__all__ = [
    "Space",
    "CoordinateTransformation",
    "Identity",
    "MapAxis",
    "ProjectAxis",
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
from ..base import OMESchemaItem, ome_schema_opt

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]
List = tz.BuiltinSequence  # list | tuple


class Space(OMESchemaItem, **ome_schema_opt):
    name: Optional[str]
    path: Optional[str]


class CoordinateTransformationBase(OMESchemaItem, **ome_schema_opt):
    type: Required[str]
    output: Recommended[Space]  # Optional if wrapped. Else required.
    input: Recommended[Space]   # Optional if wrapped. Else required.
    name: Optional[str]


class Identity(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["identity"]]


class MapAxis(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[List[int]]


class ProjectAxis(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["projectAxis"]]
    createdOutputs: Optional[List[int]]
    droppedInputs: Optional[List[int]]


class Translation(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["translation"]]
    translation: Required[List[float]]


class Scale(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["scale"]]
    scale: Required[List[float]]


class Affine(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["affine"]]


class AffineMatrix(Affine, **ome_schema_opt):
    affine: Required[List[List[float]]]


class AffinePath(Affine, **ome_schema_opt):
    path: Required[str]


class Rotation(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["rotation"]]


class RotationMatrix(Rotation, **ome_schema_opt):
    rotation: Required[List[List[float]]]


class RotationPath(Rotation, **ome_schema_opt):
    path: Required[str]


class Sequence(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["sequence"]]
    transformations: Required[List["CoordinateTransformation"]]


class Displacements(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["displacements"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


class Coordinates(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["coordinates"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


class Bijection(CoordinateTransformationBase, **ome_schema_opt):
    type: Required[tx.Literal["bijection"]]
    forward: Required["CoordinateTransformation"]
    inverse: Required["CoordinateTransformation"]


class ByDimension(CoordinateTransformationBase, **ome_schema_opt):

    class Transformation(OMESchemaItem, **ome_schema_opt):
        inputAxes: Required[List[int]]
        outputAxes: Required[List[int]]
        transformation: Required["CoordinateTransformation"]

    type: Required[tx.Literal["byDimension"]]
    transformations: Required[List[Transformation]]


CoordinateTransformation = tx.Union[
    Identity,
    MapAxis,
    ProjectAxis,
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
