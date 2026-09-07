# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

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
import typing_extensions as tx

from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Optional, Required

from ..base import OMEMetadata

Interpolation = tx.Union[tx.Literal["nearest", "linear", "bspline-cubic"], str]


@autodefine
class CoordinateTransformation(OMEMetadata):
    """A `CoordinateTransformation` maps coordinates from one coordinate system
    to another.

    `type` identifies which kind of transformation it is. Constructing a
    `CoordinateTransformation` with a recognized `type` returns the matching
    subclass, such as [Scale][abczarr.ome.v0_6dev2.transformations.Scale].
    `input` and `output` identify the
    [CoordinateSystem][abczarr.ome.v0_6dev2.systems.CoordinateSystem]s the
    transformation maps between. `name` is an optional label for the
    transformation itself.
    """

    type: Required[str] = field(factory=False)
    input: Optional[tz.Json]
    output: Optional[tz.Json]
    name: Optional[str]


@register_subclass(type="identity")
@autodefine
class Identity(CoordinateTransformation):
    """An `Identity` transformation leaves coordinates unchanged.

    It states that `input` and `output` are the same coordinate system, or
    that no numeric adjustment is needed between them.
    """

    type: Required[tx.Literal["identity"]]


@register_subclass(type="mapAxis")
@autodefine
class MapAxis(CoordinateTransformation):
    """A `MapAxis` transformation renames axes without changing any coordinate
    values.

    `mapAxis` maps each output axis name to the input axis it takes its
    values from.
    """

    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[tx.Dict[str, str]]


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    """A `Translation` transformation adds a fixed offset to every coordinate,
    one value per axis.

    `translation` gives the offset inline, one number per axis. `path` reads
    the offset instead from an array, for a translation that varies from
    point to point rather than staying constant.
    """

    type: Required[tx.Literal["translation"]]
    translation: Optional[tx.List[float]]
    path: Optional[str]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    """A `Scale` transformation multiplies every coordinate by a per-axis
    factor.

    `scale` gives the factor inline, one number per axis. `path` reads the
    factor instead from an array, for a scale that varies from point to
    point rather than staying constant.
    """

    type: Required[tx.Literal["scale"]]
    scale: Optional[tx.List[float]]
    path: Optional[str]


@register_subclass(type="affine")
@autodefine
class Affine(CoordinateTransformation):
    """An `Affine` transformation applies a linear map and a translation
    together, given as a matrix.

    `affine` gives the matrix inline. `path` reads the matrix instead from
    an array.
    """

    type: Required[tx.Literal["affine"]]
    affine: Optional[tz.Json]
    path: Optional[str]


@register_subclass(type="rotation")
@autodefine
class Rotation(CoordinateTransformation):
    """A `Rotation` transformation rotates coordinates, given as a matrix.

    `rotation` gives the matrix inline. `path` reads the matrix instead from
    an array.
    """

    type: Required[tx.Literal["rotation"]]
    rotation: Optional[tz.Json]
    path: Optional[str]


@register_subclass(type="inverseOf")
@autodefine
class InverseOf(CoordinateTransformation):
    """An `InverseOf` transformation applies another transformation in reverse.

    `transformation` is the `CoordinateTransformation` to invert. This
    transformation's `input` and `output` are that transformation's `output`
    and `input`, swapped.
    """

    type: Required[tx.Literal["inverseOf"]]
    transformation: Required[CoordinateTransformation]


@register_subclass(type="bijection")
@autodefine
class Bijection(CoordinateTransformation):
    """A `Bijection` transformation is given as an explicit forward and inverse
    pair.

    `forward` maps `input` to `output`. `inverse` maps `output` back to
    `input`. This is used when a transformation's inverse cannot be derived
    automatically from its forward direction.
    """

    type: Required[tx.Literal["bijection"]]
    forward: Required[CoordinateTransformation]
    inverse: Required[CoordinateTransformation]


@register_subclass(type="sequence")
@autodefine
class Sequence(CoordinateTransformation):
    """A `Sequence` transformation composes several transformations into one,
    applied in order.

    `transformations` lists them from `input` to `output`. Each
    transformation's output feeds into the next transformation as its input.
    """

    type: Required[tx.Literal["sequence"]]
    transformations: Required[tx.List[CoordinateTransformation]]


@register_subclass(type="byDimension")
@autodefine
class ByDimension(CoordinateTransformation):
    """A `ByDimension` transformation combines several transformations, each
    acting on a different subset of axes.

    `transformations` lists them. Together they cover every axis between
    `input` and `output`.
    """

    type: Required[tx.Literal["byDimension"]]
    transformations: Required[tx.List[CoordinateTransformation]]


@register_subclass(type="displacements")
@autodefine
class Displacements(CoordinateTransformation):
    """A `Displacements` transformation is defined by a displacement field.

    `path` names an array that gives a displacement vector for each point.
    That vector is added to the input coordinate to produce the output
    coordinate. `interpolation` says how to sample the array between its own
    points.
    """

    type: Required[tx.Literal["displacements"]]
    path: Optional[str]
    interpolation: Optional[Interpolation]


@register_subclass(type="coordinates")
@autodefine
class Coordinates(CoordinateTransformation):
    """A `Coordinates` transformation is defined by an explicit coordinate
    lookup.

    `path` names an array that gives the output coordinate for each point
    directly. `interpolation` says how to sample the array between its own
    points.
    """

    type: Required[tx.Literal["coordinates"]]
    path: Optional[str]
    interpolation: Optional[Interpolation]
