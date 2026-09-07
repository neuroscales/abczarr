"""Coordinate transformations: how one coordinate system maps to another."""

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
    """Identifies a
    [CoordinateSystem][abczarr.ome.v0_6dev1.systems.CoordinateSystem] that
    a transformation maps to or from.

    Parameters
    ----------
    name : str
        Identifies the coordinate system directly. Optional.
    path : str
        Locates the coordinate system in another group, for one defined
        outside the transformation's own group. Optional.
    """

    name: Optional[str]
    path: Optional[str]


@autodefine
class CoordinateTransformation(OMEMetadata):
    """Maps coordinates from one coordinate system to another.

    Constructing a `CoordinateTransformation` with a recognized `type`
    returns the matching subclass, such as
    [Scale][abczarr.ome.v0_6dev1.transformations.Scale].

    Parameters
    ----------
    type : str
        Which kind of transformation this is.
    input : Space
        The [Space][abczarr.ome.v0_6dev1.transformations.Space]
        identifying the coordinate system the transformation maps from.
        Recommended.
    output : Space
        The [Space][abczarr.ome.v0_6dev1.transformations.Space]
        identifying the coordinate system the transformation maps to.
        Recommended.
    name : str
        A label for the transformation itself. Optional.
    """

    type: Required[str] = field(factory=False)
    output: Recommended[Space]
    input: Recommended[Space]
    name: Optional[str]


@register_subclass(type="identity")
@autodefine
class Identity(CoordinateTransformation):
    """Leaves coordinates unchanged.

    States that `input` and `output` are the same coordinate system, or
    that no numeric adjustment is needed between them.
    """

    type: Required[tx.Literal["identity"]]


@register_subclass(type="mapAxis")
@autodefine
class MapAxis(CoordinateTransformation):
    """Permutes axes without changing any coordinate values.

    Parameters
    ----------
    mapAxis : list of int
        One entry per output axis, giving the index of the input axis
        whose values that output axis carries.
    """

    type: Required[tx.Literal["mapAxis"]]
    mapAxis: Required[tx.List[int]]


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    """Adds a fixed offset to every coordinate, one value per axis.

    Parameters
    ----------
    translation : list of float
        The offset, given inline, one number per axis. Optional.
    path : str
        The path of an array to read the offset from instead, for a
        translation that varies from point to point rather than staying
        constant. Optional.
    """

    type: Required[tx.Literal["translation"]]
    translation: Optional[tx.List[float]]
    path: Optional[str]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    """Multiplies every coordinate by a per-axis factor.

    Parameters
    ----------
    scale : list of float
        The factor, given inline, one number per axis. Optional.
    path : str
        The path of an array to read the factor from instead, for a scale
        that varies from point to point rather than staying constant.
        Optional.
    """

    type: Required[tx.Literal["scale"]]
    scale: Optional[tx.List[float]]
    path: Optional[str]


@register_subclass(type="affine")
@autodefine
class Affine(CoordinateTransformation):
    """Applies a linear map and a translation, given as a matrix.

    Parameters
    ----------
    affine : list of list of float
        The matrix, given inline. Optional.
    path : str
        The path of an array to read the matrix from instead. Optional.
    """

    type: Required[tx.Literal["affine"]]
    affine: Optional[tx.List[tx.List[float]]]
    path: Optional[str]


@register_subclass(type="rotation")
@autodefine
class Rotation(CoordinateTransformation):
    """Rotates coordinates, given as a matrix.

    Parameters
    ----------
    rotation : list of list of float
        The matrix, given inline. Optional.
    path : str
        The path of an array to read the matrix from instead. Optional.
    """

    type: Required[tx.Literal["rotation"]]
    rotation: Optional[tx.List[tx.List[float]]]
    path: Optional[str]


@register_subclass(type="sequence")
@autodefine
class Sequence(CoordinateTransformation):
    """Composes several transformations into one, applied in order.

    Each transformation's output feeds into the next transformation as
    its input.

    Parameters
    ----------
    transformations : list of CoordinateTransformation
        The transformations to compose, from `input` to `output`.
    """

    type: Required[tx.Literal["sequence"]]
    transformations: Required[tx.List[CoordinateTransformation]]


@register_subclass(type="displacements")
@autodefine
class Displacements(CoordinateTransformation):
    """Defined by a displacement field.

    The vector read for each point is added to the input coordinate to
    produce the output coordinate.

    Parameters
    ----------
    path : str
        The path of an array that gives a displacement vector for each
        point.
    interpolation : str
        How to sample the array between its own points. Optional.
    """

    type: Required[tx.Literal["displacements"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


@register_subclass(type="coordinates")
@autodefine
class Coordinates(CoordinateTransformation):
    """Defined by an explicit coordinate lookup.

    Parameters
    ----------
    path : str
        The path of an array that gives the output coordinate for each
        point directly.
    interpolation : str
        How to sample the array between its own points. Optional.
    """

    type: Required[tx.Literal["coordinates"]]
    path: Required[str]
    interpolation: Optional[Interpolation]


@register_subclass(type="bijection")
@autodefine
class Bijection(CoordinateTransformation):
    """An explicit forward and inverse pair of transformations.

    Used when a transformation's inverse cannot be derived automatically
    from its forward direction.

    Parameters
    ----------
    forward : CoordinateTransformation
        The transformation from `input` to `output`.
    inverse : CoordinateTransformation
        The transformation from `output` back to `input`.
    """

    type: Required[tx.Literal["bijection"]]
    forward: Required[CoordinateTransformation]
    inverse: Required[CoordinateTransformation]


@register_subclass(type="byDimension")
@autodefine
class ByDimension(CoordinateTransformation):
    """Combines several transformations, each acting on its own subset of
    axes.

    Together, the entries in `transformations` cover every axis between
    `input` and `output`.

    Parameters
    ----------
    transformations : list of Transformation
        The transformations to combine.
    """

    @autodefine
    class Transformation(OMEMetadata):
        """One transformation of a
        [ByDimension][abczarr.ome.v0_6dev1.transformations.ByDimension],
        and the axes it applies to.

        Parameters
        ----------
        transformation : CoordinateTransformation
            The transformation to apply. Optional.
        input_axes : list of int
            The indices, into the enclosing `ByDimension`'s `input`
            coordinate system, of the axes `transformation` reads from.
            Optional.
        output_axes : list of int
            The indices, into the enclosing `ByDimension`'s `output`
            coordinate system, of the axes `transformation` writes to.
            Optional.
        """

        transformation: Optional[CoordinateTransformation]
        input_axes: Optional[tx.List[int]]
        output_axes: Optional[tx.List[int]]

    type: Required[tx.Literal["byDimension"]]
    transformations: Required[tx.List[Transformation]]
