"""Coordinate transformations: how a resolution level maps to
physical space.
"""

__all__ = [
    "CoordinateTransformation",
    "Translation",
    "Scale",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.metadata import register_subclass
from abczarr._core.rfc2119 import Required

# locals
from ..base import OMEMetadata


@autodefine
class CoordinateTransformation(OMEMetadata):
    """A transformation from array indices to physical coordinates.

    Build [Scale][abczarr.ome.v0_1.transformations.Scale] or
    [Translation][abczarr.ome.v0_1.transformations.Translation] directly
    rather than this base class. Constructing a `CoordinateTransformation`
    with `type="scale"` or `type="translation"` returns the matching one.

    Parameters
    ----------
    type : str
        Which kind of transformation this is: `"scale"` or
        `"translation"`.
    """

    type: Required[str] = field(factory=False)


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    """An offset, one value per axis, in the axes' physical units.

    Parameters
    ----------
    translation : list of float
        The offset, one number per axis.
    """

    type: Required[tx.Literal["translation"]]
    translation: Required[tx.List[float]]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    """A per-axis scale factor from array indices to physical units.

    For a resolution level, the scale factor is the physical size of one
    array element along each axis. It turns a pixel index into a physical
    unit such as a micrometer, and it makes coarser levels of a pyramid
    line up with the finest one.

    Parameters
    ----------
    scale : list of float
        The scale factor, one number per axis.
    """

    type: Required[tx.Literal["scale"]]
    scale: Required[tx.List[float]]
