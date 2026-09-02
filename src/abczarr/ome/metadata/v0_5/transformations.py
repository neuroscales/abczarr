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

    Build [Scale][abczarr.ome.metadata.v0_5.transformations.Scale] or
    [Translation][abczarr.ome.metadata.v0_5.transformations.Translation]
    directly rather than this base class. Constructing with
    `type="scale"` or `type="translation"` returns the matching one.
    """

    type: Required[str] = field(factory=False)


@register_subclass(type="translation")
@autodefine
class Translation(CoordinateTransformation):
    """An offset, one value per axis, in the axes' physical units."""

    type: Required[tx.Literal["translation"]]
    translation: Required[tx.List[float]]


@register_subclass(type="scale")
@autodefine
class Scale(CoordinateTransformation):
    """A per-axis scale factor from array indices to physical units.

    For a resolution level, this is the physical size of one array
    element along each axis. It's what turns a pixel index into a
    micrometer, and what makes coarser levels of a pyramid line up
    with the finest one.
    """

    type: Required[tx.Literal["scale"]]
    scale: Required[tx.List[float]]
