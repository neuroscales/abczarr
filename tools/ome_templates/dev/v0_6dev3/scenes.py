"""A scene: coordinate transformations attached to a group directly."""

__all__ = ["Scene"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine
from abczarr._core.rfc2119 import Required

# locals
from ..base import OMEMetadata
from .transformations import CoordinateTransformation


@autodefine
class Scene(OMEMetadata):
    """A set of coordinate transformations attached to a group, independent
    of any single image or pyramid.

    Parameters
    ----------
    coordinateTransformations : list of CoordinateTransformation
        The transformations the scene carries.
    """

    coordinateTransformations: Required[tx.List[CoordinateTransformation]]
