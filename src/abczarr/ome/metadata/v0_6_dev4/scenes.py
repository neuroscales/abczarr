__all__ = ["Scene"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine
from abczarr._core.metadata import FlexibleMetadata

# locals
from ..rfc2119 import Required
from .transformations import CoordinateTransformation


@autodefine
class Scene(FlexibleMetadata):
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]
