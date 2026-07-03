__all__ = ["Scene"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine

# locals
from ..rfc2119 import Required
from ..base import OMEMetadata
from .transformations import CoordinateTransformation


@autodefine
class Scene(OMEMetadata):
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]
