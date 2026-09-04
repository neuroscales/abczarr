# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["Scene"]
import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine
from abczarr._core.rfc2119 import Required

from ..base import OMEMetadata
from .transformations import CoordinateTransformation


@autodefine
class Scene(OMEMetadata):
    coordinateTransformations: Required[tx.List[CoordinateTransformation]]
