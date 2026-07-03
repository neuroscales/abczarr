__all__ = ["Dataset", "Multiscale"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.attrs import autodefine, field
from abczarr._core.rfc2119 import Required, Recommended, Optional

# locals
from ..base import OMEMetadata
from .axes import Axis
from .transformations import CoordinateTransformation, Scale, Translation


@autodefine
class IntrinsicCoordinateTransformations(list):
    _scale: Required[Scale]
    _translation: Optional[Translation]

    @property
    def scale(self) -> Scale:
        return self[0]

    @scale.setter
    def scale(self, value: Scale):
        if not isinstance(value, Scale):
            raise TypeError(f"Expected a Scale instance, got {type(value).__name__}")
        if len(self) == 0:
            self.append(value)
        else:
            self[0] = value

    @property
    def translation(self) -> tx.Optional[Translation]:
        return self[1] if len(self) > 1 else None

    @translation.setter
    def translation(self, value: tx.Optional[Translation]):
        if value is not None and not isinstance(value, Translation):
            raise TypeError(f"Expected a Translation instance or None, got {type(value).__name__}")
        if value is None:
            if len(self) > 1:
                self.pop(1)
        else:
            if len(self) > 1:
                self[1] = value
            else:
                self.append(value)


@autodefine
class Dataset(OMEMetadata):
    path: Required[str] = field(factory=False)
    coordinateTransformations: Required[IntrinsicCoordinateTransformations]


@autodefine
class Multiscale(OMEMetadata):

    @autodefine
    class Metadata(OMEMetadata):
        method: Optional[str]
        version: Optional[str]
        args: Optional[tx.List[tz.JSON]]
        kwargs: Optional[tx.Dict[str, tz.JSON]]

    axes: Required[tx.List[Axis]]
    datasets: Required[tx.List[Dataset]]
    coordinateTransformations: Optional[tx.List[CoordinateTransformation]]
    name: Recommended[str]
    type: Recommended[str]
    metadata: Recommended[Metadata]
