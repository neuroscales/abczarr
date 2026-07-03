__all__ = ["OMEMetadata", "OME"]

from abczarr._core.metadata import FlexibleMetadata
from abczarr._core.attrs import autodefine, field


@autodefine
class OMEMetadata(FlexibleMetadata):
    ...


@autodefine
class OME(OMEMetadata):
    version: str = field(factory=False)
