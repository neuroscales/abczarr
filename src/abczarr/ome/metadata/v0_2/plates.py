__all__ = ["Plate"]

# stdlib
import re

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.auto.converters import NonNegativeConverter
from abczarr._core.rfc2119 import Required, Recommended, Optional

# locals
from ..base import OMEMetadata
from .version import Version

# typing
NonNegativeInt = tx.Annotated[int, NonNegativeConverter()]
AlphaNumeric = tx.Annotated[str, re.compile(r"^[a-zA-Z0-9]+$")]
WellPath = tx.Annotated[str, re.compile(r"^[A-Z][0-9]/[A-Z][0-9]+$")]


@autodefine
class Plate(OMEMetadata):


    @autodefine
    class Acquisition(OMEMetadata):
        id: Required[NonNegativeInt] = field(factory=False)
        name: Recommended[str]
        maximumfieldcount: Recommended[NonNegativeInt]
        description: Optional[str]
        starttime: Optional[int]
        endtime: Optional[int]


    @autodefine
    class Column(OMEMetadata):
        name: Required[AlphaNumeric] = field(factory=False)


    @autodefine
    class Row(OMEMetadata):
        name: Required[AlphaNumeric] = field(factory=False)


    @autodefine
    class Well(OMEMetadata):
        path: Required[WellPath] = field(factory=False)
        rowIndex: Required[NonNegativeInt]
        columnIndex: Required[NonNegativeInt]


    acquisitions: Optional[tx.List[Acquisition]]
    columns: Required[tx.List[Column]]
    field_count: Recommended[NonNegativeInt]
    name: Recommended[str]
    rows: Required[tx.List[Row]]
    wells: Required[tx.List[Well]]
    version: Recommended[Version]
