__all__ = ["Plate"]

# dependencies
import typing_extensions as tx

# core
from abczarr._core.attrs import autodefine, field
from abczarr._core.attrs import NonNegativeConverter
from abczarr._core.metadata import FlexibleMetadata

# locals
from ..rfc2119 import Required, Recommended, Optional

# typing
NonNegativeInt = tx.Annotated[int, NonNegativeConverter()]
AlphaNumeric = tx.Annotated[str, tx.Pattern(r"^[a-zA-Z0-9]+$")]
WellPath = tx.Annotated[str, tx.Pattern(r"^[A-Z][0-9]/[A-Z][0-9]+$")]


@autodefine
class Plate(FlexibleMetadata):


    @autodefine
    class Acquisition(FlexibleMetadata):
        id: Required[NonNegativeInt] = field(factory=False)
        name: Recommended[str]
        maximumfieldcount: Recommended[NonNegativeInt]
        description: Optional[str]
        starttime: Optional[int]
        endtime: Optional[int]


    @autodefine
    class Column(FlexibleMetadata):
        name: Required[AlphaNumeric] = field(factory=False)


    @autodefine
    class Row(FlexibleMetadata):
        name: Required[AlphaNumeric] = field(factory=False)


    @autodefine
    class Well(FlexibleMetadata):
        path: Required[WellPath] = field(factory=False)
        rowIndex: Required[NonNegativeInt]
        columnIndex: Required[NonNegativeInt]


    acquisitions: Optional[tx.List[Acquisition]]
    columns: Required[tx.List[Column]]
    field_count: Recommended[NonNegativeInt]
    name: Recommended[str]
    rows: Required[tx.List[Row]]
    wells: Required[tx.List[Well]]
