__all__ = ["Plate"]

# stdlib
import re

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.validators import IsNonNegative
from abczarr._core.rfc2119 import RequirementForTypedDict

# locals
from ..base import OMESchemaItem, ome_schema_opt

# typing
Required = RequirementForTypedDict.Required
Recommended = RequirementForTypedDict.Recommended
Optional = RequirementForTypedDict.Optional
NonNegativeInt = tx.Annotated[int, IsNonNegative()]
AlphaNumeric = tx.Annotated[str, re.compile(r"^[a-zA-Z0-9]+$")]
WellPath = tx.Annotated[str, re.compile(r"^[A-Z][0-9]/[A-Z][0-9]+$")]
List = tz.BuiltinSequence  # list | tuple


class Plate(OMESchemaItem, **ome_schema_opt):

    class Acquisition(OMESchemaItem, **ome_schema_opt):
        id: Required[NonNegativeInt]
        name: Recommended[str]
        maximumfieldcount: Recommended[NonNegativeInt]
        description: Optional[str]
        starttime: Optional[int]
        endtime: Optional[int]

    class Column(OMESchemaItem, **ome_schema_opt):
        name: Required[AlphaNumeric]

    class Row(OMESchemaItem, **ome_schema_opt):
        name: Required[AlphaNumeric]

    class Well(OMESchemaItem, **ome_schema_opt):
        path: Required[WellPath]
        rowIndex: Required[NonNegativeInt]
        columnIndex: Required[NonNegativeInt]

    acquisitions: Optional[List[Acquisition]]
    columns: Required[List[Column]]
    field_count: Recommended[NonNegativeInt]
    name: Recommended[str]
    rows: Required[List[Row]]
    wells: Required[List[Well]]
