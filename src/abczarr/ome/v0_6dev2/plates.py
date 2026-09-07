# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

__all__ = ["Plate"]
import re

import typing_extensions as tx

from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.auto.converters import ToNonNegative
from abczarr._core.rfc2119 import Optional, Recommended, Required

from ..base import OMEMetadata

NonNegativeInt = tx.Annotated[int, ToNonNegative()]
AlphaNumeric = tx.Annotated[str, re.compile("^[a-zA-Z0-9]+$")]
WellPath = tx.Annotated[str, re.compile("^[A-Za-z0-9]+/[A-Za-z0-9]+$")]


@autodefine
class Plate(OMEMetadata):
    """Describes a high-content screening plate.

    `rows` and `columns` name the plate's grid, such as `"A"`, `"B"`, ...
    and `"1"`, `"2"`, .... `wells` places each well in that grid and points,
    by `path`, at the group that holds the well's images. `acquisitions`
    lists the imaging runs the wells' images belong to, when the screen ran
    more than one acquisition.
    """

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
