"""A high-content screening plate: its rows, columns, wells, and
acquisitions.
"""

__all__ = ["Plate"]

# stdlib
import re

# dependencies
import typing_extensions as tx

# core
from abczarr._core.auto.attrs import autodefine, field
from abczarr._core.auto.converters import ToNonNegative
from abczarr._core.rfc2119 import Optional, Recommended, Required

# locals
from ..base import OMEMetadata

# typing
NonNegativeInt = tx.Annotated[int, ToNonNegative()]
AlphaNumeric = tx.Annotated[str, re.compile(r"^[a-zA-Z0-9]+$")]
WellPath = tx.Annotated[str, re.compile(r"^[A-Za-z0-9]+/[A-Za-z0-9]+$")]


@autodefine
class Plate(OMEMetadata):
    """A high-content screening plate.

    `rows` and `columns` name the plate's grid (`"A"`, `"B"`, ... and
    `"1"`, `"2"`, ...); `wells` places each well in it and points, by
    `path`, at the group holding that well's images. `acquisitions`
    lists the imaging runs the wells' images belong to, when the
    screen ran more than one.
    """


    @autodefine
    class Acquisition(OMEMetadata):
        """One imaging run over some or all of the plate's wells."""

        id: Required[NonNegativeInt] = field(factory=False)
        name: Recommended[str]
        maximumfieldcount: Recommended[NonNegativeInt]
        description: Optional[str]
        starttime: Optional[int]
        endtime: Optional[int]


    @autodefine
    class Column(OMEMetadata):
        """One column of the plate's grid, named as it is labeled."""

        name: Required[AlphaNumeric] = field(factory=False)


    @autodefine
    class Row(OMEMetadata):
        """One row of the plate's grid, named as it is labeled."""

        name: Required[AlphaNumeric] = field(factory=False)


    @autodefine
    class Well(OMEMetadata):
        """One well's position in the plate, and the group holding it.

        `path` is the well's group, relative to the plate group, as
        `"<row>/<column>"`; `rowIndex` and `columnIndex` are its
        position as indices into `Plate.rows` and `Plate.columns`.
        """

        path: Required[WellPath] = field(factory=False)
        rowIndex: Required[NonNegativeInt]
        columnIndex: Required[NonNegativeInt]


    acquisitions: Optional[tx.List[Acquisition]]
    columns: Required[tx.List[Column]]
    field_count: Recommended[NonNegativeInt]
    name: Recommended[str]
    rows: Required[tx.List[Row]]
    wells: Required[tx.List[Well]]
