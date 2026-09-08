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
    """Describes a high-content screening plate.

    `rows` and `columns` name the plate's grid, such as `"A"`, `"B"`, and
    so on for rows, and `"1"`, `"2"`, and so on for columns.
    """

    @autodefine
    class Acquisition(OMEMetadata):
        """One imaging run over some or all of the plate's wells."""

        id: Required[NonNegativeInt] = field(factory=False)
        """A non-negative integer identifying the acquisition. A
        [Well.Image][abczarr.ome.v0_6dev1.wells.Well.Image] refers to
        this run by this value."""
        name: Recommended[str]
        """A name for the acquisition. Recommended."""
        maximumfieldcount: Recommended[NonNegativeInt]
        """The largest number of fields of view acquired for any well
        in this run. Recommended."""
        description: Optional[str]
        """A description of the acquisition. Optional."""
        starttime: Optional[int]
        """The time the acquisition started, in Unix epoch
        milliseconds. Optional."""
        endtime: Optional[int]
        """The time the acquisition ended, in Unix epoch milliseconds.
        Optional."""


    @autodefine
    class Column(OMEMetadata):
        """One column of the plate's grid, named as it is labeled."""

        name: Required[AlphaNumeric] = field(factory=False)
        """The column's label, made of letters and digits only."""


    @autodefine
    class Row(OMEMetadata):
        """One row of the plate's grid, named as it is labeled."""

        name: Required[AlphaNumeric] = field(factory=False)
        """The row's label, made of letters and digits only."""


    @autodefine
    class Well(OMEMetadata):
        """One well's position in the plate, and the group holding it."""

        path: Required[WellPath] = field(factory=False)
        """The well's group, relative to the plate group, spelled as
        ``"<row>/<column>"``."""
        rowIndex: Required[NonNegativeInt]
        """The well's row, as an index into `Plate.rows`."""
        columnIndex: Required[NonNegativeInt]
        """The well's column, as an index into `Plate.columns`."""


    acquisitions: Optional[tx.List[Acquisition]]
    """The imaging runs the wells' images belong to, when the screen
    ran more than one. Optional."""
    columns: Required[tx.List[Column]]
    """The plate's columns, in grid order."""
    field_count: Recommended[NonNegativeInt]
    """The largest number of fields of view acquired for any well of
    the plate. Recommended."""
    name: Recommended[str]
    """A name for the plate. Recommended."""
    rows: Required[tx.List[Row]]
    """The plate's rows, in grid order."""
    wells: Required[tx.List[Well]]
    """Every well of the plate, each placed in the grid and pointing at
    the group holding its images."""
