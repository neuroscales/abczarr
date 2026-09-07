# Generated from v0_6dev1 by tools/gen_ome_metadata.py -- do not edit

"""A high-content screening plate: its rows, columns, wells, and
acquisitions.
"""

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

    `rows` and `columns` name the plate's grid, such as `"A"`, `"B"`, and
    so on for rows, and `"1"`, `"2"`, and so on for columns.

    Parameters
    ----------
    rows : list of Row
        The plate's rows, in grid order.
    columns : list of Column
        The plate's columns, in grid order.
    wells : list of Well
        Every well of the plate, each placed in the grid and pointing at
        the group holding its images.
    acquisitions : list of Acquisition
        The imaging runs the wells' images belong to, when the screen ran
        more than one. Optional.
    name : str
        A name for the plate. Recommended.
    field_count : int
        The largest number of fields of view acquired for any well of the
        plate. Recommended.
    """

    @autodefine
    class Acquisition(OMEMetadata):
        """One imaging run over some or all of the plate's wells.

        Parameters
        ----------
        id : int
            A non-negative integer identifying the acquisition. A
            [Well.Image][abczarr.ome.v0_6dev4.wells.Well.Image] refers to
            this run by this value.
        name : str
            A name for the acquisition. Recommended.
        maximumfieldcount : int
            The largest number of fields of view acquired for any well in
            this run. Recommended.
        description : str
            A description of the acquisition. Optional.
        starttime : int
            The time the acquisition started, in Unix epoch milliseconds.
            Optional.
        endtime : int
            The time the acquisition ended, in Unix epoch milliseconds.
            Optional.
        """

        id: Required[NonNegativeInt] = field(factory=False)
        name: Recommended[str]
        maximumfieldcount: Recommended[NonNegativeInt]
        description: Optional[str]
        starttime: Optional[int]
        endtime: Optional[int]

    @autodefine
    class Column(OMEMetadata):
        """One column of the plate's grid, named as it is labeled.

        Parameters
        ----------
        name : str
            The column's label, made of letters and digits only.
        """

        name: Required[AlphaNumeric] = field(factory=False)

    @autodefine
    class Row(OMEMetadata):
        """One row of the plate's grid, named as it is labeled.

        Parameters
        ----------
        name : str
            The row's label, made of letters and digits only.
        """

        name: Required[AlphaNumeric] = field(factory=False)

    @autodefine
    class Well(OMEMetadata):
        """One well's position in the plate, and the group holding it.

        Parameters
        ----------
        path : str
            The well's group, relative to the plate group, spelled as
            ``"<row>/<column>"``.
        rowIndex : int
            The well's row, as an index into `Plate.rows`.
        columnIndex : int
            The well's column, as an index into `Plate.columns`.
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
