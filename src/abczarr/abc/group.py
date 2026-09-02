"""The Zarr group interface: a container of arrays and subgroups."""

__all__ = [
    "ZarrGroup",
]

# stdlib
from abc import abstractmethod

# dependencies
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz

from .array import ZarrArray, ZarrArrayConfig

# locals
from .node import ZarrNode


class ZarrGroup(ZarrNode):
    """A Zarr group: a container of arrays and subgroups.

    Index it like a mapping to reach a member by name:

    !!! example
        ```python
        group["images"]
        group["images"] = other_array
        del group["images"]
        ```
    """

    @abstractmethod
    def __getitem__(self, key: str) -> ZarrNode:
        """Get the subgroup or array named *key*."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: ZarrNode) -> None:
        """Set the subgroup or array named *key*."""
        ...

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        """Delete the subgroup or array named *key*."""
        ...

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        """Create or open a subgroup named *name*.

        Parameters
        ----------
        name : str
            The subgroup's name.
        overwrite : bool, optional
            Replace an existing member named *name* instead of
            raising.
        """
        ...

    @abstractmethod
    def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        config: tx.Optional[ZarrArrayConfig] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrArray:
        """Create a new array named *name* within this group.

        Parameters
        ----------
        name : str
            The array's name.
        shape : tuple of int
            The array's shape.
        dtype : numpy dtype
            The array's data type.
        config : ZarrArrayConfig, optional
            Chunking, sharding, and compression options. May also be
            passed as individual keyword arguments -- see
            [ZarrArrayConfig][abczarr.abc.array.ZarrArrayConfig].
        """
        ...
