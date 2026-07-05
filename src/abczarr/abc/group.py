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
    """Abstract interface for a Zarr group (container of arrays/subgroups)."""

    @abstractmethod
    def __getitem__(self, key: str) -> ZarrNode:
        """Get a subgroup or array by name within this group."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: ZarrNode) -> None:
        """Set a subgroup or array by name within this group."""
        ...

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        """Delete a subgroup or array by name within this group."""
        ...

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        """Create or open a subgroup within this group."""
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
        """Create a new array within this group."""
        ...
