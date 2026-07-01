__all__ = [
    "ZarrNode",
]

# stdlib
import os
from abc import ABC, abstractmethod

# core
from abczarr._core import typing as tz
from abczarr._core.path import Path
from abczarr.metadata.base import NodeMetadata


class ZarrNode(ABC):
    """Base class for any Zarr-like object (group or array)."""

    def __init__(self, store_path: tz.PathLike) -> None:
        if isinstance(store_path, (str, bytes)):
            store_path = Path(store_path)
        self._store_path = store_path

    @property
    def store_path(self) -> os.PathLike:
        """Path to the Zarr store for this node."""
        return self._store_path

    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        """Access metadata for this node."""
        ...

    @property
    @abstractmethod
    def attrs(self) -> tz.Attributes:
        """Access attributes for this node."""
        ...

    @property
    @abstractmethod
    def zarr_version(self) -> tz.ZarrVersion:
        """Get the Zarr format version."""
        ...
