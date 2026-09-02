"""The base of every Zarr object.

[ZarrNode][abczarr.abc.node.ZarrNode] is the common ancestor of
[ZarrArray][abczarr.abc.array.ZarrArray] and
[ZarrGroup][abczarr.abc.group.ZarrGroup]: metadata, attributes, the
Zarr format version, and the capability query all live here.
"""

__all__ = [
    "ZarrNode",
    "KNOWN_CAPABILITIES",
    "Support",
]

# stdlib
import os
from abc import ABC, abstractmethod

# dependencies
import typing_extensions as tx
from bagof.paths import Path

# core
from abczarr._core import typing as tz
from abczarr.metadata.base import NodeMetadata

# locals -- KNOWN_CAPABILITIES and Support are re-exported for callers that
# reach them through this module (they are listed in __all__).
from .capabilities import (  # noqa: F401
    KNOWN_CAPABILITIES,
    Support,
    SupportsCapabilities,
)


class ZarrNode(SupportsCapabilities, ABC):
    """Base class for any Zarr object: a group or an array.

    Use
    [support][abczarr.abc.capabilities.SupportsCapabilities.support]
    or
    [supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
    to check what a node's backend can do.
    """

    def __init__(self, store_path: tz.PathLike) -> None:
        if isinstance(store_path, (str, bytes)):
            store_path = Path(store_path)
        self._store_path = store_path
        # The raw backend object (a zarr.Array, a tensorstore.TensorStore,
        # ...). A concrete driver sets it; it stays None where a node has no
        # single backing object (e.g. a group that is only a path).
        self._native: tx.Any = None

    @property
    def store_path(self) -> os.PathLike:
        """The path to this node's location in its store."""
        return self._store_path

    @property
    def native(self) -> tx.Any:
        """The underlying backend object, or `None`.

        The escape hatch: anything the uniform surface does not name
        is still reachable through the backend object itself -- a
        `zarr.Array`, a `tensorstore.TensorStore`, and so on.
        """
        return self._native

    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        """This node's Zarr metadata."""
        ...

    @property
    @abstractmethod
    def attrs(self) -> tz.Attributes:
        """This node's user attributes."""
        ...

    @property
    @abstractmethod
    def zarr_version(self) -> tz.ZarrVersion:
        """The Zarr format version this node was written with."""
        ...
