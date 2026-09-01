__all__ = [
    "ZarrNode",
    "KNOWN_CAPABILITIES",
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

#: The capability names :meth:`ZarrNode.supports` understands. A driver
#: advertises the subset it provides; asking about any other name simply
#: returns ``False``, so a caller written against a newer vocabulary never
#: crashes an older driver.
KNOWN_CAPABILITIES = frozenset(
    {
        "sharding",             # zarr v3 sharded chunk grids
        "async",                # a native coroutine I/O surface
        "consolidated_metadata",
        "partial_read",         # read a sub-region without the whole chunk
        "partial_write",
        "codecs_v2",
        "codecs_v3",
    }
)


class ZarrNode(ABC):
    """Base class for any Zarr-like object (group or array)."""

    #: Capabilities this driver provides, drawn from
    #: :data:`KNOWN_CAPABILITIES`. Overridden per driver; empty here.
    _CAPABILITIES: tx.ClassVar[tx.FrozenSet[str]] = frozenset()

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
        """Path to the Zarr store for this node."""
        return self._store_path

    @property
    def native(self) -> tx.Any:
        """The underlying backend object, or ``None``.

        The escape hatch: anything the uniform surface does not name is still
        reachable through the backend object itself -- a ``zarr.Array``, a
        ``tensorstore.TensorStore``, and so on. This is a supported contract,
        not an accident of attribute delegation.
        """
        return self._native

    @classmethod
    def supports(cls, capability: str) -> bool:
        """Whether this driver provides *capability*.

        Answered from the class, without opening or touching a live store, so
        a caller can branch on a backend's strengths before committing to an
        operation. ``capability`` is one of :data:`KNOWN_CAPABILITIES`; any
        other name returns ``False``.
        """
        return capability in cls._CAPABILITIES

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
