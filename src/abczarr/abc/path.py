"""Store paths, over the uniform :mod:`bagof.paths` surface.

A :class:`StorePath` is a normal :class:`bagof.paths.Path` -- one API over
local and cloud paths -- with one extra bit of store state, ``read_only``.
``bagof.paths`` already dispatches on the URL scheme (``s3://``, ``gs://``,
``memory://``, a local path, ...), so there is no per-protocol subclass to
declare here: ``StorePath("s3://bucket/key")`` picks the right driver on its
own, and the underlying driver object is always reachable through
``.wrapped``.

``read_only`` rides onto every derived path (``.parent``, ``/``,
``iterdir()``, ...) automatically: bagof.paths copies the wrapper's state
when it derives a new path, so a child of a read-only store is read-only
too.
"""

__all__ = [
    "StorePath",
    "AsyncStorePath",
]

# dependencies
import typing_extensions as tx
from bagof.paths import AsyncPath, Path


class StorePath(Path):
    """A zarr store location: a path, plus a ``read_only`` flag."""

    def __init__(
        self, *args: tx.Any, read_only: bool = False, **kwargs: tx.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.read_only = read_only


class AsyncStorePath(AsyncPath):
    """The async counterpart of :class:`StorePath`."""

    # Run StorePath's synchronous surface in the worker thread, so any
    # sync override here is honoured on the async side too.
    _sync_type: tx.ClassVar[tx.Type[Path]] = StorePath

    def __init__(
        self, *args: tx.Any, read_only: bool = False, **kwargs: tx.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.read_only = read_only
