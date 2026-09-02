"""Store paths, over the uniform `bagof.paths` surface.

[StorePath][abczarr.abc.path.StorePath] is a normal
`bagof.paths.Path` -- one API over local and cloud paths -- with one
extra bit of state, `read_only`. `bagof.paths` already dispatches on
the URL scheme (`s3://`, `gs://`, `memory://`, a local path, ...),
so `StorePath("s3://bucket/key")` picks the right backend on its
own, no per-protocol subclass needed, and the underlying backend
object is always reachable through `.wrapped`.

`read_only` rides along onto every path derived from it (`.parent`,
`/`, `iterdir()`, ...), so a child of a read-only store is read-only
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
    """A Zarr store location: a path, plus a `read_only` flag.

    Accepts everything `bagof.paths.Path` accepts -- a local path or
    a URL for any scheme it supports -- plus a `read_only` keyword.
    """

    def __init__(
        self, *args: tx.Any, read_only: bool = False, **kwargs: tx.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.read_only = read_only


class AsyncStorePath(AsyncPath):
    """The async counterpart of
    [StorePath][abczarr.abc.path.StorePath]."""

    # Run StorePath's synchronous surface in the worker thread, so any
    # sync override here is honoured on the async side too.
    _sync_type: tx.ClassVar[tx.Type[Path]] = StorePath

    def __init__(
        self, *args: tx.Any, read_only: bool = False, **kwargs: tx.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.read_only = read_only
