"""Foundation errors shared across the abczarr layers.

`UnsupportedConversion` lives here, in `_core`, rather than in
`abczarr.abc.errors`, so the metadata layer can import it at module top:
`abczarr.abc` sits above `abczarr.metadata` in the import graph, so a
metadata module reaching up into `abc.errors` would be a cycle, while
everything can depend on `_core`. `abczarr.abc.errors` re-exports it, so
its public path is unchanged.
"""

__all__ = [
    "UnsupportedConversion",
]

from abczarr._core import typing as tz


class UnsupportedConversion(ValueError):
    """A field has no representation in the target Zarr version.

    Raised by `to_version` when it is asked to convert under the
    ``"strict"`` policy and a field cannot be carried over. The
    message names the field and the version it could not be
    represented in.
    """

    def __init__(self, field: str, version: tz.ZarrVersion) -> None:
        super().__init__(
            f"cannot represent {field!r} in Zarr v{version}"
        )
        self.field = field
        self.version = version
