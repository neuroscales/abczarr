"""abczarr: one interface for reading and writing Zarr, over any backend.

Open or create a node with [open][abczarr.api.open] and
[create][abczarr.api.create], and read or write it through the uniform
[ZarrArray][abczarr.abc.sync.ZarrArray] /
[ZarrGroup][abczarr.abc.sync.ZarrGroup] surface, whatever backend or
storage is behind it. The [config][abczarr.api.config] and
[registry][abczarr.api.registry] that creation rests on live under
[api][abczarr.api].
"""

__all__ = [
    "abc",
    "api",
    "drivers",
    "errors",
    "metadata",
    "ome",
    "schemas",
    "ZarrArray",
    "ZarrGroup",
    "ZarrNode",
    "AsyncZarrArray",
    "AsyncZarrGroup",
    "AsyncZarrNode",
    "UnsupportedZarrOperation",
    "UnsupportedConversion",
    "TransactionConflict",
    "open",
    "create",
    "create_array",
    "create_group",
    "open_array",
    "open_group",
]

from . import (
    abc,
    api,
    drivers,
    errors,
    metadata,
    ome,
    schemas,
)
from .abc import (
    AsyncZarrArray,
    AsyncZarrGroup,
    AsyncZarrNode,
    ZarrArray,
    ZarrGroup,
    ZarrNode,
)
from .api import (
    create,
    create_array,
    create_group,
    open,
    open_array,
    open_group,
)
from .errors import (
    TransactionConflict,
    UnsupportedConversion,
    UnsupportedZarrOperation,
)
