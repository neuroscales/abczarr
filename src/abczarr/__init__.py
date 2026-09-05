"""abczarr: one interface for reading and writing Zarr, over any backend.

Open or create a node with [open][abczarr.open] and [create][abczarr.create],
and read or write it through the uniform
[ZarrArray][abczarr.abc.sync.ZarrArray] /
[ZarrGroup][abczarr.abc.sync.ZarrGroup] surface, whatever backend or storage is
behind it. The whole user-facing API is
re-exported here at the top level: the [ArrayConfig][abczarr.ArrayConfig] and
[GroupConfig][abczarr.GroupConfig] that creation rests on, the
[select_driver][abczarr.select_driver] registry that picks a backend, and the
errors abczarr raises. The same names are also available under
[api][abczarr.api].
"""

__all__ = [
    # subpackages
    "abc",
    "api",
    "drivers",
    "metadata",
    "ome",
    "schemas",
    # node surface
    "ZarrArray",
    "ZarrGroup",
    "ZarrNode",
    "AsyncZarrArray",
    "AsyncZarrGroup",
    "AsyncZarrNode",
    # open / create
    "open",
    "open_array",
    "open_group",
    "create",
    "create_array",
    "create_group",
    # config
    "ZarrConfig",
    "GroupConfig",
    "ArrayConfig",
    "ZarrOptions",
    "GroupOptions",
    "ArrayOptions",
    # driver registry
    "register_driver",
    "available_drivers",
    "select_driver",
    # errors
    "UnsupportedZarrOperation",
    "UnsupportedConversion",
    "TransactionConflict",
    "SchemaValidationError",
]

# Subpackages first: importing `drivers` fully loads `drivers.base` (which
# imports the config layer) before the re-exports below reach for the
# reader/writer or the registry, so those lazy imports never cycle back.
from . import (
    abc,
    api,
    drivers,
    metadata,
    ome,
    schemas,
)
from ._errors import (
    SchemaValidationError,
    TransactionConflict,
    UnsupportedConversion,
    UnsupportedZarrOperation,
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
    ArrayConfig,
    ArrayOptions,
    GroupConfig,
    GroupOptions,
    ZarrConfig,
    ZarrOptions,
    available_drivers,
    create,
    create_array,
    create_group,
    open,
    open_array,
    open_group,
    register_driver,
    select_driver,
)
