"""One interface for reading and writing Zarr, over any backend.

[open][abczarr.open] and [create][abczarr.create] open or create a
node. The node is then read or written through the uniform
[ZarrArray][abczarr.abc.sync.ZarrArray] and
[ZarrGroup][abczarr.abc.sync.ZarrGroup] surface, regardless of the
backend or the storage behind it.

The whole user-facing API is re-exported at this top level. This
includes the [ArrayConfig][abczarr.ArrayConfig] and
[GroupConfig][abczarr.GroupConfig] classes that creation rests on, the
[select_driver][abczarr.select_driver] registry that picks a backend,
and the errors abczarr raises. The open and create functions, the
config classes, and the registry are also available under
[api][abczarr.api]. The errors keep their own home in
[abczarr.errors][abczarr.errors].
"""

__all__ = [
    # subpackages
    "abc",
    "api",
    "drivers",
    "errors",
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
from .errors import (
    SchemaValidationError,
    TransactionConflict,
    UnsupportedConversion,
    UnsupportedZarrOperation,
)
