"""The user-facing API: open and create a Zarr node, and the pieces they
rest on.

- [open][abczarr.api.open] and its array/group variants read a node.
- [create][abczarr.api.create], [create_array][abczarr.api.create_array] and
  [create_group][abczarr.api.create_group] make one, from a config or a
  metadata object.
- [ArrayConfig][abczarr.api.ArrayConfig] and
  [GroupConfig][abczarr.api.GroupConfig] describe what to create.
- [select_driver][abczarr.api.select_driver],
  [available_drivers][abczarr.api.available_drivers] and
  [register_driver][abczarr.api.register_driver] choose the backend.
- the errors abczarr raises are re-exported here too.

The config and error names come from lightweight modules and are safe to
reach eagerly; the reader/writer and the driver registry are resolved on
first use, so importing this package never imports a backend and never
forms a cycle with the drivers that import the config layer.
"""

import importlib

import typing_extensions as tx

__all__ = [
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

if tx.TYPE_CHECKING:
    # For type checkers and the API-reference builder only: at runtime the
    # reader/writer and the registry are resolved lazily by __getattr__ below,
    # so importing this package never pulls in a backend.
    from ._config import (  # noqa: F401
        ArrayConfig,
        ArrayOptions,
        GroupConfig,
        GroupOptions,
        ZarrConfig,
        ZarrOptions,
    )
    from ._entry import (  # noqa: F401
        create,
        create_array,
        create_group,
        open,
        open_array,
        open_group,
    )
    from ._errors import (  # noqa: F401
        SchemaValidationError,
        TransactionConflict,
        UnsupportedConversion,
        UnsupportedZarrOperation,
    )
    from ._registry import (  # noqa: F401
        available_drivers,
        register_driver,
        select_driver,
    )

#: Which module each public name lives in, by full import path. The
#: reader/writer (`_entry`) and the registry (`_registry`) both import
#: `drivers.base`, and `drivers.base` imports the config layer from this
#: package -- so they are resolved lazily to keep that import from cycling
#: back through here. The errors come from the package-level leaf module
#: `abczarr._errors`, re-exported here for convenience.
_MODULES = {
    "abczarr.api._entry": {
        "open", "open_array", "open_group",
        "create", "create_array", "create_group",
    },
    "abczarr.api._config": {
        "ZarrConfig", "GroupConfig", "ArrayConfig",
        "ZarrOptions", "GroupOptions", "ArrayOptions",
    },
    "abczarr.api._registry": {
        "register_driver", "available_drivers", "select_driver",
    },
    "abczarr._errors": {
        "UnsupportedZarrOperation", "UnsupportedConversion",
        "TransactionConflict", "SchemaValidationError",
    },
}


def __getattr__(name: str) -> tx.Any:
    # Lazy so that importing abczarr.api never pulls in the reader/writer or a
    # driver, and so a driver that imports the config layer does not cycle.
    for module, names in _MODULES.items():
        if name in names:
            return getattr(importlib.import_module(module), name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> tx.List[str]:
    return sorted(__all__)
