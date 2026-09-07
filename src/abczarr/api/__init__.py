"""The user-facing API for opening and creating a Zarr node.

[open][abczarr.api.open] and its array and group variants read an
existing node. [create][abczarr.api.create],
[create_array][abczarr.api.create_array] and
[create_group][abczarr.api.create_group] make a new one, from a
config or from a metadata object.
[ArrayConfig][abczarr.api.ArrayConfig] and
[GroupConfig][abczarr.api.GroupConfig] describe what to create.
[select_driver][abczarr.api.select_driver],
[available_drivers][abczarr.api.available_drivers] and
[register_driver][abczarr.api.register_driver] choose the backend
that does the work.

Importing this package never imports a backend. The errors abczarr
raises live in [abczarr.errors][abczarr.errors] and are re-exported at
the package top level.
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
]

if tx.TYPE_CHECKING:
    # For type checkers and the API-reference builder only: at runtime the
    # reader/writer and the registry are resolved lazily by __getattr__ below,
    # so importing this package never pulls in a backend.
    from .config import (  # noqa: F401
        ArrayConfig,
        ArrayOptions,
        GroupConfig,
        GroupOptions,
        ZarrConfig,
        ZarrOptions,
    )
    from .entrypoint import (  # noqa: F401
        create,
        create_array,
        create_group,
        open,
        open_array,
        open_group,
    )
    from .registry import (  # noqa: F401
        available_drivers,
        register_driver,
        select_driver,
    )

#: Which module each public name is re-exported from. The reader/writer
#: module (`entrypoint`) and the `registry` module both import
#: `drivers.base`, and `drivers.base` imports the config layer from this
#: package. Resolving them lazily keeps that import from cycling back
#: through here.
_MODULES = {
    "abczarr.api.entrypoint": {
        "open", "open_array", "open_group",
        "create", "create_array", "create_group",
    },
    "abczarr.api.config": {
        "ZarrConfig", "GroupConfig", "ArrayConfig",
        "ZarrOptions", "GroupOptions", "ArrayOptions",
    },
    "abczarr.api.registry": {
        "register_driver", "available_drivers", "select_driver",
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
