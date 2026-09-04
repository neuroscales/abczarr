"""The user-facing API: open and create a Zarr node, and the pieces they
rest on.

- [open][abczarr.api.open] and its array/group variants read a node.
- [create][abczarr.api.create], [create_array][abczarr.api.create_array] and
  [create_group][abczarr.api.create_group] make one, from a
  [config][abczarr.api.config] or a metadata object.
- [config][abczarr.api.config] describes what to create.
- [registry][abczarr.api.registry] chooses a driver.

Everything is loaded on first use, so importing the package never imports
a backend and the config layer stays free of the reader and writer.
"""

import importlib

import typing_extensions as tx

__all__ = [
    "config",
    "registry",
    "open",
    "open_array",
    "open_group",
    "create",
    "create_array",
    "create_group",
]

if tx.TYPE_CHECKING:
    # For type checkers and the API-reference builder only: at runtime these
    # are resolved lazily by __getattr__ below, so importing this package
    # never pulls in the reader/writer.
    from . import config, registry  # noqa: F401
    from ._entry import (  # noqa: F401
        create,
        create_array,
        create_group,
        open,
        open_array,
        open_group,
    )

#: The names that live in the private entry module.
_ENTRY = {
    "open", "open_array", "open_group",
    "create", "create_array", "create_group",
}


def __getattr__(name: str) -> tx.Any:
    # Lazy so that importing abczarr.api (or a submodule of it, such as
    # abczarr.api.config) never pulls in the reader/writer or a driver.
    if name in ("config", "registry"):
        return importlib.import_module(f"{__name__}.{name}")
    if name in _ENTRY:
        entry = importlib.import_module(f"{__name__}._entry")
        return getattr(entry, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def __dir__() -> tx.List[str]:
    return sorted(__all__)
