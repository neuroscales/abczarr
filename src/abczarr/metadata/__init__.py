__all__ = ["base", "v1", "v2", "v3"]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

from . import base

# The version packages `v1`, `v2` and `v3` are imported on first access,
# through the module-level `__getattr__` below (PEP 562). Together they cost
# about 0.2 s to import, mostly in building their codec and data type models.
# Their concrete classes register themselves on the classes in `base` when
# imported, so `base.NodeMetadata` imports all three before it dispatches a
# construction to a registered subclass. `tests/test_import_time.py` checks
# that `import abczarr` loads none of them.
_LAZY = frozenset({"v1", "v2", "v3"})

if tx.TYPE_CHECKING:
    from . import v1, v2, v3  # noqa: F401


def __getattr__(name: str) -> tx.Any:
    if name in _LAZY:
        # Importing a submodule also binds it as an attribute of this
        # package, so later lookups never reach this function again.
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> tx.List[str]:
    return sorted(set(globals()) | set(__all__))
