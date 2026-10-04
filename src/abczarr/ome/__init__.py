__all__ = [
    "base",
    "config",
    "pyramid",
    "schemas",
    "v0_1",
    "v0_2",
    "v0_3",
    "v0_4",
    "v0_5",
    "v0_6",
    "v0_6dev1",
    "v0_6dev2",
    "v0_6dev3",
    "v0_6dev4",
    "v0_6rc0",
]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

from . import base
from .base import *  # noqa: F403
from .base import __all__ as __all_base

# Every submodule other than `base` is imported on first access, through the
# module-level `__getattr__` below (PEP 562). Each version package costs tens
# of milliseconds to import, and most programs only ever touch one or two of
# them. Version detection (`base._version_package`) and conversion
# (`base._target_class`) import the package they need by name, and every
# version registers its classes on its own bases, so nothing needs all of them
# loaded up front.
#
# `config` and `pyramid` are lazy too: `config` imports the latest version
# package, and `pyramid` imports `abczarr.abc.sync`, which is itself still
# being imported when this package is first loaded through `abczarr.ome.node`.
_LAZY_SUBMODULES = frozenset(name for name in __all__ if name != "base")

# The names that `config` and `pyramid` export, re-exported here on first
# access. `tests/test_import_time.py` checks that these lists match the
# modules' own `__all__`.
_LAZY_ATTRIBUTES = {
    "ImageConfig": "config",
    "axis": "config",
    "downsample_array": "pyramid",
    "create_pyramid": "pyramid",
    "default_levels": "pyramid",
}

__all__ += __all_base
__all__ += list(_LAZY_ATTRIBUTES)

if tx.TYPE_CHECKING:
    from . import (  # noqa: F401
        config,
        pyramid,
        schemas,
        v0_1,
        v0_2,
        v0_3,
        v0_4,
        v0_5,
        v0_6,
        v0_6dev1,
        v0_6dev2,
        v0_6dev3,
        v0_6dev4,
        v0_6rc0,
    )
    from .config import ImageConfig, axis  # noqa: F401
    from .pyramid import (  # noqa: F401
        create_pyramid,
        default_levels,
        downsample_array,
    )


def __getattr__(name: str) -> tx.Any:
    if name in _LAZY_SUBMODULES:
        # Importing a submodule also binds it as an attribute of this
        # package, so later lookups never reach this function again.
        return importlib.import_module(f"{__name__}.{name}")
    if name in _LAZY_ATTRIBUTES:
        module = importlib.import_module(
            f"{__name__}.{_LAZY_ATTRIBUTES[name]}"
        )
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> tx.List[str]:
    return sorted(set(globals()) | set(__all__))
