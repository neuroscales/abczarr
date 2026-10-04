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
#
# An entry without a colon names a lazy submodule. A "module:attribute" entry
# names an attribute of that submodule, re-exported here on first access.
# `tests/test_import_time.py` checks that the attributes listed for `config`
# and `pyramid` match those modules' own `__all__`.
_LAZY = frozenset(
    {
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
        "config:ImageConfig",
        "config:axis",
        "pyramid:downsample_array",
        "pyramid:create_pyramid",
        "pyramid:default_levels",
    }
)

__all__ = ["base", *sorted(entry for entry in _LAZY if ":" not in entry)]
__all__ += __all_base
__all__ += sorted(entry.partition(":")[2] for entry in _LAZY if ":" in entry)

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
    if ":" not in name and name in _LAZY:
        # Importing a submodule also binds it as an attribute of this
        # package, so later lookups never reach this function again.
        return importlib.import_module(f"{__name__}.{name}")
    for entry in _LAZY:
        module, colon, attribute = entry.partition(":")
        if colon and attribute == name:
            value = getattr(
                importlib.import_module(f"{__name__}.{module}"), name
            )
            # Caching the attribute in the package namespace means later
            # lookups never reach this function again.
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> tx.List[str]:
    return sorted(set(globals()) | set(__all__))
