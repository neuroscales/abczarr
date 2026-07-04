"""ZarrIO module for handling Zarr data structures."""
__all__ = [
    "abc",
    "api",
    "config",
    "drivers",
    "metadata",
    "ome",
    "registry",
    "schemas",
    "ZarrArray",
    "ZarrGroup",
    "ZarrNode",
    "from_config",
    "open_array",
    "open_group",
]

import warnings

from . import abc
from . import api
from . import config
from . import drivers
from . import metadata
from . import ome
from . import registry
from . import schemas

from .abc import ZarrArray, ZarrGroup, ZarrNode
from .api import from_config, open_array, open_group

if False:
    try:
        import zarr as _  # noqa: F401

        from .drivers.zarr_python import ZarrPythonArray, ZarrPythonGroup

        __all__ += ["ZarrPythonArray", "ZarrPythonGroup"]

    except ImportError:
        warnings.warn(
            "zarr-python is not installed, driver disabled", stacklevel=2
        )

    try:
        import tensorstore as _  # noqa: F401

        from .drivers.tensorstore import ZarrTSArray, ZarrTSGroup

        __all__ += ["ZarrTSArray", "ZarrTSGroup"]

    except ImportError:
        warnings.warn(
            "Tensorstore is not installed, driver disabled", stacklevel=2
        )
