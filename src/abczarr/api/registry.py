"""The registry of backend drivers, and choosing one for an array.

abczarr opens Zarr through a [Driver][abczarr.drivers.base.Driver]. This
module holds the list of drivers it knows about and, given an array's
metadata, picks the one that can read it -- the machinery behind
[open][abczarr.api.open].
"""

__all__ = [
    "register_driver",
    "available_drivers",
    "select_driver",
]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

# locals
from ..abc.errors import UnsupportedZarrOperation
from ..drivers.base import Driver

if tx.TYPE_CHECKING:
    from ..metadata.base import ArrayMetadata

#: The drivers abczarr knows about, as ``(name, module, class)``. They are
#: imported lazily, so importing abczarr never imports a backend. The order
#: is the preference order when several drivers can open an array.
_KNOWN_DRIVERS = [
    ("zarr-python", "abczarr.drivers.zarr_python", "ZarrPythonDriver"),
    ("tensorstore", "abczarr.drivers.tensorstore", "TensorStoreDriver"),
    ("zarrista", "abczarr.drivers.zarrista", "ZarristaDriver"),
]


def register_driver(module: str, cls: str, name: str = "") -> None:
    """Register a driver by the module and class that provide it.

    The driver is imported and instantiated only when
    [available_drivers][abczarr.api.registry.available_drivers] is called, so
    registering one never imports its backend.
    """
    _KNOWN_DRIVERS.append((name, module, cls))


def available_drivers() -> "tx.List[Driver]":
    """The installed, usable drivers, in preference order.

    Each known driver is imported and instantiated; one whose backend is not
    installed (its `available` is `False`) or that cannot be imported is left
    out.
    """
    drivers = []  # type: tx.List[Driver]
    for _name, module_path, cls_name in _KNOWN_DRIVERS:
        try:
            module = importlib.import_module(module_path)
            driver = getattr(module, cls_name)()
        except Exception:
            continue
        if driver.available:
            drivers.append(driver)
    return drivers


def select_driver(
    metadata: "ArrayMetadata", drivers: tx.Iterable[Driver]
) -> Driver:
    """Return the first driver in *drivers* that can open *metadata*.

    Raises
    ------
    [UnsupportedZarrOperation][abczarr.abc.errors.UnsupportedZarrOperation]
        When none can. The message names each candidate driver and the
        features it is missing, so the failure points at the exact gap
        rather than a backend's opaque error.
    """
    verdicts = []
    for driver in drivers:
        verdict = driver.can_open(metadata)
        if verdict:
            return driver
        verdicts.append(verdict)
    detail = "; ".join(v.reason for v in verdicts) or "no drivers available"
    raise UnsupportedZarrOperation(f"open this array -- {detail}")
