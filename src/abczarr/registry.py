
# stdlib
import warnings
from types import ModuleType

# dependencies
import typing_extensions as tx

from abczarr._core.imports import import_symbol

# locals
from ._core import typing as tz
from .abc import ZarrNode

_NODE2CLS = {
    "node": "ZarrNode",
    "array": "ZarrArray",
    "group": "ZarrGroup"
}


class UnsupportedDriverError(ValueError):
    """Exception raised when an unsupported driver is specified."""

    def __init__(self, driver: tz.AnyDriver) -> None:
        super().__init__(f"Unsupported driver: {driver}")
        self.driver = driver


class UnavailableDriverError(ValueError):
    """Exception raised when a driver is not available."""

    def __init__(self, driver: tz.AnyDriver) -> None:
        super().__init__(f"Driver not available: {driver}")
        self.driver = driver


class UnavailableDriverWarning(Warning):
    """Warning raised when a driver is not available."""

    def __init__(self, driver: tz.AnyDriver) -> None:
        super().__init__(f"Driver not available: {driver}")
        self.driver = driver


# name -> (probe_module, driver_module)
# where *_path is "pkg.module:ClassName"
_DRIVERS: tx.Dict[str, str] = {}
_LOADED_DRIVERS: tx.Dict[str, ModuleType] = {}


def register_driver(name: str, module: str) -> None:
    _DRIVERS[name] = module
    _LOADED_DRIVERS.pop(name, None)


register_driver("zarr-python", "abczarr.drivers.zarr_python")
register_driver("tensorstore", "abczarr.drivers.tensorstore")


def reload_driver(name: str, strict: bool = True) -> ModuleType:
    try:
        mod = import_symbol(_DRIVERS[name])
        _LOADED_DRIVERS[name] = mod
        return mod
    except UnavailableDriverError:
        _LOADED_DRIVERS[name] = False
        if strict:
            raise
        else:
            warnings.warn(name, UnavailableDriverWarning, stacklevel=2)
            return False


def load_driver(name: str, strict: bool = True) -> ModuleType:
    mod = _LOADED_DRIVERS.get(name)
    if mod is False:
        if strict:
            raise UnavailableDriverError(name)
        else:
            return False
    if not mod:
        mod = reload_driver(name, strict)
    return mod


def get_driver(
    driver: tx.Optional[tx.Union[str, ModuleType]] = None,
    node_type: tx.Optional[tx.Literal["node", "array", "group"]] = None,
    strict: tx.Optional[bool] = None
) -> tx.Union[ModuleType, tx.Type[ZarrNode]]:
    # Find first available driver
    if driver is None:
        for try_driver in _DRIVERS:
            driver = load_driver(
                try_driver,
                strict=False if strict is None else strict
            )
            if driver:
                break
    if not driver:
        raise UnavailableDriverError("No available drivers found.")

    # Load driver
    if isinstance(driver, str):
        driver = load_driver(
            driver,
            strict=True if strict is None else strict
        )
    if node_type is None:
        return driver

    # Get node class from driver
    node_cls = _NODE2CLS.get(node_type)
    node_cls: tx.Type[ZarrNode] = getattr(driver, node_cls, None)
    if node_cls is None:
        raise UnsupportedDriverError(driver)
    return node_cls
