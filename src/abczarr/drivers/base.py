"""What a driver is, and how one is chosen for an array.

A [Driver][abczarr.drivers.base.Driver] is the object abczarr opens
Zarr through -- zarr-python, tensorstore, or another backend. It
declares what it provides, both coarse capabilities (`"sharding"`,
`"async"`) and fine-grained feature keys (`"v3:codec:zstd"`),
through the same
[Support][abczarr.abc.capabilities.Support] model the rest of the
surface uses.

Choosing a driver for an array is then a simple comparison: the
features the array's metadata requires, against the ones each
candidate driver provides. Whatever is left over is why that driver
cannot open the array, named feature by feature --
[select_driver][abczarr.drivers.base.select_driver] raises an
[UnsupportedZarrOperation][abczarr.abc.errors.UnsupportedZarrOperation]
that points at the exact gap rather than a backend's opaque error.
"""

__all__ = [
    "Driver",
    "Verdict",
    "select_driver",
    "available_drivers",
    "register_driver",
]

# stdlib
import importlib

# dependencies
import typing_extensions as tx

# core
from abczarr.abc.capabilities import SupportsCapabilities
from abczarr.abc.errors import UnsupportedZarrOperation

if tx.TYPE_CHECKING:
    from abczarr.abc.node import ZarrNode
    from abczarr.metadata.base import ArrayMetadata


class Verdict:
    """Whether a driver can open an array, and what it lacks if not.

    `bool(verdict)` is `True` when nothing is missing; `missing`
    lists the feature keys the driver does not provide, and `reason`
    renders a one-line explanation.
    """

    def __init__(self, driver: str, missing: tx.Iterable[str]) -> None:
        self.driver = driver
        self.missing = tuple(sorted(missing))

    def __bool__(self) -> bool:
        return not self.missing

    @property
    def reason(self) -> str:
        if not self.missing:
            return f"{self.driver} can open it"
        return "{} lacks {}".format(self.driver, ", ".join(self.missing))

    def __repr__(self) -> str:
        return f"Verdict({self.driver!r}, ok={bool(self)})"


class Driver(SupportsCapabilities):
    """A backend abczarr opens Zarr through.

    A concrete driver declares what it provides -- capabilities and
    feature keys -- and answers, for a given array's metadata,
    whether it can open it.
    """

    #: The driver's registered name (`"zarr-python"`, ...).
    name: tx.ClassVar[str] = ""

    @property
    def available(self) -> bool:
        """Whether this driver's backend is installed and usable."""
        return True

    def can_open(self, metadata: "ArrayMetadata") -> Verdict:
        """Whether this driver provides every feature *metadata*
        requires."""
        missing = [
            feature
            for feature in metadata.required_features()
            if not self.supports(feature)
        ]
        return Verdict(self.name or type(self).__name__, missing)

    def open(self, location: tx.Any, mode: str = "r") -> "ZarrNode":
        """Open *location* and wrap it as a node.

        Raises
        ------
        [UnsupportedZarrOperation][abczarr.abc.errors.UnsupportedZarrOperation]
            When this driver cannot open a location.
        """
        raise UnsupportedZarrOperation("open", self.name or None)

    def create_group(
        self, location: tx.Any, *, zarr_version: int = 3,
        overwrite: bool = False,
    ) -> "ZarrNode":
        """Create a new group at *location* and wrap it.

        Raises
        ------
        [UnsupportedZarrOperation][abczarr.abc.errors.UnsupportedZarrOperation]
            When this driver cannot create a group.
        """
        raise UnsupportedZarrOperation("create_group", self.name or None)


#: The drivers abczarr knows about, as ``(name, module, class)``. They are
#: imported lazily so importing abczarr never imports a backend.
_KNOWN_DRIVERS = [
    ("zarr-python", "abczarr.drivers.zarr_python", "ZarrPythonDriver"),
]


def register_driver(module: str, cls: str, name: str = "") -> None:
    """Register a driver by the module and class that provide it.

    The driver is imported and instantiated only when
    [available_drivers][abczarr.drivers.base.available_drivers] is called.
    """
    _KNOWN_DRIVERS.append((name, module, cls))


def available_drivers() -> "tx.List[Driver]":
    """The installed, usable drivers, in registration order.

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
        When none can. The message names each candidate driver and
        the features it is missing, so the failure points at the
        exact gap rather than a backend's opaque error.
    """
    verdicts = []  # type: tx.List[Verdict]
    for driver in drivers:
        verdict = driver.can_open(metadata)
        if verdict:
            return driver
        verdicts.append(verdict)
    detail = "; ".join(v.reason for v in verdicts) or "no drivers available"
    raise UnsupportedZarrOperation(f"open this array -- {detail}")
