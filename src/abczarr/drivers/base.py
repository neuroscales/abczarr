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
]

# dependencies
import typing_extensions as tx

# core
from abczarr.abc.capabilities import SupportsCapabilities
from abczarr.abc.errors import UnsupportedZarrOperation

if tx.TYPE_CHECKING:
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

    def can_open(self, metadata: "ArrayMetadata") -> Verdict:
        """Whether this driver provides every feature *metadata*
        requires."""
        missing = [
            feature
            for feature in metadata.required_features()
            if not self.supports(feature)
        ]
        return Verdict(self.name or type(self).__name__, missing)


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
