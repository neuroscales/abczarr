"""What a driver is, and how one is chosen for an array.

A :class:`Driver` is the object abczarr opens Zarr through. It declares what
it provides -- coarse capabilities (``"sharding"``, ``"async"``) and
fine-grained feature keys (``"v3:codec:zstd"``) -- through the same
:class:`~abczarr.abc.capabilities.Support` model the surface uses, so
choosing a driver for an array is a set difference: the features the array
:meth:`~abczarr.metadata.base.ArrayMetadata.required_features` says it needs,
minus the ones the driver provides. What is left is why the driver cannot
open it, named codec by codec -- an :class:`UnsupportedZarrOperation` that
points at the exact gap rather than a backend's opaque error.

A concrete driver declares its feature support in ``_CAPABILITIES`` or, for a
backend whose support depends on the install, overrides
:meth:`~abczarr.abc.capabilities.SupportsCapabilities.support` to probe it
lazily.
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

    ``bool(verdict)`` is ``True`` when nothing is missing; :attr:`missing`
    lists the feature keys the driver does not provide, and :attr:`reason`
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

    Declares what it provides via :attr:`_CAPABILITIES` (or a probing
    ``support``) and answers, for a given array, whether it can open it.
    """

    #: The driver's registered name (``"zarr-python"``, ...).
    name: tx.ClassVar[str] = ""

    def can_open(self, metadata: "ArrayMetadata") -> Verdict:
        """Whether this driver provides every feature *metadata* requires."""
        missing = [
            feature
            for feature in metadata.required_features()
            if not self.supports(feature)
        ]
        return Verdict(self.name or type(self).__name__, missing)


def select_driver(
    metadata: "ArrayMetadata", drivers: tx.Iterable[Driver]
) -> Driver:
    """Return the first driver that can open *metadata*.

    When none can, raise :class:`UnsupportedZarrOperation` whose message names
    each candidate and the features it is missing, so the failure points at
    the exact codec rather than a backend's opaque error.
    """
    verdicts = []  # type: tx.List[Verdict]
    for driver in drivers:
        verdict = driver.can_open(metadata)
        if verdict:
            return driver
        verdicts.append(verdict)
    detail = "; ".join(v.reason for v in verdicts) or "no drivers available"
    raise UnsupportedZarrOperation(f"open this array -- {detail}")
