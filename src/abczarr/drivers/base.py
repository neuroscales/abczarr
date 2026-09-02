"""What a driver is.

A [Driver][abczarr.drivers.base.Driver] is the object abczarr opens
Zarr through -- zarr-python, tensorstore, or another backend. It
declares what it provides, both coarse capabilities (`"sharding"`,
`"async"`) and fine-grained feature keys (`"v3:codec:zstd"`),
through the same
[Support][abczarr.abc.capabilities.Support] model the rest of the
surface uses, and answers whether it can open a given array.

Which drivers exist and how one is chosen for an array live in
[abczarr.registry][abczarr.registry].
"""

__all__ = [
    "Driver",
    "Verdict",
]

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
