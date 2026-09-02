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
    from abczarr.config import ZarrConfig
    from abczarr.metadata.base import ArrayMetadata, NodeMetadata


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

    def create(self, location: tx.Any, config: "ZarrConfig") -> "ZarrNode":
        """Create the node *config* describes at *location* and open it.

        The default lowers the config to metadata and creates from that. A
        backend overrides it to create through its own machinery from the
        config's coarse fields, so the backend writes its own metadata.
        """
        return self.create_from_metadata(
            location,
            self._config_metadata(config),
            overwrite=config.overwrite,
        )

    def create_from_metadata(
        self, location: tx.Any, metadata: "NodeMetadata",
        *, overwrite: bool = False,
    ) -> "ZarrNode":
        """Create a node from an exact *metadata* document and open it.

        The escape hatch for a setup the config helpers do not express: hand
        in an [ArrayMetadata][abczarr.metadata.base.ArrayMetadata] or
        [GroupMetadata][abczarr.metadata.base.GroupMetadata] and it is written
        and opened as it is. The default writes the metadata to the store and
        opens it; a driver may override to create through its backend.

        Raises
        ------
        FileExistsError
            When something already exists at *location* and *overwrite* is
            false.
        """
        from bagof.paths import Path

        from abczarr.metadata.base import node_at

        path = Path(str(location))
        if node_at(path) is not None:
            if not overwrite:
                raise FileExistsError(f"a node already exists at {location}")
            path.rmdir(recursive=True)
        path.mkdir(parents=True, exist_ok=True)
        metadata.to_file(path)
        return self.open(location, "r+")

    def create_group(
        self, location: tx.Any, *,
        config: "tx.Optional[ZarrConfig]" = None, **fields: tx.Any,
    ) -> "ZarrNode":
        """Create a new group at *location* and open it.

        Pass a [GroupConfig][abczarr.config.GroupConfig] as *config*, or its
        fields (`zarr_version`, `overwrite`, ...) as keyword arguments, which
        override the config.
        """
        from abczarr._core.attrs import evolve
        from abczarr.config import GroupConfig

        base = config if isinstance(config, GroupConfig) else GroupConfig(
            **dict(config or {})
        )
        resolved = evolve(base, **fields) if fields else base
        return self.create(location, resolved)

    @staticmethod
    def _config_metadata(config: "ZarrConfig") -> "NodeMetadata":
        """The metadata a config lowers to: an array's, or a group's."""
        from abczarr.config import ArrayConfig
        from abczarr.metadata.base import GroupMetadataV2, GroupMetadataV3

        if isinstance(config, ArrayConfig):
            return config.to_metadata()
        group_metadata = {2: GroupMetadataV2, 3: GroupMetadataV3}.get(
            config.zarr_version
        )
        if group_metadata is None:
            raise UnsupportedZarrOperation(
                f"create a group in Zarr v{config.zarr_version}"
            )
        return group_metadata(attributes=dict(config.attributes))
