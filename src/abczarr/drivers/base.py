"""What a driver is.

A [Driver][abczarr.drivers.base.Driver] is the object abczarr opens
Zarr through -- zarr-python, tensorstore, or another backend. It
declares what it provides, both coarse capabilities (`"sharding"`,
`"async"`) and fine-grained feature keys (`"v3:codec:zstd"`),
through the same
[Support][abczarr.abc.capabilities.Support] model the rest of the
surface uses, and answers whether it can open a given array.

Which drivers exist and how one is chosen for an array live in
[abczarr.api.registry][abczarr.api.registry].
"""

__all__ = [
    "Driver",
    "Verdict",
]

# dependencies
import typing_extensions as tx
from bagof.paths import Path

# core
from abczarr._core.asyncutils import run_sync
from abczarr._core.attrs import evolve
from abczarr.abc.capabilities import SupportsCapabilities
from abczarr.api.config import ArrayConfig, GroupConfig
from abczarr.errors import UnsupportedZarrOperation
from abczarr.metadata.base import GroupMetadataV2, GroupMetadataV3, _node_at

if tx.TYPE_CHECKING:
    from abczarr.abc.async_node import AsyncZarrNode
    from abczarr.abc.node import ZarrNode
    from abczarr.api.config import ZarrConfig
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

    def open(
        self, location: tx.Any, mode: str = "r", *, asynchronous: bool = False,
    ) -> "tx.Union[ZarrNode, tx.Awaitable[AsyncZarrNode]]":
        """Open *location* and wrap it as a node.

        !!! note
            With `asynchronous=True` the return value is a **coroutine you
            await**: the metadata read is awaited, so the open does its I/O
            asynchronously and resolves to the coroutine twin of the node --
            an [AsyncZarrArray][abczarr.abc.async_array.AsyncZarrArray] or
            [AsyncZarrGroup][abczarr.abc.async_group.AsyncZarrGroup]. Whether
            that surface is native to the backend or synthesized in a thread
            pool depends on the driver. Without the flag, the node is opened
            synchronously and returned directly.

        Parameters
        ----------
        location : Any
            The location to open.
        mode : str
            The access mode.
        asynchronous : bool, optional
            When true, return a coroutine resolving to the async twin.

        Returns
        -------
        ZarrNode or Awaitable[AsyncZarrNode]
            The node, or -- when *asynchronous* is true -- a coroutine
            resolving to its async twin.

        Raises
        ------
        [UnsupportedZarrOperation][abczarr.errors.UnsupportedZarrOperation]
            When this driver cannot open a location.
        """
        if asynchronous:
            return self._open_async(location, mode)
        return self._open_sync(location, mode)

    def _open_sync(self, location: tx.Any, mode: str) -> "ZarrNode":
        """Open *location* synchronously. A driver overrides this."""
        raise UnsupportedZarrOperation("open", self.name or None)

    async def _open_async(
        self, location: tx.Any, mode: str
    ) -> "AsyncZarrNode":
        """Open *location* asynchronously.

        A backend with a native coroutine open overrides this to await its own
        I/O. The default runs the synchronous open in a worker thread and
        returns its async twin, so a driver with no native async open still
        presents the coroutine surface.
        """
        node = await run_sync(self._open_sync, location, mode)
        return node.as_async()

    def create(
        self, location: tx.Any, config: "ZarrConfig",
        *, asynchronous: bool = False,
    ) -> "tx.Union[ZarrNode, tx.Awaitable[AsyncZarrNode]]":
        """Create the node *config* describes at *location* and open it.

        The default lowers the config to metadata and creates from that. A
        backend overrides the create it runs to build through its own
        machinery from the config's coarse fields, so the backend writes its
        own metadata.

        !!! note
            With `asynchronous=True` the return value is a coroutine you await,
            resolving to the async twin of the node, mirroring async
            [open][abczarr.drivers.base.Driver.open].

        Parameters
        ----------
        location : Any
            Where to create the node.
        config : ZarrConfig
            The array or group to create.
        asynchronous : bool, optional
            When true, return a coroutine resolving to the async twin.

        Returns
        -------
        ZarrNode or Awaitable[AsyncZarrNode]
        """
        if asynchronous:
            return self._create_async(location, config)
        return self._create_sync(location, config)

    def _create_sync(
        self, location: tx.Any, config: "ZarrConfig"
    ) -> "ZarrNode":
        """Create *config* synchronously. A backend overrides this to create
        natively from the config's coarse fields."""
        return self._create_from_metadata_sync(
            location,
            self._config_metadata(config),
            overwrite=config.overwrite,
        )

    async def _create_async(
        self, location: tx.Any, config: "ZarrConfig"
    ) -> "AsyncZarrNode":
        """Create *config* asynchronously. The default thread-bridges the
        synchronous create; a backend with a native coroutine create
        overrides this."""
        node = await run_sync(self._create_sync, location, config)
        return node.as_async()

    def create_from_metadata(
        self, location: tx.Any, metadata: "NodeMetadata",
        *, overwrite: bool = False, asynchronous: bool = False,
    ) -> "tx.Union[ZarrNode, tx.Awaitable[AsyncZarrNode]]":
        """Create a node from an exact *metadata* document and open it.

        The escape hatch for a setup the config helpers do not express: hand
        in an [ArrayMetadata][abczarr.metadata.base.ArrayMetadata] or
        [GroupMetadata][abczarr.metadata.base.GroupMetadata] and it is written
        and opened as it is. The default writes the metadata to the store and
        opens it; a driver may override the create it runs to build through
        its backend.

        !!! note
            With `asynchronous=True` the return value is a coroutine you await,
            resolving to the async twin of the node.

        Parameters
        ----------
        location : Any
            Where to create the node.
        metadata : NodeMetadata
            The exact metadata document to write.
        overwrite : bool, optional
            Replace whatever is already at *location*.
        asynchronous : bool, optional
            When true, return a coroutine resolving to the async twin.

        Returns
        -------
        ZarrNode or Awaitable[AsyncZarrNode]

        Raises
        ------
        FileExistsError
            When something already exists at *location* and *overwrite* is
            false.
        """
        if asynchronous:
            return self._create_from_metadata_async(
                location, metadata, overwrite=overwrite
            )
        return self._create_from_metadata_sync(
            location, metadata, overwrite=overwrite
        )

    def _create_from_metadata_sync(
        self, location: tx.Any, metadata: "NodeMetadata",
        *, overwrite: bool = False,
    ) -> "ZarrNode":
        """Write *metadata* to the store and open it. A backend overrides this
        to create through its own machinery."""
        path = Path(str(location))
        if _node_at(path) is not None:
            if not overwrite:
                raise FileExistsError(f"a node already exists at {location}")
            path.rmdir(recursive=True)
        path.mkdir(parents=True, exist_ok=True)
        metadata.to_file(path)
        return self._open_sync(location, "r+")

    async def _create_from_metadata_async(
        self, location: tx.Any, metadata: "NodeMetadata",
        *, overwrite: bool = False,
    ) -> "AsyncZarrNode":
        """Create from *metadata* asynchronously. The default thread-bridges
        the synchronous create; a backend with a native coroutine create
        overrides this."""
        node = await run_sync(
            self._create_from_metadata_sync, location, metadata,
            overwrite=overwrite,
        )
        return node.as_async()

    def create_group(
        self, location: tx.Any, *,
        config: "tx.Optional[ZarrConfig]" = None, **fields: tx.Any,
    ) -> "ZarrNode":
        """Create a new group at *location* and open it.

        Pass a [GroupConfig][abczarr.api.config.GroupConfig] as *config*, or
        its fields (`zarr_version`, `overwrite`, ...) as keyword arguments,
        which
        override the config.
        """
        base = config if isinstance(config, GroupConfig) else GroupConfig(
            **dict(config or {})
        )
        resolved = evolve(base, **fields) if fields else base
        return tx.cast("ZarrNode", self.create(location, resolved))

    @staticmethod
    def _config_metadata(config: "ZarrConfig") -> "NodeMetadata":
        """The metadata a config lowers to: an array's, or a group's."""
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
