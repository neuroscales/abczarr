"""Open Zarr arrays and groups through a selected backend driver.

[open][abczarr.api.open] opens a location and returns the array or group
there, wrapped in the uniform surface. When no driver is named it reads the
array's metadata, picks a driver that provides every codec the array needs
(see [select_driver][abczarr.drivers.base.select_driver]), and opens through
it; [open_array][abczarr.api.open_array] and
[open_group][abczarr.api.open_group] additionally check what they opened.
"""

__all__ = [
    "open",
    "open_array",
    "open_group",
    "create",
    "create_group",
]

# stdlib
import json

# dependencies
import typing_extensions as tx

# locals
from ._core import typing as tz
from ._core.attrs import evolve
from .abc.array import ZarrArray
from .abc.errors import UnsupportedZarrOperation
from .abc.group import ZarrGroup
from .abc.node import ZarrNode
from .abc.store import PathStore
from .config import ArrayConfig, GroupConfig, ZarrConfig
from .drivers._metadata import metadata_from_dict
from .drivers.base import Driver
from .metadata.base import ArrayMetadata, NodeMetadata
from .registry import available_drivers, select_driver

_DriverArg = tx.Optional[tx.Union[str, Driver]]


def open(
    path: tz.PathLike, mode: str = "a", *, driver: _DriverArg = None
) -> ZarrNode:
    """Open the Zarr array or group at *path*.

    Parameters
    ----------
    path : PathLike
        A local path or a URL (`"s3://bucket/dataset.zarr"`).
    mode : str
        The access mode, as zarr-python understands it (`"r"`, `"r+"`,
        `"a"`, `"w"`).
    driver : str or Driver, optional
        A driver, or its name, to open with. When omitted, a driver is
        chosen for what the array needs.

    Returns
    -------
    ZarrNode
        The wrapped array or group.
    """
    chosen = _choose(path, _resolve_drivers(driver))
    return chosen.open(path, mode)


def open_array(
    path: tz.PathLike, mode: str = "a", *, driver: _DriverArg = None
) -> ZarrArray:
    """Open *path*, requiring it to be an array.

    Like [open][abczarr.api.open], but raises if *path* is a group.
    """
    node = open(path, mode, driver=driver)
    if not isinstance(node, ZarrArray):
        raise UnsupportedZarrOperation("open_array on a group")
    return node


def open_group(
    path: tz.PathLike, mode: str = "a", *, driver: _DriverArg = None
) -> ZarrGroup:
    """Open *path*, requiring it to be a group.

    Like [open][abczarr.api.open], but raises if *path* is an array.
    """
    node = open(path, mode, driver=driver)
    if not isinstance(node, ZarrGroup):
        raise UnsupportedZarrOperation("open_group on an array")
    return node


@tx.overload
def create(
    location: tz.PathLike, spec: ArrayConfig, **fields: tx.Any
) -> ZarrArray: ...
@tx.overload
def create(
    location: tz.PathLike, spec: GroupConfig, **fields: tx.Any
) -> ZarrGroup: ...
@tx.overload
def create(
    location: tz.PathLike, spec: NodeMetadata, **fields: tx.Any
) -> ZarrNode: ...


def create(
    location: tz.PathLike,
    spec: "tx.Union[ZarrConfig, NodeMetadata, tz.JSONDict]",
    **fields: tx.Any,
) -> ZarrNode:
    """Create the array or group *spec* describes at *location*.

    *spec* is usually a config: an [ArrayConfig][abczarr.config.ArrayConfig]
    creates an array, a [GroupConfig][abczarr.config.GroupConfig] a group, and
    keyword arguments override the config's fields. For full control beyond
    what the config helpers express, *spec* may instead be an exact metadata
    document -- an [ArrayMetadata][abczarr.metadata.base.ArrayMetadata] or
    [GroupMetadata][abczarr.metadata.base.GroupMetadata], or its dict -- which
    is created as it is; then `overwrite` and `driver` may be passed as
    keywords.
    """
    if isinstance(spec, ZarrConfig):
        config = evolve(spec, **fields) if fields else spec
        if isinstance(config, ArrayConfig):
            config = config.resolve()
            metadata = config.to_metadata()
        else:
            metadata = None
        return _choose_create_driver(config.driver, metadata).create(
            location, config
        )

    metadata = spec if isinstance(spec, NodeMetadata) else metadata_from_dict(
        spec
    )
    overwrite = bool(fields.pop("overwrite", False))
    driver = fields.pop("driver", None)
    if fields:
        names = ", ".join(sorted(fields))
        raise TypeError(f"create() got unexpected keyword arguments: {names}")
    return _choose_create_driver(driver, metadata).create_metadata(
        location, metadata, overwrite=overwrite
    )


def create_group(
    location: tz.PathLike, *,
    config: tx.Optional[GroupConfig] = None, **fields: tx.Any,
) -> ZarrGroup:
    """Create a group at *location*, the metadata-free way.

    Pass a [GroupConfig][abczarr.config.GroupConfig] as *config*, or its
    fields (`zarr_version`, `overwrite`, ...) as keyword arguments.
    """
    base = config if isinstance(config, GroupConfig) else GroupConfig(
        **dict(config or {})
    )
    node = create(location, base, **fields)
    if not isinstance(node, ZarrGroup):
        raise UnsupportedZarrOperation("create_group produced a non-group")
    return node


def _choose_create_driver(driver: _DriverArg, metadata: tx.Any) -> Driver:
    """The driver to create with -- the array's features decide when there is
    a choice, else the first available."""
    drivers = _resolve_drivers(driver)
    if len(drivers) == 1:
        return drivers[0]
    if isinstance(metadata, ArrayMetadata):
        return select_driver(metadata, drivers)
    return drivers[0]


def _resolve_drivers(driver: _DriverArg) -> "tx.List[Driver]":
    if isinstance(driver, Driver):
        return [driver]
    drivers = available_drivers()
    if driver is None:
        if not drivers:
            raise UnsupportedZarrOperation("open (no backend installed)")
        return drivers
    named = [d for d in drivers if d.name == driver]
    if not named:
        raise UnsupportedZarrOperation(
            "open", driver if isinstance(driver, str) else None
        )
    return named


def _choose(path: tz.PathLike, drivers: "tx.List[Driver]") -> Driver:
    """The driver to open *path* with -- selected by the array's features
    when there is a choice, else the first available."""
    if len(drivers) == 1:
        return drivers[0]
    metadata = _peek_array_metadata(path)
    if metadata is not None:
        return select_driver(metadata, drivers)
    return drivers[0]


def _peek_array_metadata(path: tz.PathLike) -> tx.Any:
    """Read an array's metadata straight from the store, for selection, or
    ``None`` when *path* is a group or its metadata cannot be read."""
    from .metadata import v3

    try:
        raw = PathStore(str(path)).get("zarr.json")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    is_v3_array = (
        isinstance(data, dict)
        and data.get("node_type") == "array"
        and data.get("zarr_format") == 3
    )
    if not is_v3_array:
        return None
    try:
        return v3.ArrayMetadata.from_dict(data)
    except Exception:
        return None
