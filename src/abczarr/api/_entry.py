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
from .._core import typing as tz
from .._core.attrs import evolve
from ..abc.array import ZarrArray
from ..abc.async_array import AsyncZarrArray
from ..abc.async_group import AsyncZarrGroup
from ..abc.async_node import AsyncZarrNode
from ..abc.errors import UnsupportedZarrOperation
from ..abc.group import ZarrGroup
from ..abc.node import ZarrNode
from ..abc.store import PathBasedStore
from ..drivers.base import Driver
from ..metadata.base import ArrayMetadata, NodeMetadata
from .config import ArrayConfig, GroupConfig, ZarrConfig
from .registry import available_drivers, select_driver

_DriverArg = tx.Optional[tx.Union[str, Driver]]


@tx.overload
def open(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[False]" = ..., driver: _DriverArg = ...,
) -> ZarrNode: ...
@tx.overload
def open(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[True]", driver: _DriverArg = ...,
) -> tx.Awaitable[AsyncZarrNode]: ...


def open(
    path: tz.PathLike, mode: str = "a", *,
    asynchronous: bool = False, driver: _DriverArg = None,
) -> tx.Union[ZarrNode, tx.Awaitable[AsyncZarrNode]]:
    """Open the Zarr array or group at *path*.

    With `asynchronous=True` the return value is a **coroutine you await**: it
    opens through the backend's own async I/O -- the metadata read is awaited
    -- and resolves to the coroutine twin of the node. Without it, the array
    or group is opened synchronously and returned directly.

    !!! example
        ```python
        node = abczarr.open("data.zarr")                       # sync node
        node = await abczarr.open("data.zarr", asynchronous=True)  # async twin
        ```

    Parameters
    ----------
    path : PathLike
        A local path or a URL (`"s3://bucket/dataset.zarr"`).
    mode : str
        The access mode, as zarr-python understands it (`"r"`, `"r+"`,
        `"a"`, `"w"`).
    asynchronous : bool, optional
        When true, return a coroutine that opens *path* asynchronously and
        resolves to the coroutine twin -- an
        [AsyncZarrArray][abczarr.abc.async_array.AsyncZarrArray] or
        [AsyncZarrGroup][abczarr.abc.async_group.AsyncZarrGroup] -- whose I/O
        is awaited. Whether that surface is native to the backend or
        synthesized in a thread pool is reported by
        `node.supports("async", native=True)`. When false (the default), open
        synchronously and return the node directly.
    driver : str or Driver, optional
        A driver, or its name, to open with. When omitted, a driver is
        chosen for what the array needs.

    Returns
    -------
    ZarrNode or Awaitable[AsyncZarrNode]
        The wrapped array or group directly, or -- when *asynchronous* is
        true -- a coroutine resolving to its async twin.
    """
    if asynchronous:
        return _aopen(path, mode, driver)
    chosen = _choose(path, _resolve_drivers(driver))
    return chosen.open(path, mode)


async def _aopen(
    path: tz.PathLike, mode: str, driver: _DriverArg
) -> AsyncZarrNode:
    """Open *path* asynchronously: pick a driver through an async metadata
    peek, then await its native async open."""
    drivers = _resolve_drivers(driver)
    chosen = await _achoose(path, drivers)
    return await chosen.open_async(path, mode)


@tx.overload
def open_array(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[False]" = ..., driver: _DriverArg = ...,
) -> ZarrArray: ...
@tx.overload
def open_array(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[True]", driver: _DriverArg = ...,
) -> tx.Awaitable[AsyncZarrArray]: ...


def open_array(
    path: tz.PathLike, mode: str = "a", *,
    asynchronous: bool = False, driver: _DriverArg = None,
) -> tx.Union[ZarrArray, tx.Awaitable[AsyncZarrArray]]:
    """Open *path*, requiring it to be an array.

    Like [open][abczarr.api.open], but raises if *path* is a group. When
    *asynchronous* is true, returns a coroutine resolving to the async array
    twin.
    """
    if asynchronous:
        return _aopen_array(path, mode, driver)
    node = open(path, mode, driver=driver)
    if not isinstance(node, ZarrArray):
        raise UnsupportedZarrOperation("open_array on a group")
    return node


async def _aopen_array(
    path: tz.PathLike, mode: str, driver: _DriverArg
) -> AsyncZarrArray:
    node = await _aopen(path, mode, driver)
    if not isinstance(node, AsyncZarrArray):
        raise UnsupportedZarrOperation("open_array on a group")
    return node


@tx.overload
def open_group(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[False]" = ..., driver: _DriverArg = ...,
) -> ZarrGroup: ...
@tx.overload
def open_group(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[True]", driver: _DriverArg = ...,
) -> tx.Awaitable[AsyncZarrGroup]: ...


def open_group(
    path: tz.PathLike, mode: str = "a", *,
    asynchronous: bool = False, driver: _DriverArg = None,
) -> tx.Union[ZarrGroup, tx.Awaitable[AsyncZarrGroup]]:
    """Open *path*, requiring it to be a group.

    Like [open][abczarr.api.open], but raises if *path* is an array. When
    *asynchronous* is true, returns a coroutine resolving to the async group
    twin.
    """
    if asynchronous:
        return _aopen_group(path, mode, driver)
    node = open(path, mode, driver=driver)
    if not isinstance(node, ZarrGroup):
        raise UnsupportedZarrOperation("open_group on an array")
    return node


async def _aopen_group(
    path: tz.PathLike, mode: str, driver: _DriverArg
) -> AsyncZarrGroup:
    node = await _aopen(path, mode, driver)
    if not isinstance(node, AsyncZarrGroup):
        raise UnsupportedZarrOperation("open_group on an array")
    return node


@tx.overload
def create(
    location: tz.PathLike, config: ArrayConfig, **fields: tx.Any
) -> ZarrArray: ...
@tx.overload
def create(
    location: tz.PathLike, config: GroupConfig, **fields: tx.Any
) -> ZarrGroup: ...
@tx.overload
def create(
    location: tz.PathLike, config: NodeMetadata, **fields: tx.Any
) -> ZarrNode: ...


def create(
    location: tz.PathLike,
    config: tx.Union[ZarrConfig, NodeMetadata],
    **fields: tx.Any,
) -> ZarrNode:
    """Create the array or group *config* describes at *location*.

    *config* is usually an [ArrayConfig][abczarr.api.config.ArrayConfig]
    (creates an array) or a [GroupConfig][abczarr.api.config.GroupConfig]
    (creates a
    group), and keyword arguments override its fields; the backend creates the
    node natively. For full control beyond what the config helpers express,
    *config* may instead be an exact metadata document -- an
    [ArrayMetadata][abczarr.metadata.base.ArrayMetadata] or
    [GroupMetadata][abczarr.metadata.base.GroupMetadata], the lowered form a
    config would produce -- which is created as it is; there `driver` and
    `overwrite` are the only keywords. For a plain dict, wrap it first with
    `ArrayMetadata.from_dict(...)` or `ArrayConfig(**...)`.
    """
    if isinstance(config, ZarrConfig):
        if fields:
            config = evolve(config, **fields)
        if isinstance(config, ArrayConfig):
            config = config.resolve()
            metadata = config.to_metadata()
        else:
            metadata = None
        return _choose_create_driver(config.driver, metadata).create(
            location, config
        )
    if isinstance(config, NodeMetadata):
        overwrite = bool(fields.pop("overwrite", False))
        driver = fields.pop("driver", None)
        if fields:
            names = ", ".join(sorted(fields))
            raise TypeError(
                "create() from a metadata document got unexpected keyword "
                f"arguments: {names}"
            )
        return _choose_create_driver(
            driver, config
        ).create_from_metadata(location, config, overwrite=overwrite)
    raise TypeError(
        "create() takes a config (ArrayConfig/GroupConfig) or a metadata "
        "document (ArrayMetadata/GroupMetadata); for a dict, wrap it with "
        "ArrayMetadata.from_dict(...) or ArrayConfig(**...)"
    )


def create_group(
    location: tz.PathLike, *,
    config: tx.Optional[GroupConfig] = None, **fields: tx.Any,
) -> ZarrGroup:
    """Create a group at *location*, the metadata-free way.

    Pass a [GroupConfig][abczarr.api.config.GroupConfig] as *config*, or its
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
    from ..metadata import v3

    try:
        raw = PathBasedStore(str(path)).get("zarr.json")
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


async def _achoose(path: tz.PathLike, drivers: "tx.List[Driver]") -> Driver:
    """The driver to open *path* with, selected through an async metadata
    peek -- the async twin of [_choose][abczarr.api._entry]."""
    if len(drivers) == 1:
        return drivers[0]
    metadata = await _apeek_array_metadata(path)
    if metadata is not None:
        return select_driver(metadata, drivers)
    return drivers[0]


async def _apeek_array_metadata(path: tz.PathLike) -> tx.Any:
    """Read an array's metadata through an async store, for selection, or
    ``None`` when *path* is a group or its metadata cannot be read -- the
    async twin of [_peek_array_metadata][abczarr.api._entry]."""
    from ..abc.store import AsyncPathBasedStore
    from ..metadata import v3

    try:
        raw = await AsyncPathBasedStore(str(path)).get("zarr.json")
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
