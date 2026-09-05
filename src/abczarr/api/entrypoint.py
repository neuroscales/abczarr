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
    "create_array",
    "create_group",
]

# stdlib
import json

# dependencies
import typing_extensions as tx

from abczarr.errors import UnsupportedZarrOperation

# locals
from .._core import constants
from .._core import typing as tz
from .._core.attrs import evolve, fields
from ..abc.asynchronous import (
    AsyncZarrArray,
    AsyncZarrGroup,
    AsyncZarrNode,
)
from ..abc.store import AsyncPathBasedStore, PathBasedStore
from ..abc.sync import ZarrArray, ZarrGroup, ZarrNode
from ..drivers.base import Driver
from ..metadata import v3
from ..metadata.base import ArrayMetadata, NodeMetadata
from .config import ArrayConfig, GroupConfig, ZarrConfig
from .registry import available_drivers, select_driver

_DriverArg = tx.Optional[tx.Union[str, Driver]]

#: The access modes that open an existing node (never create).
_READ_MODES = frozenset({"r", "r+"})

#: The metadata keys whose presence marks a node at a location, across the
#: format versions (a v3 ``zarr.json``, a v2 ``.zgroup`` / ``.zarray``, a v1
#: ``meta``).
_METADATA_KEYS = (
    constants.Z3_JSON,
    constants.Z2GROUP_JSON,
    constants.Z2ARRAY_JSON,
    constants.Z1META_JSON,
)


def _array_only_fields() -> "tx.FrozenSet[str]":
    """The [ArrayConfig][abczarr.api.config.ArrayConfig] fields that a
    [GroupConfig][abczarr.api.config.GroupConfig] does not have -- the fields
    whose presence means the caller is describing an array, not a group."""
    group_names = {f.name for f in fields(GroupConfig)}
    return frozenset(
        f.name for f in fields(ArrayConfig) if f.name not in group_names
    )


_ARRAY_ONLY_FIELDS = _array_only_fields()


def _create_plan(mode: str, exists: bool) -> tx.Optional[bool]:
    """Whether *mode* creates at a location that does (or does not) *exist*.

    Returns ``None`` to open the existing node, or the ``overwrite`` flag to
    pass to [create][abczarr.api.create]. ``"w"`` overwrites; ``"w-"`` /
    ``"x"`` create and fail if something is already there; ``"a"`` creates
    only when nothing exists; every other mode (``"r"``, ``"r+"``) opens.
    """
    if mode == "w":
        return True
    if mode in ("w-", "x"):
        return False
    if mode == "a":
        return None if exists else False
    return None


def _build_create_config(
    create_fields: "tx.Dict[str, tx.Any]",
    want: tx.Optional[str],
) -> tx.Union[ArrayConfig, GroupConfig]:
    """Build the config a create-mode open funnels into
    [create][abczarr.api.create].

    An array config when *create_fields* carry array parameters (a `shape`,
    a `dtype`, ...), a group config otherwise. *want* pins the kind --
    ``"array"`` or ``"group"`` for [open_array][abczarr.api.open_array] /
    [open_group][abczarr.api.open_group], ``None`` to decide from the fields.
    """
    if "overwrite" in create_fields:
        raise TypeError(
            "open() takes whether to overwrite from its mode, not an "
            "overwrite= field: use mode=\"w\" to overwrite, or "
            "mode=\"w-\" to fail if the target exists"
        )
    given_array = _ARRAY_ONLY_FIELDS & set(create_fields)
    if want == "group":
        if given_array:
            names = ", ".join(sorted(given_array))
            raise TypeError(
                "open_group() creates a group, which takes no array fields; "
                f"got {names}. Use open() or open_array() to create an array."
            )
        return GroupConfig(**create_fields)
    if want == "array" and "shape" not in create_fields:
        got = ", ".join(sorted(create_fields)) or "no creation fields"
        raise TypeError(
            "open_array() with a create mode needs at least a shape (and a "
            f"dtype) to create the array; got {got}"
        )
    if want == "array" or given_array:
        return ArrayConfig(**create_fields)
    return GroupConfig(**create_fields)


def _exists(path: tz.PathLike) -> bool:
    """Whether a Zarr node is present at *path*, read through a store."""
    try:
        store = PathBasedStore(str(path))
    except Exception:
        return False
    for key in _METADATA_KEYS:
        try:
            if store.exists(key):
                return True
        except Exception:
            return False
    return False


async def _aexists(path: tz.PathLike) -> bool:
    """The async twin of [_exists][abczarr.api.entrypoint], through an async
    store."""
    try:
        store = AsyncPathBasedStore(str(path))
    except Exception:
        return False
    for key in _METADATA_KEYS:
        try:
            if await store.exists(key):
                return True
        except Exception:
            return False
    return False


def _require_kind(node: tx.Any, want: tx.Optional[str]) -> tx.Any:
    """Return *node* when it matches *want* (``"array"`` / ``"group"`` / any),
    else raise -- shared by both colors, since it only checks the kind."""
    if want == "array" and not isinstance(node, (ZarrArray, AsyncZarrArray)):
        raise UnsupportedZarrOperation("open_array on a group")
    if want == "group" and not isinstance(node, (ZarrGroup, AsyncZarrGroup)):
        raise UnsupportedZarrOperation("open_group on an array")
    return node


@tx.overload
def open(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[False]" = ..., driver: _DriverArg = ...,
    **fields: tx.Any,
) -> ZarrNode: ...
@tx.overload
def open(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[True]", driver: _DriverArg = ...,
    **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrNode]: ...


def open(
    path: tz.PathLike, mode: str = "a", *,
    asynchronous: bool = False, driver: _DriverArg = None,
    **fields: tx.Any,
) -> tx.Union[ZarrNode, tx.Awaitable[AsyncZarrNode]]:
    """Open -- or, on a create mode, create -- the Zarr node at *path*.

    The *mode* follows the h5py/zarr convention, so `open` both opens an
    existing node and creates a new one:

    | mode         | if it exists     | if it is missing |
    | ------------ | ---------------- | ---------------- |
    | `"r"`        | open read-only   | error            |
    | `"r+"`       | open read-write  | error            |
    | `"a"`        | open             | create           |
    | `"w"`        | overwrite        | create           |
    | `"w-"`, `"x"`| error            | create           |

    On a create mode the keyword *fields* describe the new node -- the same
    fields [create][abczarr.api.create],
    [ArrayConfig][abczarr.api.config.ArrayConfig] and
    [GroupConfig][abczarr.api.config.GroupConfig] accept. Array parameters
    (a `shape`, a `dtype`, ...) create an array; with none, an empty group is
    created.

    With `asynchronous=True` the return value is a **coroutine you await**: it
    opens (or creates) through the backend's own async I/O and resolves to the
    coroutine twin of the node. Without it, the node is opened synchronously
    and returned directly.

    !!! example
        ```python
        node = abczarr.open("data.zarr")                       # open (or make)
        arr = abczarr.open("a.zarr", mode="w", shape=(4, 4), dtype="int8")
        grp = abczarr.open("g.zarr", mode="w")                 # an empty group
        node = await abczarr.open("data.zarr", asynchronous=True)  # async twin
        ```

    Parameters
    ----------
    path : PathLike
        A local path or a URL (`"s3://bucket/dataset.zarr"`).
    mode : str
        The access mode, in the h5py/zarr convention (see the table above).
        `"r"` / `"r+"` open an existing node (read-only / read-write) and error
        if it is missing; `"a"` (the default) opens it or creates one when
        nothing is there; `"w"` creates, overwriting whatever is there; `"w-"`
        (or its alias `"x"`) creates and fails if the target already exists.
    asynchronous : bool, optional
        When true, return a coroutine that opens or creates *path*
        asynchronously and resolves to the coroutine twin -- an
        [AsyncZarrArray][abczarr.abc.asynchronous.AsyncZarrArray] or
        [AsyncZarrGroup][abczarr.abc.asynchronous.AsyncZarrGroup] -- whose I/O
        is awaited. Whether that surface is native to the backend or
        synthesized in a thread pool is reported by
        `node.supports("async", native=True)`. When false (the default), open
        synchronously and return the node directly.
    driver : str or Driver, optional
        A driver, or its name, to open or create with. When omitted, a driver
        is chosen for what the node needs.
    **fields
        Creation parameters, consulted only on a create mode -- the fields an
        [ArrayConfig][abczarr.api.config.ArrayConfig] or
        [GroupConfig][abczarr.api.config.GroupConfig] accepts.

    Returns
    -------
    ZarrNode or Awaitable[AsyncZarrNode]
        The wrapped node directly, or -- when *asynchronous* is true -- a
        coroutine resolving to its async twin.
    """
    if asynchronous:
        return _aopen(path, mode, driver, fields, None)
    return _open(path, mode, driver, fields, None)


def _open(
    path: tz.PathLike, mode: str, driver: _DriverArg,
    create_fields: "tx.Dict[str, tx.Any]", want: tx.Optional[str],
) -> ZarrNode:
    """Open *path*, or create it when *mode* is a create mode."""
    overwrite = _create_plan(mode, _exists(path) if mode == "a" else False)
    if overwrite is None:
        if create_fields and mode not in ("a",):
            _reject_open_fields(mode, create_fields)
        chosen = _choose(path, _resolve_drivers(driver))
        return _require_kind(chosen.open(path, mode), want)
    config = _build_create_config(create_fields, want)
    extra = {"overwrite": overwrite}  # type: tx.Dict[str, tx.Any]
    if driver is not None:
        extra["driver"] = driver
    return _require_kind(create(path, config, **extra), want)


async def _aopen(
    path: tz.PathLike, mode: str, driver: _DriverArg,
    create_fields: "tx.Dict[str, tx.Any]", want: tx.Optional[str],
) -> AsyncZarrNode:
    """Open *path* asynchronously, or create it on a create mode: pick a
    driver through an async metadata peek, then await the native open or
    create."""
    exists = await _aexists(path) if mode == "a" else False
    overwrite = _create_plan(mode, exists)
    if overwrite is None:
        if create_fields and mode not in ("a",):
            _reject_open_fields(mode, create_fields)
        chosen = await _achoose(path, _resolve_drivers(driver))
        node = await chosen.open(path, mode, asynchronous=True)
        return _require_kind(node, want)
    config = _build_create_config(create_fields, want)
    extra = {"overwrite": overwrite}  # type: tx.Dict[str, tx.Any]
    if driver is not None:
        extra["driver"] = driver
    node = await create(path, config, asynchronous=True, **extra)
    return _require_kind(node, want)


def _reject_open_fields(
    mode: str, create_fields: "tx.Dict[str, tx.Any]"
) -> None:
    names = ", ".join(sorted(create_fields))
    raise TypeError(
        f"open(mode={mode!r}) opens an existing node, so the creation fields "
        f"({names}) have no effect; pass them with a create mode "
        "(\"w\", \"w-\"/\"x\", or \"a\" when nothing exists yet)"
    )


@tx.overload
def open_array(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[False]" = ..., driver: _DriverArg = ...,
    **fields: tx.Any,
) -> ZarrArray: ...
@tx.overload
def open_array(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[True]", driver: _DriverArg = ...,
    **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrArray]: ...


def open_array(
    path: tz.PathLike, mode: str = "a", *,
    asynchronous: bool = False, driver: _DriverArg = None,
    **fields: tx.Any,
) -> tx.Union[ZarrArray, tx.Awaitable[AsyncZarrArray]]:
    """Open *path*, requiring it to be an array.

    Like [open][abczarr.api.open], but raises if *path* is a group. On a
    create mode the *fields* must include array parameters (at least a
    `shape`); creating a group this way is an error. When *asynchronous* is
    true, returns a coroutine resolving to the async array twin.
    """
    if asynchronous:
        return _aopen(path, mode, driver, fields, "array")
    return tx.cast(ZarrArray, _open(path, mode, driver, fields, "array"))


@tx.overload
def open_group(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[False]" = ..., driver: _DriverArg = ...,
    **fields: tx.Any,
) -> ZarrGroup: ...
@tx.overload
def open_group(
    path: tz.PathLike, mode: str = ..., *,
    asynchronous: "tx.Literal[True]", driver: _DriverArg = ...,
    **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrGroup]: ...


def open_group(
    path: tz.PathLike, mode: str = "a", *,
    asynchronous: bool = False, driver: _DriverArg = None,
    **fields: tx.Any,
) -> tx.Union[ZarrGroup, tx.Awaitable[AsyncZarrGroup]]:
    """Open *path*, requiring it to be a group.

    Like [open][abczarr.api.open], but raises if *path* is an array. On a
    create mode an (empty) group is created; array parameters are an error.
    When *asynchronous* is true, returns a coroutine resolving to the async
    group twin.
    """
    if asynchronous:
        return _aopen(path, mode, driver, fields, "group")
    return tx.cast(ZarrGroup, _open(path, mode, driver, fields, "group"))


@tx.overload
def create(
    location: tz.PathLike, config: ArrayConfig, *,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrArray: ...
@tx.overload
def create(
    location: tz.PathLike, config: ArrayConfig, *,
    asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrArray]: ...
@tx.overload
def create(
    location: tz.PathLike, config: GroupConfig, *,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrGroup: ...
@tx.overload
def create(
    location: tz.PathLike, config: GroupConfig, *,
    asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrGroup]: ...
@tx.overload
def create(
    location: tz.PathLike, config: NodeMetadata, *,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrNode: ...
@tx.overload
def create(
    location: tz.PathLike, config: NodeMetadata, *,
    asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrNode]: ...


def create(
    location: tz.PathLike,
    config: tx.Union[ZarrConfig, NodeMetadata],
    *, asynchronous: bool = False,
    **fields: tx.Any,
) -> tx.Union[ZarrNode, tx.Awaitable[AsyncZarrNode]]:
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
    `ArrayMetadata.from_json(...)` or `ArrayConfig(**...)`.

    With `asynchronous=True` the return value is a **coroutine you await**: the
    backend creates through its own async I/O and resolves to the coroutine
    twin of the node, mirroring async [open][abczarr.api.open].

    !!! example
        ```python
        arr = abczarr.create("a.zarr", ArrayConfig(shape=(4, 4), dtype="i1"))
        arr = await abczarr.create(
            "a.zarr", ArrayConfig(shape=(4, 4), dtype="i1"), asynchronous=True
        )
        ```
    """
    if asynchronous:
        return _acreate(location, config, fields)
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
        overwrite, driver = _metadata_create_keywords(fields)
        return _choose_create_driver(
            driver, config
        ).create_from_metadata(location, config, overwrite=overwrite)
    raise TypeError(_CREATE_TYPE_ERROR)


async def _acreate(
    location: tz.PathLike,
    config: tx.Union[ZarrConfig, NodeMetadata],
    fields: "tx.Dict[str, tx.Any]",
) -> AsyncZarrNode:
    """Create *config* at *location* asynchronously, awaiting the backend's
    native async create -- the async twin of [create][abczarr.api.create]."""
    if isinstance(config, ZarrConfig):
        if fields:
            config = evolve(config, **fields)
        if isinstance(config, ArrayConfig):
            config = config.resolve()
            metadata = config.to_metadata()  # type: tx.Any
        else:
            metadata = None
        driver = _choose_create_driver(config.driver, metadata)
        return await driver.create(location, config, asynchronous=True)
    if isinstance(config, NodeMetadata):
        overwrite, driver_arg = _metadata_create_keywords(fields)
        driver = _choose_create_driver(driver_arg, config)
        return await driver.create_from_metadata(
            location, config, overwrite=overwrite, asynchronous=True
        )
    raise TypeError(_CREATE_TYPE_ERROR)


_CREATE_TYPE_ERROR = (
    "create() takes a config (ArrayConfig/GroupConfig) or a metadata "
    "document (ArrayMetadata/GroupMetadata); for a dict, wrap it with "
    "ArrayMetadata.from_json(...) or ArrayConfig(**...)"
)


def _metadata_create_keywords(
    fields: "tx.Dict[str, tx.Any]",
) -> "tx.Tuple[bool, _DriverArg]":
    """The `overwrite` and `driver` keywords a metadata-document create takes,
    rejecting anything else."""
    overwrite = bool(fields.pop("overwrite", False))
    driver = fields.pop("driver", None)
    if fields:
        names = ", ".join(sorted(fields))
        raise TypeError(
            "create() from a metadata document got unexpected keyword "
            f"arguments: {names}"
        )
    return overwrite, driver


@tx.overload
def create_group(
    location: tz.PathLike, *,
    config: tx.Optional[GroupConfig] = ...,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrGroup: ...
@tx.overload
def create_group(
    location: tz.PathLike, *,
    config: tx.Optional[GroupConfig] = ...,
    asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrGroup]: ...


def create_group(
    location: tz.PathLike, *,
    config: tx.Optional[GroupConfig] = None,
    asynchronous: bool = False, **fields: tx.Any,
) -> tx.Union[ZarrGroup, tx.Awaitable[AsyncZarrGroup]]:
    """Create a group at *location*, the metadata-free way.

    Pass a [GroupConfig][abczarr.api.config.GroupConfig] as *config*, or its
    fields (`zarr_version`, `overwrite`, ...) as keyword arguments. With
    `asynchronous=True` the return value is a coroutine resolving to the async
    group twin, mirroring async [create][abczarr.api.create].
    """
    base = config if isinstance(config, GroupConfig) else GroupConfig(
        **dict(config or {})
    )
    if asynchronous:
        return _acreate_group(location, base, fields)
    node = create(location, base, **fields)
    if not isinstance(node, ZarrGroup):
        raise UnsupportedZarrOperation("create_group produced a non-group")
    return node


async def _acreate_group(
    location: tz.PathLike, base: GroupConfig, fields: "tx.Dict[str, tx.Any]",
) -> AsyncZarrGroup:
    node = await create(location, base, asynchronous=True, **fields)
    if not isinstance(node, AsyncZarrGroup):
        raise UnsupportedZarrOperation("create_group produced a non-group")
    return node


@tx.overload
def create_array(
    location: tz.PathLike, *,
    config: tx.Optional[ArrayConfig] = ...,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrArray: ...
@tx.overload
def create_array(
    location: tz.PathLike, *,
    config: tx.Optional[ArrayConfig] = ...,
    asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrArray]: ...


def create_array(
    location: tz.PathLike, *,
    config: tx.Optional[ArrayConfig] = None,
    asynchronous: bool = False, **fields: tx.Any,
) -> tx.Union[ZarrArray, tx.Awaitable[AsyncZarrArray]]:
    """Create an array at *location*, the metadata-free way.

    Pass an [ArrayConfig][abczarr.api.config.ArrayConfig] as *config*, or its
    fields (`shape`, `dtype`, `chunks`, ...) as keyword arguments. At least a
    `shape` (and a `dtype`) is needed to describe the array; a request with no
    array fields is a group, so use [create_group][abczarr.api.create_group]
    for that. With `asynchronous=True` the return value is a coroutine
    resolving to the async array twin, mirroring async
    [create][abczarr.api.create].
    """
    base = config if isinstance(config, ArrayConfig) else ArrayConfig(
        **dict(config or {})
    )
    if fields.get("shape", base.shape) is None:
        got = ", ".join(sorted(fields)) or "no creation fields"
        raise TypeError(
            "create_array() needs at least a shape (and a dtype) to create "
            f"the array; got {got}. Use create_group() to create a group."
        )
    if asynchronous:
        return _acreate_array(location, base, fields)
    node = create(location, base, **fields)
    if not isinstance(node, ZarrArray):
        raise UnsupportedZarrOperation("create_array produced a non-array")
    return node


async def _acreate_array(
    location: tz.PathLike, base: ArrayConfig, fields: "tx.Dict[str, tx.Any]",
) -> AsyncZarrArray:
    node = await create(location, base, asynchronous=True, **fields)
    if not isinstance(node, AsyncZarrArray):
        raise UnsupportedZarrOperation("create_array produced a non-array")
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
        return v3.ArrayMetadata.from_json(data)
    except Exception:
        return None


async def _achoose(path: tz.PathLike, drivers: "tx.List[Driver]") -> Driver:
    """The driver to open *path* with, selected through an async metadata
    peek -- the async twin of [_choose][abczarr.api.entrypoint]."""
    if len(drivers) == 1:
        return drivers[0]
    metadata = await _apeek_array_metadata(path)
    if metadata is not None:
        return select_driver(metadata, drivers)
    return drivers[0]


async def _apeek_array_metadata(path: tz.PathLike) -> tx.Any:
    """Read an array's metadata through an async store, for selection, or
    ``None`` when *path* is a group or its metadata cannot be read -- the
    async twin of [_peek_array_metadata][abczarr.api.entrypoint]."""
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
        return v3.ArrayMetadata.from_json(data)
    except Exception:
        return None
