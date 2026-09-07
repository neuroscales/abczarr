"""Open Zarr arrays and groups through a selected backend driver.

[open][abczarr.api.open] opens a location and returns the array or
group there, wrapped in the uniform surface. When no driver is
named, it reads the array's metadata and picks a driver that
provides every codec the array needs, described in
[select_driver][abczarr.drivers.base.select_driver], then opens
through that driver. [open_array][abczarr.api.open_array] and
[open_group][abczarr.api.open_group] additionally check what they
opened.
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
    [GroupConfig][abczarr.api.config.GroupConfig] does not have.

    The presence of one of these fields means the caller is
    describing an array, not a group.
    """
    group_names = {f.name for f in fields(GroupConfig)}
    return frozenset(
        f.name for f in fields(ArrayConfig) if f.name not in group_names
    )


_ARRAY_ONLY_FIELDS = _array_only_fields()


def _create_plan(mode: str, exists: bool) -> tx.Optional[bool]:
    """Whether *mode* creates at a location that does, or does not,
    *exist*.

    The return value is ``None`` to open the existing node, or the
    ``overwrite`` flag to pass to [create][abczarr.api.create].
    ``"w"`` overwrites. ``"w-"`` and ``"x"`` create, and fail if
    something is already there. ``"a"`` creates only when nothing
    exists. Every other mode, ``"r"`` and ``"r+"``, opens.
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

    The result is an array config when *create_fields* carry array
    parameters, such as a `shape` or a `dtype`, and a group config
    otherwise. *want* pins the kind: ``"array"`` or ``"group"`` for
    [open_array][abczarr.api.open_array] and
    [open_group][abczarr.api.open_group], or ``None`` to decide from
    the fields.
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
    """Return *node* when it matches *want*, else raise.

    *want* is ``"array"``, ``"group"``, or ``None`` to accept either
    kind. This function is shared by both the sync and async colors,
    since it only checks the kind.
    """
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
    """Open, or on a create mode create, the Zarr node at *path*.

    The *mode* follows the h5py/zarr convention, so `open` both opens an
    existing node and creates a new one:

    | mode         | if it exists     | if it is missing |
    | ------------ | ---------------- | ---------------- |
    | `"r"`        | open read-only   | error            |
    | `"r+"`       | open read-write  | error            |
    | `"a"`        | open             | create           |
    | `"w"`        | overwrite        | create           |
    | `"w-"`, `"x"`| error            | create           |

    On a create mode the keyword *fields* describe the new node.
    These are the same fields [create][abczarr.api.create],
    [ArrayConfig][abczarr.api.config.ArrayConfig] and
    [GroupConfig][abczarr.api.config.GroupConfig] accept. Array
    parameters, such as a `shape` or a `dtype`, create an array. With
    none, an empty group is created instead.

    With `asynchronous=True` the return value is a coroutine that must
    be awaited. It opens or creates *path* through the backend's own
    async I/O and resolves to the coroutine twin of the node. Without
    it, the node is opened synchronously and returned directly.

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
        A local path or a URL, such as `"s3://bucket/dataset.zarr"`.
    mode : str
        The access mode, in the h5py/zarr convention described in the table
        above. `"r"` and `"r+"` open an existing node, read-only and
        read-write respectively, and error if it is missing. `"a"`, the
        default, opens it or creates one when nothing is there. `"w"`
        creates, overwriting whatever is there. `"w-"`, or its alias `"x"`,
        creates and fails if the target already exists.
    asynchronous : bool, optional
        When true, return a coroutine that opens or creates *path*
        asynchronously and resolves to the coroutine twin, an
        [AsyncZarrArray][abczarr.abc.asynchronous.AsyncZarrArray] or
        [AsyncZarrGroup][abczarr.abc.asynchronous.AsyncZarrGroup], whose I/O
        is awaited. Whether that surface is native to the backend or
        synthesized in a thread pool is reported by
        `node.supports("async", native=True)`. When false, the default, the
        node is opened synchronously and returned directly.
    driver : str or Driver, optional
        A driver, or its name, to open or create with. When omitted, a driver
        is chosen for what the node needs.
    **fields
        Creation parameters, consulted only on a create mode. These are the
        fields an [ArrayConfig][abczarr.api.config.ArrayConfig] or
        [GroupConfig][abczarr.api.config.GroupConfig] accepts.

    Returns
    -------
    ZarrNode or Awaitable[AsyncZarrNode]
        The wrapped node directly, or, when *asynchronous* is true, a
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
    """Open *path* asynchronously, or create it on a create mode.

    A driver is chosen through an async metadata peek, and the
    native open or create is then awaited.
    """
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

    This function behaves like [open][abczarr.api.open], but raises
    if *path* is a group. On a create mode, *fields* must include
    array parameters, at least a `shape`. Creating a group this way
    is an error.

    Parameters
    ----------
    path : PathLike
        A local path or a URL, such as `"s3://bucket/dataset.zarr"`.
    mode : str
        The access mode. See [open][abczarr.api.open] for the modes and
        what each one does.
    asynchronous : bool, optional
        When true, return a coroutine that opens or creates *path*
        asynchronously and resolves to the async array twin. When false,
        the default, the array is opened synchronously and returned
        directly.
    driver : str or Driver, optional
        A driver, or its name, to open or create with. When omitted, a driver
        is chosen for what the array needs.
    **fields
        Creation parameters, consulted only on a create mode. These are the
        fields an [ArrayConfig][abczarr.api.config.ArrayConfig] accepts.

    Returns
    -------
    ZarrArray or Awaitable[AsyncZarrArray]
        The wrapped array directly, or, when *asynchronous* is true, a
        coroutine resolving to its async twin.
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

    This function behaves like [open][abczarr.api.open], but raises
    if *path* is an array. On a create mode, an empty group is
    created. Array parameters are an error.

    Parameters
    ----------
    path : PathLike
        A local path or a URL, such as `"s3://bucket/dataset.zarr"`.
    mode : str
        The access mode. See [open][abczarr.api.open] for the modes and
        what each one does.
    asynchronous : bool, optional
        When true, return a coroutine that opens or creates *path*
        asynchronously and resolves to the async group twin. When false,
        the default, the group is opened synchronously and returned
        directly.
    driver : str or Driver, optional
        A driver, or its name, to open or create with. When omitted, a driver
        is chosen for what the group needs.
    **fields
        Creation parameters, consulted only on a create mode. These are the
        fields a [GroupConfig][abczarr.api.config.GroupConfig] accepts.

    Returns
    -------
    ZarrGroup or Awaitable[AsyncZarrGroup]
        The wrapped group directly, or, when *asynchronous* is true, a
        coroutine resolving to its async twin.
    """
    if asynchronous:
        return _aopen(path, mode, driver, fields, "group")
    return tx.cast(ZarrGroup, _open(path, mode, driver, fields, "group"))


@tx.overload
def create(
    location: tz.PathLike, config: tx.Optional[ArrayConfig] = ..., *,
    data: tx.Any, ome: tx.Any = ...,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrArray: ...
@tx.overload
def create(
    location: tz.PathLike, config: tx.Optional[ArrayConfig] = ..., *,
    data: tx.Any, ome: tx.Any = ...,
    asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrArray]: ...
@tx.overload
def create(
    location: tz.PathLike, config: ArrayConfig, *,
    ome: tx.Any = ...,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrArray: ...
@tx.overload
def create(
    location: tz.PathLike, config: ArrayConfig, *,
    ome: tx.Any = ..., asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrArray]: ...
@tx.overload
def create(
    location: tz.PathLike, config: GroupConfig, *,
    ome: tx.Any = ...,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrGroup: ...
@tx.overload
def create(
    location: tz.PathLike, config: GroupConfig, *,
    ome: tx.Any = ..., asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrGroup]: ...
@tx.overload
def create(
    location: tz.PathLike, config: NodeMetadata, *,
    ome: tx.Any = ...,
    asynchronous: "tx.Literal[False]" = ..., **fields: tx.Any,
) -> ZarrNode: ...
@tx.overload
def create(
    location: tz.PathLike, config: NodeMetadata, *,
    ome: tx.Any = ..., asynchronous: "tx.Literal[True]", **fields: tx.Any,
) -> tx.Awaitable[AsyncZarrNode]: ...


def create(
    location: tz.PathLike,
    config: tx.Union[ZarrConfig, NodeMetadata, None] = None,
    *,
    data: tx.Any = None,
    ome: tx.Any = None,
    asynchronous: bool = False,
    **fields: tx.Any,
) -> tx.Union[ZarrNode, tx.Awaitable[AsyncZarrNode]]:
    """Create the array or group *config* describes at *location*.

    *config* is usually an [ArrayConfig][abczarr.api.config.ArrayConfig],
    which creates an array, or a [GroupConfig][abczarr.api.config.GroupConfig],
    which creates a group. Keyword arguments override its fields, and the
    backend creates the node natively. For full control beyond what the
    config helpers express, *config* may instead be an exact metadata
    document, an [ArrayMetadata][abczarr.metadata.base.ArrayMetadata] or
    [GroupMetadata][abczarr.metadata.base.GroupMetadata], the lowered form a
    config would produce. Such a document is created as it is, and there
    `driver` and `overwrite` are the only accepted keywords. For a plain
    dict, wrap it first with `ArrayMetadata.from_json(...)` or
    `ArrayConfig(**...)`.

    An array is created from existing *data* when *data* is given. The array's
    shape and dtype then default to the data's. A `shape` or `dtype` in
    *config* or in the keyword arguments takes precedence over the data. The
    data is written into the new array, and *config*, when given, must be an
    [ArrayConfig][abczarr.api.config.ArrayConfig].

    OME-Zarr metadata is written on the new node when *ome* is given. An
    [OME][abczarr.ome.base.OME] object or a plain mapping is written as it is.
    An [ImageConfig][abczarr.ome.config.ImageConfig] is lowered to base-level
    metadata first.

    With `asynchronous=True` the return value is a coroutine that must
    be awaited. The backend creates through its own async I/O and
    resolves to the coroutine twin of the node, mirroring async
    [open][abczarr.api.open].

    !!! example
        ```python
        arr = abczarr.create("a.zarr", ArrayConfig(shape=(4, 4), dtype="i1"))
        arr = abczarr.create("a.zarr", data=np.zeros((4, 4), "i1"))
        arr = await abczarr.create(
            "a.zarr", ArrayConfig(shape=(4, 4), dtype="i1"), asynchronous=True
        )
        ```

    Parameters
    ----------
    location : PathLike
        A local path or a URL, such as `"s3://bucket/dataset.zarr"`.
    config : ArrayConfig, GroupConfig or NodeMetadata, optional
        What to create. An `ArrayConfig` or `GroupConfig` describes the
        node through abczarr's own creation options. An `ArrayMetadata` or
        `GroupMetadata` document is created exactly as it is. Omitted when
        *data* alone describes the array to create.
    data : array-like, optional
        Existing data to size the new array from and write into it. Requires
        *config* to be an `ArrayConfig`, or omitted entirely.
    ome : OME, ImageConfig or dict, optional
        OME-Zarr metadata to write on the new node.
    asynchronous : bool, optional
        When true, return a coroutine that creates the node asynchronously
        and resolves to its coroutine twin. When false, the default, the
        node is created synchronously and returned directly.
    **fields
        Individual fields that override the same field on *config*, such as
        `chunks` or `compressor`. Not accepted when *config* is a metadata
        document. There, only `driver` and `overwrite` are accepted.

    Returns
    -------
    ZarrNode or Awaitable[AsyncZarrNode]
        The newly created node directly, or, when *asynchronous* is true, a
        coroutine resolving to its async twin.
    """
    if data is not None:
        data = _as_stored(data)
    if asynchronous:
        return _acreate(location, config, data, ome, fields)
    node = _create_node(location, config, data, fields)
    if data is not None:
        node.store(data)
    if ome is not None:
        _apply_ome(node, ome)
    return node


def _create_node(
    location: tz.PathLike,
    config: tx.Union[ZarrConfig, NodeMetadata, None],
    data: tx.Any,
    fields: "tx.Dict[str, tx.Any]",
) -> ZarrNode:
    """Create the node itself, without storing data or writing OME metadata."""
    config = _prepare_config(config, data)
    if isinstance(config, ZarrConfig):
        if fields:
            config = evolve(config, **fields)
        if isinstance(config, ArrayConfig):
            config = config.resolve(data)
            metadata = config.to_metadata()  # type: tx.Any
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


def _prepare_config(
    config: tx.Union[ZarrConfig, NodeMetadata, None], data: tx.Any
) -> tx.Union[ZarrConfig, NodeMetadata]:
    """The config to create from, an array by default when *data* is given."""
    if config is None:
        if data is None:
            raise TypeError(_CREATE_TYPE_ERROR)
        return ArrayConfig()
    if data is not None and isinstance(config, GroupConfig):
        raise TypeError(
            "create() with data creates an array; pass an ArrayConfig or omit "
            "the config"
        )
    return config


def _as_stored(data: tx.Any) -> tx.Any:
    """*data* as an array with a shape and a dtype to read and to store.

    An array, whether numpy, Dask, or another Zarr array, is used as it is.
    Anything else, such as a nested list, is turned into a numpy array first.
    """
    has_shape = getattr(data, "shape", None) is not None
    has_dtype = getattr(data, "dtype", None) is not None
    if has_shape and has_dtype:
        return data
    import numpy as np

    return np.asarray(data)


def _apply_ome(node: ZarrNode, ome: tx.Any) -> None:
    """Write *ome* onto *node*, lowering an ImageConfig to metadata first."""
    from ..ome.config import ImageConfig

    if isinstance(ome, ImageConfig):
        ome.apply(node)
    else:
        node.ome = ome


async def _acreate(
    location: tz.PathLike,
    config: tx.Union[ZarrConfig, NodeMetadata, None],
    data: tx.Any,
    ome: tx.Any,
    fields: "tx.Dict[str, tx.Any]",
) -> AsyncZarrNode:
    """Create asynchronously, the async twin of
    [create][abczarr.api.create]."""
    config = _prepare_config(config, data)
    if isinstance(config, ZarrConfig):
        if fields:
            config = evolve(config, **fields)
        if isinstance(config, ArrayConfig):
            config = config.resolve(data)
            metadata = config.to_metadata()  # type: tx.Any
        else:
            metadata = None
        driver = _choose_create_driver(config.driver, metadata)
        node = await driver.create(
            location, config, asynchronous=True
        )  # type: tx.Any
    elif isinstance(config, NodeMetadata):
        overwrite, driver_arg = _metadata_create_keywords(fields)
        driver = _choose_create_driver(driver_arg, config)
        node = await driver.create_from_metadata(
            location, config, overwrite=overwrite, asynchronous=True
        )
    else:
        raise TypeError(_CREATE_TYPE_ERROR)
    if data is not None:
        await node.setitem(Ellipsis, data)
    if ome is not None:
        await _aapply_ome(node, ome)
    return node


async def _aapply_ome(node: AsyncZarrNode, ome: tx.Any) -> None:
    """Write *ome* onto an async *node*, lowering an ImageConfig."""
    from ..ome.config import ImageConfig

    if isinstance(ome, ImageConfig):
        await node.set_ome(ome.to_ome())
    else:
        await node.set_ome(ome)


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

    Parameters
    ----------
    location : PathLike
        A local path or a URL, such as `"s3://bucket/group.zarr"`.
    config : GroupConfig, optional
        A reusable [GroupConfig][abczarr.api.config.GroupConfig]. Individual
        fields, such as `zarr_version` or `overwrite`, may also be passed as
        keyword arguments, which override the config.
    asynchronous : bool, optional
        When true, return a coroutine that creates the group asynchronously
        and resolves to the async group twin, mirroring async
        [create][abczarr.api.create]. When false, the default, the group is
        created synchronously and returned directly.
    **fields
        Individual [GroupConfig][abczarr.api.config.GroupConfig] fields
        that override the same field on *config*.

    Returns
    -------
    ZarrGroup or Awaitable[AsyncZarrGroup]
        The newly created group directly, or, when *asynchronous* is
        true, a coroutine resolving to its async twin.
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

    At least a `shape`, and a `dtype`, is needed to describe the
    array. A request with no array fields describes a group instead,
    so [create_group][abczarr.api.create_group] should be used for
    that.

    Parameters
    ----------
    location : PathLike
        A local path or a URL, such as `"s3://bucket/array.zarr"`.
    config : ArrayConfig, optional
        A reusable [ArrayConfig][abczarr.api.config.ArrayConfig]. Individual
        fields, such as `shape`, `dtype` or `chunks`, may also be passed as
        keyword arguments, which override the config.
    asynchronous : bool, optional
        When true, return a coroutine that creates the array asynchronously
        and resolves to the async array twin, mirroring async
        [create][abczarr.api.create]. When false, the default, the array is
        created synchronously and returned directly.
    **fields
        Individual [ArrayConfig][abczarr.api.config.ArrayConfig] fields
        that override the same field on *config*.

    Returns
    -------
    ZarrArray or Awaitable[AsyncZarrArray]
        The newly created array directly, or, when *asynchronous* is
        true, a coroutine resolving to its async twin.
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
    """The driver to create with.

    The array's features decide when there is a choice among several
    drivers, else the first available driver is used.
    """
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
    """The driver to open *path* with.

    The array's features decide when there is a choice among several
    drivers, else the first available driver is used.
    """
    if len(drivers) == 1:
        return drivers[0]
    metadata = _peek_array_metadata(path)
    if metadata is not None:
        return select_driver(metadata, drivers)
    return drivers[0]


def _peek_array_metadata(path: tz.PathLike) -> tx.Any:
    """Read an array's metadata straight from the store, for
    selection.

    The result is ``None`` when *path* is a group or its metadata
    cannot be read.
    """
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
    """The driver to open *path* with, selected through an async
    metadata peek.

    This function is the async twin of
    [_choose][abczarr.api.entrypoint].
    """
    if len(drivers) == 1:
        return drivers[0]
    metadata = await _apeek_array_metadata(path)
    if metadata is not None:
        return select_driver(metadata, drivers)
    return drivers[0]


async def _apeek_array_metadata(path: tz.PathLike) -> tx.Any:
    """Read an array's metadata through an async store, for
    selection.

    The result is ``None`` when *path* is a group or its metadata
    cannot be read. This function is the async twin of
    [_peek_array_metadata][abczarr.api.entrypoint].
    """
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
