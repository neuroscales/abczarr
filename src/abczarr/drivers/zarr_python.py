"""The zarr-python backend driver.

Declares what a given install of zarr-python can read and write -- coarse
capabilities and the individual codecs, chunk grids and chunk-key encodings
it has -- by asking the installed library, so selection reflects the real
build rather than a guess. The node adapters wrap a ``zarr.Array`` or
``zarr.Group`` as a [ZarrArray][abczarr.abc.sync.ZarrArray] /
[ZarrGroup][abczarr.abc.sync.ZarrGroup] so data is read and written through
the uniform surface; [abczarr.open][abczarr.api.open] opens a location and
returns whatever is there.
"""

__all__ = [
    "ZarrPythonDriver",
    "ZarrPythonNode",
    "ZarrPythonArray",
    "ZarrPythonGroup",
    "AsyncZarrPythonArray",
    "AsyncZarrPythonGroup",
]

# stdlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

# dependencies
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.features import FEATURE_KINDS, FEATURE_VERSIONS
from abczarr.abc.asynchronous import (
    AsyncZarrArray,
    AsyncZarrGroup,
    AsyncZarrNode,
)
from abczarr.abc.capabilities import Support
from abczarr.abc.sync import ZarrArray, ZarrGroup, ZarrNode
from abczarr.api.config import ArrayConfig
from abczarr.drivers._metadata import metadata_from_dict
from abczarr.drivers.base import Driver

# optionals -- the module imports without zarr; a driver with no zarr simply
# reports that it can open nothing.
try:
    import numcodecs
    import zarr
    import zarr.registry as _registry
except ImportError:  # pragma: no cover - exercised only without zarr
    numcodecs = None
    zarr = None
    _registry = None


#: Coarse capabilities a zarr-python 3.x install provides.
_V3_CAPABILITIES = {
    "async": Support.NATIVE,
    "sharding": Support.NATIVE,
    "consolidated_metadata": Support.NATIVE,
    "codecs_v2": Support.NATIVE,
    "codecs_v3": Support.NATIVE,
    "listing": Support.NATIVE,
    "writes": Support.NATIVE,
    "deletes": Support.NATIVE,
    "partial_read": Support.NATIVE,
}


def _installed_major() -> int:
    """The major version of the installed zarr, or 0 when it is absent."""
    if zarr is None:
        return 0
    try:
        return int(_dist_version("zarr").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 0


def _parse_feature(key: str) -> tx.Optional[tx.Tuple[str, str, str]]:
    """Split a feature key into (version, kind, name), or None when it is
    not a well-formed one."""
    parts = key.split(":", 2)
    if len(parts) != 3:
        return None
    version, kind, name = parts
    if version not in FEATURE_VERSIONS or kind not in FEATURE_KINDS:
        return None
    return version, kind, name


def _resolves(lookup: tx.Callable[[str], object], name: str) -> bool:
    """Whether *lookup* returns something for *name* rather than raising."""
    try:
        lookup(name)
        return True
    except Exception:
        return False


def _supports_v3_feature(kind: str, name: str) -> bool:
    """Whether zarr-python's v3 registry provides one codec / grid / etc."""
    if kind == "codec":
        return _resolves(_registry.get_codec_class, name)
    if kind == "chunk_key_encoding":
        return _resolves(_registry.get_chunk_key_encoding_class, name)
    if kind == "chunk_grid":
        # only the regular grid has a zarr-python representation
        return name == "regular"
    if kind == "data_type":
        # zarr-python handles the standard numeric data types
        return True
    return False


def _has_numcodec(name: str) -> bool:
    """Whether numcodecs provides the v1/v2 codec or filter *name*.

    A name numcodecs does not know raises ``UnknownCodecError``; a known one
    that merely needs more configuration (a filter like ``delta`` wants a
    dtype) raises something else -- and is still provided.
    """
    if numcodecs is None:
        return False
    try:
        numcodecs.get_codec({"id": name})
    except numcodecs.errors.UnknownCodecError:
        return False
    except Exception:
        return True
    return True


def _zarr_create_kwargs(config: tx.Any) -> tx.Dict[str, tx.Any]:
    """Map a resolved [ArrayConfig][abczarr.api.config.ArrayConfig] to the
    keywords ``zarr.create_array`` takes, so zarr-python creates the array and
    writes its own metadata."""
    kwargs = {
        "chunks": config.chunks,
        "fill_value": config.resolved_fill_value(),
        "compressors": config.compressor_codecs(),
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": config.resolved_separator()},
        },
    }  # type: tx.Dict[str, tx.Any]
    if config.shards is not None:
        kwargs["shards"] = config.shards
    if config.dimension_names is not None:
        kwargs["dimension_names"] = config.dimension_names
    if config.filters:
        kwargs["filters"] = [dict(f) for f in config.filters]
    # order is Zarr v2 metadata; passing it on v3 only warns (v3 order is a
    # runtime layout, not stored)
    if config.zarr_version == 2:
        kwargs["order"] = config.order
    return kwargs


class ZarrPythonDriver(Driver):
    """The zarr-python backend, as a driver.

    Reports what the installed zarr-python can do -- its coarse capabilities
    and, codec by codec, what its registry holds -- so an array is only
    routed to it when it actually has everything the array needs.
    """

    name = "zarr-python"

    def __init__(self) -> None:
        self._major = _installed_major()

    @property
    def available(self) -> bool:
        return self._major >= 3

    def _open_sync(
        self, location: tx.Any, mode: str
    ) -> "ZarrPythonNode":
        node = zarr.open(location, mode=mode)
        if isinstance(node, zarr.Group):
            return ZarrPythonGroup(node)
        return ZarrPythonArray(node)

    async def _open_async(
        self, location: tx.Any, mode: str
    ) -> AsyncZarrNode:
        # delegate to zarr-python's own async open (a coroutine); its metadata
        # read is awaited, and the AsyncArray/AsyncGroup it returns is wrapped
        # as the native async twin
        import zarr.api.asynchronous as async_api

        node = await async_api.open(store=location, mode=mode)
        return _wrap_async(node).as_async()

    def _create_sync(
        self, location: tx.Any, config: tx.Any
    ) -> "ZarrPythonNode":
        if isinstance(config, ArrayConfig):
            array = zarr.create_array(
                store=str(location),
                shape=config.shape,
                dtype=config.dtype,
                overwrite=config.overwrite,
                zarr_format=config.zarr_version,
                **_zarr_create_kwargs(config),
            )
            return ZarrPythonArray(array)
        group = zarr.open_group(
            str(location),
            mode="w" if config.overwrite else "w-",
            zarr_format=config.zarr_version,
        )
        return ZarrPythonGroup(group)

    async def _create_async(
        self, location: tx.Any, config: tx.Any
    ) -> AsyncZarrNode:
        # delegate to zarr-python's own async create (a coroutine), so it
        # writes its own metadata and its caches stay consistent; the
        # AsyncArray/AsyncGroup it returns is wrapped as the native async twin
        import zarr.api.asynchronous as async_api

        if isinstance(config, ArrayConfig):
            array = await async_api.create_array(
                store=str(location),
                shape=config.shape,
                dtype=config.dtype,
                overwrite=config.overwrite,
                zarr_format=config.zarr_version,
                **_zarr_create_kwargs(config),
            )
            return _wrap_async(array).as_async()
        group = await async_api.open_group(
            store=str(location),
            mode="w" if config.overwrite else "w-",
            zarr_format=config.zarr_version,
        )
        return _wrap_async(group).as_async()

    def capability(self, capability: str) -> Support:
        if self._major < 3:
            # zarr 2.x uses a different library API; its support lands with
            # the version adapter.
            return Support.NONE
        if capability in _V3_CAPABILITIES:
            return _V3_CAPABILITIES[capability]
        parsed = _parse_feature(capability)
        if parsed is None:
            return Support.NONE
        return self._feature_support(*parsed)

    def _feature_support(
        self, version: str, kind: str, name: str
    ) -> Support:
        """Whether the installed zarr provides one codec / grid / etc.

        zarr-python 3.x reads and writes both the v3 codec pipeline (through
        its own registry) and the v2 numcodecs model (through numcodecs).
        """
        if version == "v3":
            found = _supports_v3_feature(kind, name)
        elif version in ("v1", "v2"):
            # v1 and v2 name a numcodecs compressor or filter
            found = kind in ("codec", "filter") and _has_numcodec(name)
        else:
            found = False
        return Support.NATIVE if found else Support.NONE


# ----------------------------------------------------------------------
#   NODES
# ----------------------------------------------------------------------

#: Node capabilities a zarr-python 3.x array or group provides.
_NODE_CAPABILITIES = {
    "sharding": Support.NATIVE,
    "async": Support.NATIVE,
    "codecs_v3": Support.NATIVE,
    "consolidated_metadata": Support.NATIVE,
}


class ZarrPythonNode(ZarrNode):
    """Common base for the zarr-python array and group adapters.

    Both wrap a live zarr-python object (a ``zarr.Array`` or a ``zarr.Group``)
    and share the same metadata, attributes and version accessors -- the only
    difference between the two is the data surface each adds. The wrapped
    object is reachable as [native][abczarr.abc.sync.ZarrNode.native].
    """

    _CAPABILITIES = _NODE_CAPABILITIES

    def __init__(self, obj: tx.Any) -> None:
        super().__init__(str(obj.store_path))
        self._obj = obj
        self._native = obj

    @property
    def metadata(self) -> tx.Any:
        # read live from the wrapped zarr object, which keeps its own cache
        return metadata_from_dict(self._obj.metadata.to_dict())

    # attrs and update_attributes are inherited from ZarrNode: reads come from
    # the metadata above (zarr-python's live cache), and a write is delegated
    # to zarr-python by _write_metadata.

    def _write_metadata(self, new_metadata: tx.Any) -> None:
        """Persist new attributes by delegating to zarr-python.

        zarr-python's own `update_attributes` writes the metadata and keeps
        the object's caches consistent; going through it -- rather than
        writing the store directly -- means abczarr and zarr-python never
        disagree about the attributes. The new attributes are applied whole
        (clear, then update), so a removed key is removed, matching
        zarr-python's own attribute mapping.
        """
        obj = self._obj
        obj.metadata.attributes.clear()
        self._obj = obj.update_attributes(dict(new_metadata.attributes))
        self._native = self._obj

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        return self._obj.metadata.zarr_format


class ZarrPythonArray(ZarrPythonNode, ZarrArray):
    """A [ZarrArray][abczarr.abc.sync.ZarrArray] backed by a ``zarr.Array``.

    Wraps an open array so it reads and writes through the uniform surface.
    The underlying ``zarr.Array`` is reachable as
    [native][abczarr.abc.sync.ZarrNode.native].
    """

    @property
    def ndim(self) -> int:
        return self._obj.ndim

    @property
    def shape(self) -> tz.Shape:
        return tuple(self._obj.shape)

    @property
    def dtype(self) -> "npt.DTypeLike":
        return self._obj.dtype

    @property
    def chunks(self) -> tz.Shape:
        return tuple(self._obj.chunks)

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        shards = getattr(self._obj, "shards", None)
        return None if shards is None else tuple(shards)

    def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
        return self._obj[index]

    def __setitem__(self, index: tx.Any, value: npt.ArrayLike) -> None:
        self._obj[index] = value

    def as_async(self) -> "AsyncZarrPythonArray":
        """The native async twin, delegating to zarr-python's own
        ``AsyncArray``."""
        return AsyncZarrPythonArray(self)


class ZarrPythonGroup(ZarrPythonNode, ZarrGroup):
    """A [ZarrGroup][abczarr.abc.sync.ZarrGroup] backed by a ``zarr.Group``.

    Indexing returns a wrapped child array or group; the underlying
    ``zarr.Group`` is reachable as
    [native][abczarr.abc.sync.ZarrNode.native].
    """

    def keys(self) -> tx.Iterator[str]:
        yield from self._obj.keys()

    def __iter__(self) -> tx.Iterator[str]:
        yield from self._obj.keys()

    def __getitem__(self, key: str) -> ZarrNode:
        item = self._obj[key]
        if isinstance(item, zarr.Group):
            return ZarrPythonGroup(item)
        if isinstance(item, zarr.Array):
            return ZarrPythonArray(item)
        raise TypeError(f"unexpected child type for {key!r}: {item}")

    def __setitem__(self, key: str, value: ZarrNode) -> None:
        self._obj[key] = value.native

    def __delitem__(self, key: str) -> None:
        del self._obj[key]

    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        return ZarrPythonGroup(
            self._obj.create_group(name, overwrite=overwrite)
        )

    def _create_array(
        self, name: str, config: tx.Any
    ) -> ZarrPythonArray:
        # delegate to zarr-python, so it writes its own metadata
        array = self._obj.create_array(
            name, shape=config.shape, dtype=config.dtype,
            **_zarr_create_kwargs(config),
        )
        return ZarrPythonArray(array)

    def as_async(self) -> "AsyncZarrPythonGroup":
        """The native async twin, delegating to zarr-python's own
        ``AsyncGroup``."""
        return AsyncZarrPythonGroup(self)


# ----------------------------------------------------------------------
#   ASYNC NODES -- native, over zarr-python's AsyncArray / AsyncGroup
# ----------------------------------------------------------------------


def _wrap_async(obj: tx.Any) -> ZarrPythonNode:
    """Wrap a zarr-python ``AsyncArray`` / ``AsyncGroup`` as the abczarr sync
    node over its reconstructed sync twin, ready for ``as_async``.

    zarr-python's sync ``Array``/``Group`` are thin covers over the async
    ones, so rebuilding the sync wrapper keeps a single backend handle
    behind both colors.
    """
    if isinstance(obj, zarr.AsyncGroup):
        return ZarrPythonGroup(zarr.Group(obj))
    return ZarrPythonArray(zarr.Array(obj))


class AsyncZarrPythonArray(AsyncZarrArray):
    """The native async twin of a
    [ZarrPythonArray][abczarr.drivers.zarr_python.ZarrPythonArray].

    Reads and writes delegate straight to zarr-python's ``AsyncArray`` -- the
    real coroutine implementation the sync ``Array`` itself drives -- so its
    `"async"` capability is `Support.NATIVE`.
    """

    _async_support = Support.NATIVE

    def _async(self) -> tx.Any:
        return tx.cast(ZarrPythonArray, self._sync)._obj.async_array

    async def _awrite_metadata(self, new_metadata: tx.Any) -> None:
        """Persist new attributes by delegating to zarr-python's own async
        `update_attributes`, so zarr-python's caches stay consistent."""
        async_array = self._async()
        async_array.metadata.attributes.clear()
        await async_array.update_attributes(dict(new_metadata.attributes))

    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        return await self._async().getitem(index)

    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        await self._async().setitem(index, value)


class AsyncZarrPythonGroup(AsyncZarrGroup):
    """The native async twin of a
    [ZarrPythonGroup][abczarr.drivers.zarr_python.ZarrPythonGroup].

    Navigation and creation delegate to zarr-python's ``AsyncGroup``; its
    `"async"` capability is `Support.NATIVE`. Members come back in the async
    color.
    """

    _async_support = Support.NATIVE

    def _async(self) -> tx.Any:
        return tx.cast(ZarrPythonGroup, self._sync)._obj._async_group

    async def _awrite_metadata(self, new_metadata: tx.Any) -> None:
        """Persist new attributes by delegating to zarr-python's own async
        `update_attributes`, so zarr-python's caches stay consistent."""
        async_group = self._async()
        async_group.metadata.attributes.clear()
        await async_group.update_attributes(dict(new_metadata.attributes))

    async def getitem(self, key: str) -> AsyncZarrNode:
        item = await self._async().getitem(key)
        return _wrap_async(item).as_async()

    async def keys(self) -> tx.AsyncIterator[str]:
        async for name in self._async().keys():
            yield name

    async def _create_array(
        self, name: str, config: tx.Any
    ) -> AsyncZarrPythonArray:
        array = await self._async().create_array(
            name, shape=config.shape, dtype=config.dtype,
            **_zarr_create_kwargs(config),
        )
        return AsyncZarrPythonArray(ZarrPythonArray(zarr.Array(array)))

    async def create_group(
        self, name: str, overwrite: bool = False
    ) -> "AsyncZarrPythonGroup":
        group = await self._async().create_group(name, overwrite=overwrite)
        return AsyncZarrPythonGroup(ZarrPythonGroup(zarr.Group(group)))


