"""The TensorStore backend driver.

Opens a Zarr array through Google's TensorStore -- a fast, C++ backed reader
and writer -- and wraps it as a [ZarrArray][abczarr.abc.sync.ZarrArray] so it
reads and writes through the uniform surface. A v3 array goes through
TensorStore's ``zarr3`` driver and a v2 array through its native ``zarr``
driver, so both versions read and write. [abczarr.open][abczarr.api.open]
opens an array through it. TensorStore has no group object, so a group is read
straight from the store by [PathGroup][abczarr.abc.sync.PathGroup] while its
arrays are opened through TensorStore.
"""

__all__ = [
    "TensorStoreDriver",
    "TensorStoreNode",
    "TensorStoreArray",
    "TensorStoreGroup",
    "AsyncTensorStoreArray",
]

# stdlib
import json

# dependencies
import numpy.typing as npt
import typing_extensions as tx
from bagof.paths import Path

# core
from abczarr._core import constants
from abczarr._core import typing as tz
from abczarr._core.features import FEATURE_KINDS, FEATURE_VERSIONS
from abczarr.abc.asynchronous import AsyncZarrArray, AsyncZarrNode
from abczarr.abc.capabilities import Support
from abczarr.abc.store import AsyncPathBasedStore, PathBasedStore
from abczarr.abc.sync import PathGroup, ZarrArray, ZarrNode
from abczarr.api._config import ArrayConfig
from abczarr.drivers._metadata import metadata_from_json
from abczarr.drivers.base import Driver
from abczarr.metadata.base import ArrayMetadata, NodeMetadata, _node_at

# optionals -- the module imports without tensorstore; a driver with no
# tensorstore reports that it can open nothing.
try:
    import tensorstore as ts
except ImportError:  # pragma: no cover - exercised only without tensorstore
    ts = None


#: Coarse capabilities TensorStore provides for the v3 format.
_V3_CAPABILITIES = {
    "async": Support.NATIVE,
    "sharding": Support.NATIVE,
    "codecs_v3": Support.NATIVE,
    "listing": Support.NATIVE,
    "writes": Support.NATIVE,
    "deletes": Support.NATIVE,
    "partial_read": Support.NATIVE,
}

#: The v3 codecs TensorStore reads and writes. TensorStore has no runtime
#: registry to query, and its support is stable per release, so the set is
#: declared here.
_V3_CODECS = frozenset(
    {"bytes", "zstd", "gzip", "blosc", "crc32c", "transpose",
     "sharding_indexed"}
)

#: The chunk-key encodings TensorStore supports.
_V3_CHUNK_KEY_ENCODINGS = frozenset({"default", "v2"})


def _parse_feature(key: str) -> tx.Optional[tx.Tuple[str, str, str]]:
    """Split a feature key into (version, kind, name), or None if malformed."""
    parts = key.split(":", 2)
    if len(parts) != 3:
        return None
    version, kind, name = parts
    if version not in FEATURE_VERSIONS or kind not in FEATURE_KINDS:
        return None
    return version, kind, name


def _kvstore_spec(location: tx.Any) -> tx.Any:
    """The TensorStore kvstore spec for *location*.

    A kvstore spec (a dict or URL) is used as it is; a local path becomes the
    file kvstore.
    """
    if isinstance(location, (dict, str)) and not _looks_like_path(location):
        return location
    return {"driver": "file", "path": str(location)}


def _looks_like_path(location: tx.Any) -> bool:
    return isinstance(location, str) and "://" not in location


def _ts_array_driver(version: tx.Any) -> str:
    """The TensorStore driver name for a Zarr array of *version*.

    TensorStore reads a v3 array through its ``zarr3`` driver and a v2 array
    through its native ``zarr`` driver.
    """
    return "zarr" if version == 2 else "zarr3"


def _ts_metadata(metadata: "ArrayMetadata") -> tx.Any:
    """The metadata document TensorStore's driver accepts for *metadata*.

    A v3 array's document is TensorStore's ``zarr.json`` as it is. A v2
    array's ``.zarray`` carries neither ``node_type`` nor the user attributes
    (those live in ``.zattrs``), and TensorStore's ``zarr`` driver rejects
    both, so they are dropped here; the attributes are persisted separately.
    """
    doc = metadata.to_json()
    if metadata.zarr_format == 2:
        doc.pop("node_type", None)
        doc.pop("attributes", None)
    return doc


def _v2_attributes_payload(
    metadata: "ArrayMetadata",
) -> tx.Optional[bytes]:
    """The ``.zattrs`` bytes for a v2 array's user attributes, or None.

    TensorStore's ``zarr`` driver writes only ``.zarray``, so a v2 array's
    attributes are persisted through the store separately. Returns None when
    there is nothing to write (a v3 array, or no attributes).
    """
    if metadata.zarr_format != 2 or not metadata.attributes:
        return None
    return json.dumps(dict(metadata.attributes)).encode("utf-8")


class TensorStoreNode(ZarrNode):
    """Common base for the TensorStore array and group adapters.

    It marks a node as one the TensorStore driver produced, so
    [open][abczarr.drivers.tensorstore.TensorStoreDriver.open] has one return
    type covering both. TensorStore keeps no user attributes of its own, so
    both nodes read attributes from the cached metadata and persist a write by
    rewriting the metadata document through the store -- the behaviour
    inherited from [ZarrNode][abczarr.abc.sync.ZarrNode] -- and there is
    nothing driver-wide to override here.
    """


class TensorStoreArray(TensorStoreNode, ZarrArray):
    """A [ZarrArray][abczarr.abc.sync.ZarrArray] backed by a TensorStore.

    Wraps an open ``tensorstore.TensorStore`` so it reads and writes through
    the uniform surface. TensorStore's richer indexing (its ``oindex`` /
    ``vindex`` and index transforms) stays reachable through
    [native][abczarr.abc.sync.ZarrNode.native].
    """

    _CAPABILITIES = {
        "sharding": Support.NATIVE,
        "async": Support.NATIVE,
        "codecs_v3": Support.NATIVE,
        "partial_read": Support.NATIVE,
    }

    def __init__(self, array: tx.Any) -> None:
        super().__init__(str(array.kvstore.url) if array.kvstore else "")
        self._array = array
        self._native = array

    @property
    def metadata(self) -> tx.Any:
        # read from TensorStore's spec once, then cached (an attribute write
        # updates this cache; TensorStore's own spec would not see a
        # store-routed rewrite)
        if self._cached_metadata is None:
            meta = metadata_from_json(
                self._array.spec().to_json()["metadata"]
            )
            # TensorStore's zarr (v2) driver keeps no user attributes in its
            # spec -- they live in .zattrs -- so read them from the store.
            if meta.zarr_format == 2:
                meta = self._with_v2_attributes(meta)
            self._cached_metadata = meta
        return self._cached_metadata

    def _with_v2_attributes(self, metadata: tx.Any) -> tx.Any:
        """Merge a v2 array's ``.zattrs`` user attributes into *metadata*."""
        try:
            raw = PathBasedStore(str(self._store_path)).get(
                constants.Z2ATTRS_JSON
            )
        except Exception:
            return metadata
        if not raw:
            return metadata
        try:
            attributes = json.loads(raw)
        except (ValueError, TypeError):
            return metadata
        return metadata.update_attributes(attributes)

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        return self.metadata.zarr_format

    @property
    def ndim(self) -> int:
        return self._array.rank

    @property
    def shape(self) -> tz.Shape:
        return tuple(self._array.shape)

    @property
    def dtype(self) -> "npt.DTypeLike":
        return self._array.dtype.numpy_dtype

    @property
    def chunks(self) -> tz.Shape:
        return tuple(self._array.chunk_layout.read_chunk.shape)

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        read = tuple(self._array.chunk_layout.read_chunk.shape)
        write = tuple(self._array.chunk_layout.write_chunk.shape)
        # a shard is the write unit when it groups several read chunks
        return write if write != read else None

    def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
        return self._array[index].read().result()

    def __setitem__(self, index: tx.Any, value: npt.ArrayLike) -> None:
        self._array[index].write(value).result()

    def as_async(self) -> "AsyncTensorStoreArray":
        """The native coroutine twin: reads and writes await TensorStore's
        own futures rather than blocking on `.result()`."""
        return AsyncTensorStoreArray(self)


class AsyncTensorStoreArray(AsyncZarrArray):
    """The native async twin of a
    [TensorStoreArray][abczarr.drivers.tensorstore.TensorStoreArray].

    Every TensorStore op returns an awaitable future, so `getitem` and
    `setitem` await it directly -- the fast path, never `.result()`. Its
    `"async"` capability is `Support.NATIVE`.
    """

    _async_support = Support.NATIVE

    async def getitem(self, index: tx.Any) -> npt.ArrayLike:
        return await self._array()[index].read()

    async def setitem(self, index: tx.Any, value: npt.ArrayLike) -> None:
        await self._array()[index].write(value)

    def _array(self) -> tx.Any:
        """The wrapped ``tensorstore.TensorStore`` handle."""
        return tx.cast(TensorStoreArray, self._sync)._array


def _open_ts_array(
    location: tx.Any, mode: str, version: tx.Any = 3
) -> TensorStoreArray:
    """Open the array at *location* through TensorStore and wrap it.

    *version* selects TensorStore's driver -- ``zarr3`` for a v3 array,
    ``zarr`` for a v2 array.
    """
    spec = {
        "driver": _ts_array_driver(version),
        "kvstore": _kvstore_spec(location),
    }
    array = ts.open(
        spec, open=True, read=True, write=mode not in ("r", "read")
    ).result()
    return TensorStoreArray(array)


async def _aopen_ts_array(
    location: tx.Any, mode: str, version: tx.Any = 3
) -> "AsyncTensorStoreArray":
    """Open the array at *location* through TensorStore asynchronously.

    Awaits TensorStore's own open future rather than blocking on
    ``.result()``, then wraps the array as its native async twin. *version*
    selects TensorStore's driver, as for the synchronous open.
    """
    spec = {
        "driver": _ts_array_driver(version),
        "kvstore": _kvstore_spec(location),
    }
    array = await ts.open(
        spec, open=True, read=True, write=mode not in ("r", "read")
    )
    return TensorStoreArray(array).as_async()


def _create_ts_array(
    location: tx.Any, metadata: ArrayMetadata, *, overwrite: bool
) -> TensorStoreArray:
    """Create the array *metadata* describes at *location*.

    TensorStore creates from the metadata document, filling in each codec's
    defaults and validating it, which a bare write of the metadata would not.
    A v3 array goes through the ``zarr3`` driver, a v2 array through the
    ``zarr`` driver; a v2 array's user attributes are written to ``.zattrs``
    afterwards, since TensorStore's ``zarr`` driver writes only ``.zarray``.
    """
    if _node_at(Path(str(location))) is not None and not overwrite:
        raise FileExistsError(f"a node already exists at {location}")
    spec = {
        "driver": _ts_array_driver(metadata.zarr_format),
        "kvstore": _kvstore_spec(location),
        "metadata": _ts_metadata(metadata),
    }
    array = ts.open(spec, create=True, delete_existing=overwrite).result()
    attrs = _v2_attributes_payload(metadata)
    if attrs is not None:
        PathBasedStore(str(location)).set(constants.Z2ATTRS_JSON, attrs)
    return TensorStoreArray(array)


async def _acreate_ts_array(
    location: tx.Any, metadata: ArrayMetadata, *, overwrite: bool
) -> "AsyncTensorStoreArray":
    """Create the array *metadata* describes at *location* asynchronously.

    Awaits TensorStore's own create future rather than blocking on
    ``.result()``, then wraps the array as its native async twin. Driver
    selection and v2 attribute persistence mirror the synchronous create.
    """
    if _node_at(Path(str(location))) is not None and not overwrite:
        raise FileExistsError(f"a node already exists at {location}")
    spec = {
        "driver": _ts_array_driver(metadata.zarr_format),
        "kvstore": _kvstore_spec(location),
        "metadata": _ts_metadata(metadata),
    }
    array = await ts.open(spec, create=True, delete_existing=overwrite)
    attrs = _v2_attributes_payload(metadata)
    if attrs is not None:
        await AsyncPathBasedStore(str(location)).set(
            constants.Z2ATTRS_JSON, attrs
        )
    return TensorStoreArray(array).as_async()


class TensorStoreGroup(TensorStoreNode, PathGroup):
    """The group returned when the TensorStore driver opens a group.

    TensorStore has no group object of its own, so
    [PathGroup][abczarr.abc.sync.PathGroup] reads the group itself -- its
    metadata and the names of its members -- straight from the store, while
    each child array is opened through TensorStore. Subgroups are more
    `TensorStoreGroup`s, so a whole hierarchy is reachable from one opened
    group.
    """

    def _open_array(self, store_path: tz.PathLike) -> TensorStoreArray:
        # a child array is written in the group's own version
        return _open_ts_array(str(store_path), self._mode, self.zarr_version)

    def _create_array(
        self, name: str, config: ArrayConfig
    ) -> TensorStoreArray:
        # tensorstore validates and fills a codec's defaults on create, so we
        # hand it the config's metadata document
        return _create_ts_array(
            str(self._store_path / name), config.to_metadata(), overwrite=False
        )


class TensorStoreDriver(Driver):
    """The TensorStore backend, as a driver.

    Reports the v3 codecs TensorStore reads and writes, and opens a Zarr
    array through it -- a v3 array through the ``zarr3`` driver, a v2 array
    through the native ``zarr`` driver. A group has no TensorStore object of
    its own, so it is read straight from the store by the path group.
    """

    name = "tensorstore"

    @property
    def available(self) -> bool:
        return ts is not None

    def _open_sync(self, location: tx.Any, mode: str) -> TensorStoreNode:
        peeked = _peek_node(location)
        if peeked is not None:
            node_type, version = peeked
            # TensorStore has no group object, so a group of any version is
            # read straight from the store by the path group.
            if node_type == "group":
                return TensorStoreGroup(location, mode)
            # a v2 array opens through the ``zarr`` driver, a v3 array through
            # ``zarr3`` -- selected from the version the peek found.
            return _open_ts_array(location, mode, version)
        return _open_ts_array(location, mode)

    async def _open_async(
        self, location: tx.Any, mode: str
    ) -> AsyncZarrNode:
        # TensorStore has no group object, so a group opens as the async path
        # group (which lists and navigates through an async store); an array
        # awaits TensorStore's own open future.
        peeked = await _apeek_node(location)
        if peeked is not None:
            node_type, version = peeked
            if node_type == "group":
                return TensorStoreGroup(location, mode).as_async()
            return await _aopen_ts_array(location, mode, version)
        return await _aopen_ts_array(location, mode)

    def _create_from_metadata_sync(
        self, location: tx.Any, metadata: NodeMetadata,
        *, overwrite: bool = False,
    ) -> ZarrNode:
        if isinstance(metadata, ArrayMetadata):
            return _create_ts_array(location, metadata, overwrite=overwrite)
        # a group is just a directory; the base's write-then-open handles it
        return super()._create_from_metadata_sync(
            location, metadata, overwrite=overwrite
        )

    async def _create_async(
        self, location: tx.Any, config: "tx.Any"
    ) -> AsyncZarrNode:
        # lower the config to metadata (cheap, synchronous) then await
        # TensorStore's own create future
        return await self._create_from_metadata_async(
            location, self._config_metadata(config),
            overwrite=config.overwrite,
        )

    async def _create_from_metadata_async(
        self, location: tx.Any, metadata: NodeMetadata,
        *, overwrite: bool = False,
    ) -> AsyncZarrNode:
        if isinstance(metadata, ArrayMetadata):
            return await _acreate_ts_array(
                location, metadata, overwrite=overwrite
            )
        # a group is just a directory; the base's thread-bridged create handles
        # it, opening the async path group over the async store
        return await super()._create_from_metadata_async(
            location, metadata, overwrite=overwrite
        )

    def capability(self, capability: str) -> Support:
        if ts is None:
            return Support.NONE
        if capability in _V3_CAPABILITIES:
            return _V3_CAPABILITIES[capability]
        parsed = _parse_feature(capability)
        if parsed is None:
            return Support.NONE
        version, kind, name = parsed
        if version != "v3":
            return Support.NONE
        found = _supports_v3_feature(kind, name)
        return Support.NATIVE if found else Support.NONE


def _supports_v3_feature(kind: str, name: str) -> bool:
    if kind == "codec":
        return name in _V3_CODECS
    if kind == "chunk_key_encoding":
        return name in _V3_CHUNK_KEY_ENCODINGS
    if kind == "chunk_grid":
        return name == "regular"
    if kind == "data_type":
        return True
    return False


def _v3_node(raw: tx.Any) -> tx.Optional[tx.Tuple[str, int]]:
    """The ``(node_type, 3)`` pair a v3 ``zarr.json`` document names, or None.

    A ``zarr.json`` without a valid ``node_type`` is treated as no node, since
    the format requires one.
    """
    try:
        node_type = json.loads(raw).get("node_type")
    except (ValueError, TypeError):
        return None
    if node_type in ("array", "group"):
        return node_type, 3
    return None


def _peek_node(location: tx.Any) -> tx.Optional[tx.Tuple[str, int]]:
    """The node kind and Zarr version recorded at *location*, or None.

    Detects both a v3 ``zarr.json`` (through its ``node_type`` field) and a v2
    node (through which of ``.zgroup`` and ``.zarray`` is present), so a v2
    group or array is recognised as well as a v3 one. Read through a
    [PathBasedStore][abczarr.abc.store.PathBasedStore], so every scheme
    bagof.paths understands is inspected the same way -- a local path, an
    fsspec URL (``memory://``), or a cloud one (``s3://``). A raw kvstore dict
    spec is not a location to peek, so it returns None.

    Returns
    -------
    tuple of (str, int) or None
        A ``("array" | "group", zarr_format)`` pair for a node, else None.
    """
    if isinstance(location, dict):  # a kvstore spec, not a location
        return None
    try:
        store = PathBasedStore(location)
        raw = store.get(constants.Z3_JSON)
        if raw is not None:
            return _v3_node(raw)
        if store.get(constants.Z2GROUP_JSON) is not None:
            return "group", 2
        if store.get(constants.Z2ARRAY_JSON) is not None:
            return "array", 2
    except Exception:
        return None
    return None


async def _apeek_node(location: tx.Any) -> tx.Optional[tx.Tuple[str, int]]:
    """The node kind and Zarr version at *location*, read through an async
    store, or None -- the async twin of
    [_peek_node][abczarr.drivers.tensorstore].

    Read through an
    [AsyncPathBasedStore][abczarr.abc.store.AsyncPathBasedStore], so a URL
    (``memory://``, ``s3://``) is inspected exactly like a local path.
    """
    if isinstance(location, dict):  # a kvstore spec, not a location
        return None
    try:
        store = AsyncPathBasedStore(location)
        raw = await store.get(constants.Z3_JSON)
        if raw is not None:
            return _v3_node(raw)
        if await store.get(constants.Z2GROUP_JSON) is not None:
            return "group", 2
        if await store.get(constants.Z2ARRAY_JSON) is not None:
            return "array", 2
    except Exception:
        return None
    return None
