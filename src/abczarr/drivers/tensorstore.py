"""The TensorStore backend driver.

Opens a Zarr v3 array through Google's TensorStore -- a fast, C++ backed
reader and writer -- and wraps it as a [ZarrArray][abczarr.abc.array.ZarrArray]
so it reads and writes through the uniform surface.
[abczarr.open][abczarr.api.open] opens an array through it. TensorStore has
no group object, so a group is read straight from the store by
[PathGroup][abczarr.abc.group.PathGroup] while its arrays are opened through
TensorStore.
"""

__all__ = [
    "TensorStoreDriver",
    "TensorStoreNode",
    "TensorStoreArray",
    "TensorStoreGroup",
]

# dependencies
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.features import FEATURE_KINDS, FEATURE_VERSIONS
from abczarr.abc.array import ZarrArray
from abczarr.abc.capabilities import Support
from abczarr.abc.group import PathGroup
from abczarr.abc.node import ZarrNode
from abczarr.config import ArrayConfig
from abczarr.drivers._metadata import metadata_from_dict
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


class TensorStoreNode(ZarrNode):
    """Common base for the TensorStore array and group adapters.

    It marks a node as one the TensorStore driver produced, so
    [open][abczarr.drivers.tensorstore.TensorStoreDriver.open] has one return
    type covering both. TensorStore keeps no user attributes of its own, so
    both nodes read and write attributes through the metadata file -- the
    write-through mapping inherited from
    [ZarrNode][abczarr.abc.node.ZarrNode] -- and there is nothing driver-wide
    to override here.
    """


class TensorStoreArray(TensorStoreNode, ZarrArray):
    """A [ZarrArray][abczarr.abc.array.ZarrArray] backed by a TensorStore.

    Wraps an open ``tensorstore.TensorStore`` so it reads and writes through
    the uniform surface. TensorStore's richer indexing (its ``oindex`` /
    ``vindex`` and index transforms) stays reachable through
    [native][abczarr.abc.node.ZarrNode.native].
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
        return metadata_from_dict(self._array.spec().to_json()["metadata"])

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        return self._array.spec().to_json()["metadata"]["zarr_format"]

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


def _open_ts_array(location: tx.Any, mode: str) -> TensorStoreArray:
    """Open the v3 array at *location* through TensorStore and wrap it."""
    spec = {"driver": "zarr3", "kvstore": _kvstore_spec(location)}
    array = ts.open(
        spec, open=True, read=True, write=mode not in ("r", "read")
    ).result()
    return TensorStoreArray(array)


def _create_ts_array(
    location: tx.Any, metadata: ArrayMetadata, *, overwrite: bool
) -> TensorStoreArray:
    """Create the v3 array *metadata* describes at *location*.

    TensorStore creates from the metadata document, filling in each codec's
    defaults and validating it, which a bare write of the metadata would not.
    """
    from bagof.paths import Path

    if _node_at(Path(str(location))) is not None and not overwrite:
        raise FileExistsError(f"a node already exists at {location}")
    spec = {
        "driver": "zarr3",
        "kvstore": _kvstore_spec(location),
        "metadata": metadata.to_dict(),
    }
    array = ts.open(spec, create=True, delete_existing=overwrite).result()
    return TensorStoreArray(array)


class TensorStoreGroup(TensorStoreNode, PathGroup):
    """The group returned when the TensorStore driver opens a group.

    TensorStore has no group object of its own, so
    [PathGroup][abczarr.abc.group.PathGroup] reads the group itself -- its
    metadata and the names of its members -- straight from the store, while
    each child array is opened through TensorStore. Subgroups are more
    `TensorStoreGroup`s, so a whole hierarchy is reachable from one opened
    group.
    """

    def _open_array(self, store_path: tz.PathLike) -> TensorStoreArray:
        return _open_ts_array(str(store_path), self._mode)

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

    Reports the v3 codecs TensorStore reads and writes, and opens a v3 array
    through it. A group has no TensorStore representation, so opening one is
    left to another driver.
    """

    name = "tensorstore"

    @property
    def available(self) -> bool:
        return ts is not None

    def open(self, location: tx.Any, mode: str = "r") -> TensorStoreNode:
        if _peek_node_type(location) == "group":
            return TensorStoreGroup(location, mode)
        return _open_ts_array(location, mode)

    def create_from_metadata(
        self, location: tx.Any, metadata: NodeMetadata,
        *, overwrite: bool = False,
    ) -> ZarrNode:
        if isinstance(metadata, ArrayMetadata):
            return _create_ts_array(location, metadata, overwrite=overwrite)
        # a group is just a directory; the base's write-then-open handles it
        return super().create_from_metadata(
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


def _peek_node_type(location: tx.Any) -> tx.Optional[str]:
    """The node type recorded at *location*'s ``zarr.json``, or None."""
    import json

    from abczarr.abc.store import PathBasedStore

    if not isinstance(location, str) or "://" in location:
        return None
    try:
        raw = PathBasedStore(location).get("zarr.json")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw).get("node_type")
    except (ValueError, TypeError):
        return None


