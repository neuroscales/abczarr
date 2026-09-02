"""The TensorStore backend driver.

Opens a Zarr v3 array through Google's TensorStore -- a fast, C++ backed
reader and writer -- and wraps it as a [ZarrArray][abczarr.abc.array.ZarrArray]
so it reads and writes through the uniform surface.
[abczarr.open][abczarr.api.open] opens an array through it. TensorStore has
no group object, so a group is opened through another driver; this one
handles arrays.
"""

__all__ = [
    "TensorStoreDriver",
    "TensorStoreArray",
]

# dependencies
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.features import FEATURE_KINDS, FEATURE_VERSIONS
from abczarr.abc.array import ZarrArray
from abczarr.abc.capabilities import Support
from abczarr.abc.errors import UnsupportedZarrOperation
from abczarr.drivers._metadata import metadata_from_dict
from abczarr.drivers.base import Driver

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


class TensorStoreArray(ZarrArray):
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
    def attrs(self) -> tz.Attributes:
        attributes = self._array.spec().to_json()["metadata"].get(
            "attributes", {}
        )
        return dict(attributes)

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

    def open(self, location: tx.Any, mode: str = "r") -> TensorStoreArray:
        if _peek_node_type(location) == "group":
            raise UnsupportedZarrOperation(
                "open a group (TensorStore opens arrays)", "tensorstore"
            )
        spec = {"driver": "zarr3", "kvstore": _kvstore_spec(location)}
        array = ts.open(
            spec, open=True, read=True, write=mode not in ("r", "read")
        ).result()
        return TensorStoreArray(array)

    def support(self, capability: str) -> Support:
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

    from abczarr.abc.store import PathStore

    if not isinstance(location, str) or "://" in location:
        return None
    try:
        raw = PathStore(location).get("zarr.json")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw).get("node_type")
    except (ValueError, TypeError):
        return None


