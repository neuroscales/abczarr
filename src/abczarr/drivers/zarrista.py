"""The zarrista backend driver.

Opens a Zarr v3 array or group through `zarrista`, a small pure-Python Zarr
v3 implementation, and wraps it as a
[ZarrArray][abczarr.abc.array.ZarrArray] /
[ZarrGroup][abczarr.abc.group.ZarrGroup] so it reads and writes through the
uniform surface. A group is read straight from the store by
[PathGroup][abczarr.abc.group.PathGroup] while its arrays are
opened through zarrista.
"""

__all__ = [
    "ZarristaDriver",
    "ZarristaNode",
    "ZarristaArray",
    "ZarristaGroup",
]

# dependencies
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.dtypes import asdtype
from abczarr._core.features import FEATURE_KINDS, FEATURE_VERSIONS
from abczarr.abc.array import ZarrArray
from abczarr.abc.capabilities import Support
from abczarr.abc.group import PathGroup
from abczarr.abc.node import ZarrNode
from abczarr.drivers._metadata import metadata_from_dict
from abczarr.drivers.base import Driver

# optionals -- the module imports without zarrista; a driver with no zarrista
# reports that it can open nothing.
try:
    import zarrista
    from zarrista.store import FilesystemStore
except ImportError:  # pragma: no cover - exercised only without zarrista
    zarrista = None
    FilesystemStore = None


#: Coarse capabilities zarrista provides for the v3 format.
_V3_CAPABILITIES = {
    "sharding": Support.NATIVE,
    "codecs_v3": Support.NATIVE,
    "listing": Support.NATIVE,
    "writes": Support.NATIVE,
    "deletes": Support.NATIVE,
    "partial_read": Support.NATIVE,
}

#: The v3 codecs zarrista reads and writes, from its `zarrista.codec` module.
#: It has no runtime registry to query, so the set is declared here.
_V3_CODECS = frozenset(
    {"bytes", "zstd", "gzip", "blosc", "crc32c", "transpose", "bitround",
     "sharding_indexed"}
)

#: The chunk-key encodings zarrista supports.
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


class ZarristaNode(ZarrNode):
    """Common base for the zarrista array and group adapters.

    It marks a node as one the zarrista driver produced, so
    [open][abczarr.drivers.zarrista.ZarristaDriver.open] has one return type
    covering both. zarrista keeps no user attributes of its own, so both
    nodes read attributes from the cached metadata and persist a write by
    rewriting the metadata document through the store -- the behaviour
    inherited from [ZarrNode][abczarr.abc.node.ZarrNode].
    """


class ZarristaArray(ZarristaNode, ZarrArray):
    """A [ZarrArray][abczarr.abc.array.ZarrArray] backed by a zarrista array.

    Its shape, dtype and chunking come from the Zarr metadata; reads and
    writes go through zarrista. The underlying ``zarrista.Array`` is reachable
    as [native][abczarr.abc.node.ZarrNode.native].
    """

    _CAPABILITIES = {
        "sharding": Support.NATIVE,
        "codecs_v3": Support.NATIVE,
    }

    def __init__(self, array: tx.Any, location: tz.PathLike) -> None:
        super().__init__(str(location))
        self._array = array
        self._native = array

    @property
    def metadata(self) -> tx.Any:
        # read from the zarrista array once, then cached (an attribute write
        # updates this cache; zarrista's own copy would not see a store-routed
        # rewrite)
        if self._cached_metadata is None:
            self._cached_metadata = metadata_from_dict(self._array.metadata)
        return self._cached_metadata

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        return self.metadata.zarr_format

    @property
    def ndim(self) -> int:
        return len(self.metadata.shape)

    @property
    def shape(self) -> tz.Shape:
        return tuple(self.metadata.shape)

    @property
    def dtype(self) -> npt.DTypeLike:
        return asdtype(self.metadata.data_type)

    @property
    def chunks(self) -> tz.Shape:
        return tuple(self._grid_and_shard()[0])

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        return self._grid_and_shard()[1]

    def _grid_and_shard(self) -> tx.Tuple[tz.Shape, tx.Optional[tz.Shape]]:
        """The read chunk and the shard, worked out from the metadata.

        A sharding codec makes the chunk-grid cell the shard (the write
        unit) and its own inner shape the read chunk; without one, the
        chunk-grid cell is the chunk and there is no shard.
        """
        meta = self.metadata
        cell = tuple(meta.chunk_grid.configuration.chunk_shape)
        for codec in meta.codecs:
            if codec.name == "sharding_indexed":
                inner = tuple(codec.configuration.chunk_shape)
                return inner, cell
        return cell, None

    def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
        return self._array.retrieve_array_subset(index)

    def __setitem__(self, index: tx.Any, value: npt.ArrayLike) -> None:
        import numpy as np

        self._array.store_array_subset(index, np.asarray(value, self.dtype))


def _open_zarrista_array(location: tx.Any) -> ZarristaArray:
    """Open the v3 array at *location* through zarrista and wrap it.

    The store is pointed at the array's own directory (its ``zarr.json`` at
    the root), so a full path opens a single array.
    """
    store = FilesystemStore(str(location))
    array = zarrista.Array.open(store, path="/")
    return ZarristaArray(array, location)


class ZarristaGroup(ZarristaNode, PathGroup):
    """The group returned when the zarrista driver opens a group.

    [PathGroup][abczarr.abc.group.PathGroup] reads the group and
    lists its members straight from the store, while each child array is
    opened through zarrista. Subgroups are more `ZarristaGroup`s, so a whole
    hierarchy is reachable from one opened group.
    """

    def _open_array(self, store_path: tz.PathLike) -> ZarristaArray:
        return _open_zarrista_array(str(store_path))


class ZarristaDriver(Driver):
    """The zarrista backend, as a driver.

    Reports the v3 codecs zarrista reads and writes, and opens a v3 array or
    group through it.
    """

    name = "zarrista"

    @property
    def available(self) -> bool:
        return zarrista is not None

    def open(self, location: tx.Any, mode: str = "r") -> ZarristaNode:
        if _peek_node_type(location) == "group":
            return ZarristaGroup(location, mode)
        return _open_zarrista_array(location)

    # Creation uses the base write-then-open: abczarr writes the metadata
    # document (now schema-conformant, so zarrista's stricter reader accepts
    # it) and this driver opens it.

    def capability(self, capability: str) -> Support:
        if zarrista is None:
            return Support.NONE
        if capability in _V3_CAPABILITIES:
            return _V3_CAPABILITIES[capability]
        parsed = _parse_feature(capability)
        if parsed is None:
            return Support.NONE
        version, kind, name = parsed
        if version != "v3":
            return Support.NONE
        return Support.NATIVE if _supports_v3_feature(kind, name) else (
            Support.NONE
        )


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
