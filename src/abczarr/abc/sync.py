"""The synchronous Zarr nodes: the node base, the array, and the group.

[ZarrNode][abczarr.abc.sync.ZarrNode] is the common ancestor of
[ZarrArray][abczarr.abc.sync.ZarrArray] and
[ZarrGroup][abczarr.abc.sync.ZarrGroup]: metadata, attributes, the
Zarr format version, and the capability query all live here.

* [ZarrArray][abczarr.abc.sync.ZarrArray] is the n-dimensional node,
  read and written like a NumPy array.
* [ZarrGroup][abczarr.abc.sync.ZarrGroup] is the container node, indexed
  like a mapping; [PathGroup][abczarr.abc.sync.PathGroup] is its
  implementation for a backend with no group object of its own.
"""

__all__ = [
    "ZarrNode",
    "KNOWN_CAPABILITIES",
    "Support",
    "ZarrArray",
    "ZarrGroup",
    "PathGroup",
]

# stdlib
import json
import os
from abc import ABC, abstractmethod

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx
from bagof.paths import Path

# core
from abczarr._core import constants
from abczarr._core import typing as tz
from abczarr._core.attributes import NodeAttributes, attribute_writes
from abczarr._core.attrs import evolve
from abczarr.abc.store import PathBasedStore
from abczarr.api.config import ArrayConfig, ArrayOptions
from abczarr.errors import UnsupportedZarrOperation
from abczarr.metadata.base import (
    GroupMetadataV2,
    GroupMetadataV3,
    NodeMetadata,
    _node_at,
    _node_type_at,
)

# locals -- KNOWN_CAPABILITIES and Support are re-exported for callers that
# reach them through this module (they are listed in __all__).
from .capabilities import (  # noqa: F401
    KNOWN_CAPABILITIES,
    Support,
    SupportsCapabilities,
)

if tx.TYPE_CHECKING:
    import dask.array as da

    from .asynchronous import (
        AsyncPathGroup,
        AsyncZarrArray,
        AsyncZarrGroup,
    )


class ZarrNode(SupportsCapabilities, ABC):
    """Base class for any Zarr object: a group or an array.

    Use
    [capability][abczarr.abc.capabilities.SupportsCapabilities.capability]
    or
    [supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
    to check what a node's backend can do.
    """

    def __init__(self, store_path: tz.PathLike) -> None:
        # An os.PathLike (a pathlib.Path, say) that is not already a
        # bagof.paths Path becomes its path string, so it is wrapped below
        # rather than reaching driver code raw -- as Store.__init__ does.
        if isinstance(store_path, os.PathLike) and not isinstance(
            store_path, Path
        ):
            store_path = os.fspath(store_path)
        if isinstance(store_path, (str, bytes)):
            store_path = Path(store_path)
        self._store_path = store_path
        # The raw backend object (a zarr.Array, a tensorstore.TensorStore,
        # ...). A concrete driver sets it; it stays None where a node has no
        # single backing object (e.g. a group that is only a path).
        self._native: tx.Any = None
        # The node's metadata, loaded once and kept in memory. A node that
        # reads its metadata from a store or a metadata file caches it here
        # (the I/O is the open); a node backed by a live Zarr object reads
        # from that object instead and leaves this None.
        self._cached_metadata: tx.Optional[NodeMetadata] = None

    @property
    def store_path(self) -> os.PathLike:
        """The path to this node's location in its store."""
        return self._store_path

    @property
    def native(self) -> tx.Any:
        """The underlying backend object, or `None`.

        The escape hatch: anything the uniform surface does not name
        is still reachable through the backend object itself -- a
        `zarr.Array`, a `tensorstore.TensorStore`, and so on.
        """
        return self._native

    @property
    @abstractmethod
    def metadata(self) -> NodeMetadata:
        """This node's Zarr metadata."""
        ...

    @property
    def attrs(self) -> NodeAttributes:
        """This node's user attributes, as a live, write-through mapping.

        Reads are served from this node's metadata, the single source of
        truth. Mutations persist: `node.attrs["k"] = v` adds or replaces `k`
        and `del node.attrs["k"]` removes it, both routed through the node's
        persistence path.

        !!! example
            ```python
            node.attrs["unit"] = "micrometer"
            del node.attrs["unit"]
            ```
        """
        return NodeAttributes(self)

    def update_attributes(self, attributes: tz.JsonDict) -> "ZarrNode":
        """Add or replace several attributes at once, and persist them.

        The *attributes* are merged into this node's existing attributes --
        an existing key is overwritten, the rest are kept -- and the change is
        written through the node's persistence path. Mirrors zarr-python's
        `update_attributes`.

        !!! example
            ```python
            node.update_attributes({"unit": "micrometer", "scale": 0.5})
            ```

        Parameters
        ----------
        attributes : dict
            The attributes to add or replace. Values must be JSON-compatible.

        Returns
        -------
        ZarrNode
            This node, with the updated attributes visible on
            [attrs][abczarr.abc.sync.ZarrNode.attrs] and
            [metadata][abczarr.abc.sync.ZarrNode.metadata].
        """
        merged = dict(self.metadata.attributes)
        merged.update(attributes)
        return self._replace_attributes(merged)

    def _replace_attributes(self, attributes: tz.JsonDict) -> "ZarrNode":
        """Replace this node's attributes wholesale, and persist them."""
        new_metadata = self.metadata.update_attributes(attributes)
        self._write_metadata(new_metadata)
        return self

    def _write_metadata(self, new_metadata: NodeMetadata) -> None:
        """Persist *new_metadata*, then update the cached metadata.

        The default rewrites the node's metadata document through a
        [Store][abczarr.abc.store.Store] over the node's location, so the
        write goes through the store rather than straight to a file. A driver
        that wraps a live Zarr object overrides this to delegate to that
        object, keeping the backend's own caches consistent.
        """
        store = PathBasedStore(str(self._store_path))
        version = new_metadata.zarr_format
        existing = None  # type: tx.Optional[tx.Dict[str, tx.Any]]
        if version >= 3:
            raw = store.get(constants.Z3_JSON)
            existing = json.loads(raw) if raw else new_metadata.to_json()
        for key, value in attribute_writes(
            version, new_metadata.attributes, existing
        ):
            store.set(key, value)
        self._cache_metadata(new_metadata)

    def _cache_metadata(self, metadata: NodeMetadata) -> None:
        """Record *metadata* as this node's in-memory metadata."""
        self._cached_metadata = metadata

    @property
    @abstractmethod
    def zarr_version(self) -> tz.ZarrVersion:
        """The Zarr format version this node was written with."""
        ...


class ZarrArray(ZarrNode):
    """An n-dimensional Zarr array.

    Read and write it like a NumPy array, with NumPy-style
    selections:

    !!! example
        ```python
        array[0, :10]
        array[...] = data
        ```
    """

    @property
    @abstractmethod
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tz.Shape:
        """The shape of the array."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """The data type of the array."""
        ...

    @property
    @abstractmethod
    def chunks(self) -> tz.Shape:
        """The chunk shape of the array.

        Raises when the array's chunk grid is not regular.
        """
        ...

    @property
    @abstractmethod
    def shards(self) -> tx.Optional[tz.Shape]:
        """The shard shape of the array, or `None` if it is not
        sharded.

        Raises when the array's shard grid is not regular.
        """
        ...

    @abstractmethod
    def __getitem__(self, index: tx.Any) -> npt.ArrayLike:
        """Read data from the array at *index* (a NumPy-style
        selection)."""
        ...

    @abstractmethod
    def __setitem__(self, index: tx.Any, value: npt.ArrayLike) -> None:
        """Write *value* at *index* (a NumPy-style selection)."""
        ...

    def __array__(
        self, dtype: tx.Optional[npt.DTypeLike] = None
    ) -> npt.ArrayLike:
        """Convert this array to a NumPy array."""
        return np.asarray(self[()], dtype=dtype)

    def as_async(self) -> "AsyncZarrArray":
        """The coroutine twin of this array, over the same backend handle.

        The default runs this array's reads and writes in a bounded thread
        pool and reports `"async"` as `Support.SYNTHESIZED`. A driver whose
        backend has a native coroutine surface overrides this to return a
        `Support.NATIVE` twin that awaits the backend's own futures.

        !!! example
            ```python
            block = await array.as_async().getitem((slice(0, 8),))
            ```
        """
        from .asynchronous import ThreadedAsyncArray

        return ThreadedAsyncArray(self)

    def to_dask(
        self, chunks: tx.Union[str, tz.ShapeLike, None] = None
    ) -> "da.Array":
        """Convert this array to a Dask array.

        Parameters
        ----------
        chunks : optional
            The Dask block size. `"shards"` uses the write unit (the shard
            when sharded, otherwise the chunk); `"chunks"` uses the chunk;
            or pass an explicit block shape. The default aligns to the write
            unit, which reads a shard once rather than once per inner chunk.
            Align to `"chunks"` instead when you mean to read from the array,
            or to `"shards"` when you mean to write back into it.
        """
        import dask.array as da

        if chunks is None or chunks == "shards":
            chunks = self.shards or self.chunks
        elif chunks == "chunks":
            chunks = self.chunks
        return da.from_array(self, chunks=chunks)

    def store(
        self,
        source: npt.ArrayLike,
        *,
        lock: tx.Union[bool, str] = "auto",
    ) -> None:
        """Write *source* into this array, block by block.

        *source* is any array-like with a matching shape. A Dask array
        is written one block at a time, so a source too large to hold
        in memory never is; a plain array is written in one go. This is
        the write counterpart of
        [to_dask][abczarr.abc.sync.ZarrArray.to_dask], and it works
        for every backend (unlike `dask.array.to_zarr`, which requires
        a native `zarr.Array`).

        !!! example
            ```python
            array.store(dask_array)
            ```

        Parameters
        ----------
        source : array-like
            The data to write. Its shape must match this array's.
        lock : bool or str, optional
            Serialize concurrent block writes. The default, `"auto"`, locks
            only when the source's blocks do not line up with this array's
            write unit, since blocks that each fall on whole chunks never
            write the same chunk at once. Pass `True` or `False` to decide
            it yourself.
        """
        import dask.array as da

        darr = da.asarray(source)
        if lock == "auto":
            unit = self.shards or self.chunks
            lock = not _blocks_align_to(darr.chunks, unit)
        da.store(darr, self, lock=lock)


def _blocks_align_to(
    dask_chunks: tx.Sequence[tx.Sequence[int]], unit: tz.ShapeLike
) -> bool:
    """Whether Dask blocks fall on whole *unit*-sized chunks.

    *dask_chunks* is a Dask array's `.chunks` (per axis, the block sizes);
    *unit* is the array's write unit. True when every interior block
    boundary lands on a multiple of the unit size, so no two blocks ever
    write the same chunk and a lock is unnecessary. Conservative: anything
    it cannot prove aligned counts as not aligned.
    """
    unit = tuple(unit)
    if len(dask_chunks) != len(unit):
        return False
    for axis_blocks, size in zip(dask_chunks, unit):
        if not size:
            return False
        offset = 0
        # the final boundary is the array end; a partial last chunk there
        # is still written by a single block, so it need not align
        for block in tuple(axis_blocks)[:-1]:
            offset += block
            if offset % size:
                return False
    return True


#: The group-metadata class to write for each Zarr format version. Zarr v1
#: has no groups, so it is absent.
_GROUP_METADATA = {
    2: GroupMetadataV2,
    3: GroupMetadataV3,
}


def _resolve_array_config(
    shape: tz.ShapeLike,
    dtype: npt.DTypeLike,
    config: tx.Union[ArrayConfig, ArrayOptions, None],
    options: ArrayOptions,
    version: tz.ZarrVersion,
) -> ArrayConfig:
    """Build the resolved [ArrayConfig][abczarr.api.config.ArrayConfig] a
    `create_array` call describes.

    A config (an `ArrayConfig` or a mapping of its fields) is the base;
    *shape*, *dtype* and the per-call *options* are layered on top, with the
    array taking the group's format version. `"auto"` chunking and sharding
    are worked out, so a driver receives concrete values.
    """
    base = config if isinstance(config, ArrayConfig) else ArrayConfig(
        **dict(config or {})
    )
    merged = dict(options)
    merged.update(shape=shape, dtype=dtype, zarr_version=version)
    return evolve(base, **merged).resolve()


class ZarrGroup(ZarrNode):
    """A Zarr group: a container of arrays and subgroups.

    Index it like a mapping to reach a member by name:

    !!! example
        ```python
        group["images"]
        group["images"] = other_array
        del group["images"]
        ```
    """

    @abstractmethod
    def __getitem__(self, key: str) -> ZarrNode:
        """Get the subgroup or array named *key*."""
        ...

    @abstractmethod
    def __setitem__(self, key: str, value: ZarrNode) -> None:
        """Set the subgroup or array named *key*."""
        ...

    @abstractmethod
    def __delitem__(self, key: str) -> None:
        """Delete the subgroup or array named *key*."""
        ...

    @abstractmethod
    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        """Create or open a subgroup named *name*.

        Parameters
        ----------
        name : str
            The subgroup's name.
        overwrite : bool, optional
            Replace an existing member named *name* instead of
            raising.
        """
        ...

    def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        config: tx.Union[ArrayConfig, ArrayOptions, None] = None,
        **options: tx.Unpack[ArrayOptions],
    ) -> ZarrArray:
        """Create a new array named *name* within this group.

        Parameters
        ----------
        name : str
            The array's name.
        shape : tuple of int
            The array's shape.
        dtype : numpy dtype
            The array's data type.
        config : ArrayConfig or mapping, optional
            A reusable [ArrayConfig][abczarr.api.config.ArrayConfig], or a
            mapping of the same fields. Individual fields may also be passed as
            keyword arguments, which override the config.
        """
        resolved = _resolve_array_config(
            shape, dtype, config, options, self.zarr_version
        )
        return self._create_array(name, resolved)

    @abstractmethod
    def _create_array(self, name: str, config: ArrayConfig) -> ZarrArray:
        """Create the array named *name* from a resolved *config*, with the
        backend's own creation, so the backend writes its own metadata."""
        ...

    def as_async(self) -> "AsyncZarrGroup":
        """The coroutine twin of this group, over the same backend handle.

        The default runs this group's navigation and creation in a bounded
        thread pool and reports `"async"` as `Support.SYNTHESIZED`. A group
        that stores its members as directories (a
        [PathGroup][abczarr.abc.sync.PathGroup]) or a backend with its own
        async group overrides this to return a `Support.NATIVE` twin.
        """
        from .asynchronous import ThreadedAsyncGroup

        return ThreadedAsyncGroup(self)


class PathGroup(ZarrGroup):
    """A [ZarrGroup][abczarr.abc.sync.ZarrGroup] for a backend with no
    group object of its own.

    Some backends never construct a "group" -- TensorStore opens arrays
    only, and a bare key-value store holds nothing but keys. For those, a
    group is just a directory that carries Zarr group metadata, and
    `PathGroup` provides the whole group surface over it: reading its own
    metadata, listing its members, and navigating into subgroups and
    arrays -- using nothing but abczarr's own path and metadata layers.
    It can also create subgroups on its own, since that only means writing
    group metadata.

    A driver subclasses `PathGroup` and overrides `_open_array` (and, to
    support creating arrays too, `_create_array`) to say how a child array
    is opened and created with its own backend. Subgroups need
    no override -- they are more `PathGroup`s of the same subclass, so a
    whole hierarchy is reachable from one opened group.

    Parameters
    ----------
    store_path : PathLike
        The group's directory in its store.
    mode : str, optional
        The access mode child arrays are opened with (`"r"`, `"a"`, ...).
    """

    def __init__(self, store_path: tz.PathLike, mode: str = "r") -> None:
        super().__init__(store_path)
        self._mode = mode

    @property
    def metadata(self) -> NodeMetadata:
        # loaded from the store once, then kept in memory (the I/O is the
        # open); an attribute write updates this cache in place
        if self._cached_metadata is None:
            self._cached_metadata = NodeMetadata.from_file(self._store_path)
        return self._cached_metadata

    # attrs and update_attributes are inherited from ZarrNode: reads come from
    # the cached metadata above, and writes rewrite the metadata document
    # through the store.

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        return self.metadata.zarr_format

    def _member(
        self, store_path: tz.PathLike
    ) -> tx.Optional[tx.Tuple[tz.NodeType, tz.ZarrVersion]]:
        """The kind and version of *store_path* when it is a member of this
        group, else None.

        A member is a Zarr node written in this group's own format version.
        A node of a different version is not treated as a child, since a Zarr
        hierarchy is written in a single version.
        """
        detected = _node_at(store_path)
        if detected is None or detected[1] != self.zarr_version:
            return None
        return detected

    def keys(self) -> tx.Iterator[str]:
        """The names of this group's members (subgroups and arrays), in
        store order."""
        if not self._store_path.is_dir():
            return
        for child in self._store_path.iterdir():
            if child.is_dir() and self._member(child) is not None:
                yield child.name

    def __iter__(self) -> tx.Iterator[str]:
        return self.keys()

    def __contains__(self, name: str) -> bool:
        return self._member(self._store_path / name) is not None

    def __getitem__(self, key: str) -> ZarrNode:
        child = self._store_path / key
        detected = self._member(child)
        if detected is None:
            raise KeyError(key)
        if detected[0] == "group":
            return type(self)(child, self._mode)
        return self._open_array(child)

    def __setitem__(self, key: str, value: ZarrNode) -> None:
        raise UnsupportedZarrOperation("assign a group member")

    def __delitem__(self, key: str) -> None:
        child = self._store_path / key
        if self._member(child) is None:
            raise KeyError(key)
        child.rmdir(recursive=True)

    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        child = self._store_path / name
        if _node_type_at(child) is not None and not overwrite:
            raise FileExistsError(
                f"a member named {name!r} already exists"
            )
        version = self.zarr_version
        metadata_cls = _GROUP_METADATA.get(version)
        if metadata_cls is None:
            raise UnsupportedZarrOperation(
                f"create a group in Zarr v{version}"
            )
        child.mkdir(parents=True, exist_ok=True)
        metadata_cls(attributes={}).to_file(child)
        return type(self)(child, self._mode)

    def _create_array(self, name: str, config: ArrayConfig) -> ZarrArray:
        """Create a child array by writing its metadata, then opening it.

        This is the fallback for a backend with no native creation: write the
        config's metadata to the child directory and open it through
        `_open_array`. A backend that creates natively (zarr-python,
        TensorStore) overrides this.
        """
        child = self._store_path / name
        if _node_type_at(child) is not None:
            raise FileExistsError(
                f"a member named {name!r} already exists"
            )
        child.mkdir(parents=True, exist_ok=True)
        config.to_metadata().to_file(child)
        return self._open_array(child)

    # -- backend hook ------------------------------------------------------
    # A driver overrides this to open a child array with its backend. The
    # rest of the surface -- listing, navigation, subgroups, and writing an
    # array's metadata -- is backend-independent.

    def _open_array(self, store_path: tz.PathLike) -> ZarrArray:
        """Open the child array at *store_path*.

        `PathGroup` does not know how to open an array on its own; a driver
        overrides this method to open one with its own backend.
        """
        raise UnsupportedZarrOperation("open an array")

    def as_async(self) -> "AsyncPathGroup":
        """The coroutine twin of this group: a real async path group.

        Unlike the generic default, this twin does its own listing and
        navigation through an [AsyncStore][abczarr.abc.store.AsyncStore] over
        the group's location, and opens its array children in the async
        color -- so its `"async"` capability is `Support.NATIVE`.
        """
        from .asynchronous import AsyncPathGroup

        return AsyncPathGroup(self)
