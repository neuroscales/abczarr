"""The synchronous Zarr node classes: the shared base, the array, and
the group.

[ZarrNode][abczarr.abc.sync.ZarrNode] is the base class both
[ZarrArray][abczarr.abc.sync.ZarrArray] and
[ZarrGroup][abczarr.abc.sync.ZarrGroup] inherit from. It defines the
metadata, the user attributes, the Zarr format version, and the
capability query that every node shares.

* [ZarrArray][abczarr.abc.sync.ZarrArray] is the n-dimensional node.
  It is read and written like a NumPy array.
* [ZarrGroup][abczarr.abc.sync.ZarrGroup] is the container node. It
  is indexed like a mapping, with subgroups and arrays as its
  members. [PathGroup][abczarr.abc.sync.PathGroup] implements a
  group for a backend that has no group object of its own.
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
from abczarr.ome.node import (
    delete_ome,
    read_ome,
    update_ome,
    write_ome,
)

# locals: KNOWN_CAPABILITIES and Support are re-exported for callers that
# reach them through this module (they are listed in __all__).
from .capabilities import (  # noqa: F401
    KNOWN_CAPABILITIES,
    Support,
    SupportsCapabilities,
)

if tx.TYPE_CHECKING:
    import dask.array as da

    from abczarr.ome.base import OME

    from .asynchronous import (
        AsyncPathGroup,
        AsyncZarrArray,
        AsyncZarrGroup,
    )


class ZarrNode(SupportsCapabilities, ABC):
    """The base class for a Zarr array or a Zarr group.

    A `ZarrNode` carries what every Zarr object shares: its location,
    its metadata, its user attributes, its OME-Zarr metadata, and its
    format version.
    [ZarrArray][abczarr.abc.sync.ZarrArray] and
    [ZarrGroup][abczarr.abc.sync.ZarrGroup] add the data-specific
    surface on top of it.

    A node's backend determines which capabilities it has. Use
    [capability][abczarr.abc.capabilities.SupportsCapabilities.capability]
    to find out how a given capability is provided, and
    [supports][abczarr.abc.capabilities.SupportsCapabilities.supports]
    to check whether it is provided at all.
    """

    def __init__(self, store_path: tz.PathLike) -> None:
        # Wrap a raw path in a bagof.paths Path, which itself converts an
        # os.PathLike such as a pathlib.Path, so it does not reach driver
        # code raw. An already-wrapped Path (or StorePath) is left as is. A
        # node always has a path, so nothing but a Path is left unwrapped.
        if not isinstance(store_path, Path):
            store_path = Path(store_path)
        self._store_path = store_path
        # The raw backend object, such as a zarr.Array or a
        # tensorstore.TensorStore. A concrete driver sets it. It stays None
        # where a node has no single backing object, such as a group that
        # is only a path.
        self._native: tx.Any = None
        # The node's metadata, loaded once and kept in memory. A node that
        # reads its metadata from a store or a metadata file caches it
        # here, since the I/O is the open. A node backed by a live Zarr
        # object reads from that object instead and leaves this None.
        self._cached_metadata: tx.Optional[NodeMetadata] = None

    @property
    def store_path(self) -> os.PathLike:
        """The path to this node's location in its store."""
        return self._store_path

    @property
    def native(self) -> tx.Any:
        """The underlying backend object, or `None` when the node
        has none.

        This property is the escape hatch for anything the uniform
        node surface does not expose. A `zarr.Array`, a
        `tensorstore.TensorStore`, or another backend's own object is
        reachable here, with its full native API. A node backed by
        nothing but a path, such as a
        [PathGroup][abczarr.abc.sync.PathGroup], has no such object
        and returns `None`.
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

        Reads come from this node's metadata, which is the single
        source of truth for what the mapping holds. A write persists
        immediately: `node.attrs["unit"] = "micrometer"` adds or
        replaces the key, and `del node.attrs["unit"]` removes it.
        Both operations are written through the node's own
        persistence path, so the change reaches the store before the
        call returns.

        !!! example
            ```python
            node.attrs["unit"] = "micrometer"
            del node.attrs["unit"]
            ```
        """
        return NodeAttributes(self)

    @property
    def ome(self) -> "tx.Optional[OME]":
        """This node's OME-Zarr metadata as a typed object, read and
        write.

        Reading parses the node's OME-NGFF metadata and returns the
        matching version's [OME][abczarr.ome.base.OME] object, or
        `None` when the node carries no OME metadata. Assigning a
        value serializes and persists it, write-through in the same
        way as [attrs][abczarr.abc.sync.ZarrNode.attrs]. The assigned
        value may be a typed [OME][abczarr.ome.base.OME] object or a
        plain JSON-style mapping that carries a ``version`` key.
        Deleting the attribute, with `del node.ome`, removes the
        metadata.

        Both directions understand every NGFF envelope. From version
        0.5 on, the metadata is nested under a single ``"ome"``
        attribute. Before 0.5, its fields are written directly at the
        top level of the node's attributes. Neither direction touches
        any attribute that is not part of the OME metadata.

        !!! example
            ```python
            from abczarr.ome import v0_5

            node.ome = v0_5.OME.from_json(
                {"version": "0.5", "multiscales": [...]}
            )
            image = node.ome           # a typed OME object
            del node.ome               # clear it
            ```
        """
        return read_ome(self)

    @ome.setter
    def ome(self, value: "tx.Union[OME, tz.JsonDict]") -> None:

        write_ome(self, value)

    @ome.deleter
    def ome(self) -> None:

        delete_ome(self)

    def update_ome(self, ome: "tx.Union[OME, tz.JsonDict]") -> "ZarrNode":
        """Merge OME metadata into this node's existing OME metadata,
        and persist the result.

        This method is the OME-metadata counterpart of
        [update_attributes][abczarr.abc.sync.ZarrNode.update_attributes].
        A top-level key present in *ome*, such as ``version``,
        ``multiscales`` or ``omero``, replaces the value already
        stored under that key. A key the node already carries that
        *ome* does not name is preserved. When the node has no OME
        metadata yet and the merged result still names no version,
        the metadata is written with the latest released OME version.

        The merge is shallow. A nested structure, such as a
        multiscale or a plate definition, is replaced as a whole
        rather than merged recursively. For a structured edit, read
        [ome][abczarr.abc.sync.ZarrNode.ome], change the typed object,
        and assign the result back.

        !!! example
            ```python
            node.update_ome({"omero": {"channels": [...]}})
            ```

        Parameters
        ----------
        ome : OME or dict
            The metadata whose top-level keys are merged in.

        Returns
        -------
        ZarrNode
            This node, with the merged metadata visible on
            [ome][abczarr.abc.sync.ZarrNode.ome].
        """

        update_ome(self, ome)
        return self

    def update_attributes(self, attributes: tz.JsonDict) -> "ZarrNode":
        """Add or replace several attributes at once, and persist the
        change.

        The keys in *attributes* are merged into this node's existing
        attributes. An existing key is overwritten with the new
        value, and every other key is kept unchanged. The merged
        result is written through the node's persistence path before
        this method returns. The behavior mirrors zarr-python's own
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

    A `ZarrArray` is read and written like a NumPy array. Indexing
    accepts NumPy-style selections, including integers, slices, and
    ellipses.

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

        This property raises when the array's chunk grid is not
        regular.
        """
        ...

    @property
    @abstractmethod
    def shards(self) -> tx.Optional[tz.Shape]:
        """The shard shape of the array, or `None` if it is not
        sharded.

        This property raises when the array's shard grid is not
        regular.
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
        self,
        dtype: tx.Optional[npt.DTypeLike] = None,
        copy: tx.Optional[bool] = None,
    ) -> npt.ArrayLike:
        """Convert this array to a NumPy array.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            The dtype of the returned array.
        copy : bool, optional
            Whether to copy the data. NumPy 2's array protocol passes
            this argument. Reading the array always materializes a
            fresh array in memory, so a view without copying is never
            possible. ``copy=False`` is refused for that reason. Both
            ``copy=True`` and ``copy=None`` return the freshly read
            array.

        Raises
        ------
        ValueError
            If *copy* is `False`.
        """
        if copy is False:
            raise ValueError(
                "cannot read this array without making a copy: "
                "pass copy=True or copy=None"
            )
        return np.asarray(self[()], dtype=dtype)

    def as_async(self) -> "AsyncZarrArray":
        """The coroutine twin of this array, over the same backend
        handle.

        By default, the returned twin runs this array's reads and
        writes in a bounded thread pool, and its `"async"` capability
        reports `Support.SYNTHESIZED`. A driver whose backend has its
        own coroutine surface returns a `Support.NATIVE` twin instead,
        one that awaits the backend's own futures directly rather than
        running in a thread.

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
            The Dask block size. Three spellings are accepted.
            `"shards"` uses the array's write unit: the shard, when
            the array is sharded, or otherwise the chunk. `"chunks"`
            uses the chunk shape directly. An explicit block shape may
            also be passed instead of either name. The default,
            `"shards"`, aligns the Dask blocks to the write unit, so a
            shard is read once rather than once per inner chunk.
            `"chunks"` suits reading the array in smaller pieces.
            `"shards"` suits writing back into the array, since a
            write to a whole shard at once avoids a partial rewrite of
            it.
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

        *source* is any array-like object whose shape matches this
        array's. A Dask array is written one block at a time, so a
        source too large to fit in memory is never fully
        materialized. A plain array is written in a single write.
        This method is the write counterpart of
        [to_dask][abczarr.abc.sync.ZarrArray.to_dask]. It works for
        every backend, unlike `dask.array.to_zarr`, which requires a
        native `zarr.Array`.

        !!! example
            ```python
            array.store(dask_array)
            ```

        Parameters
        ----------
        source : array-like
            The data to write. Its shape must match this array's.
        lock : bool or str, optional
            Whether to serialize concurrent block writes. The
            default, `"auto"`, locks only when the source's blocks do
            not line up with this array's write unit, since blocks
            that each fall entirely within one chunk never write to
            the same chunk at the same time. Passing `True` or `False`
            decides the locking explicitly.
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

    *dask_chunks* is a Dask array's `.chunks`, the block sizes per
    axis. *unit* is the array's write unit. The result is true when
    every interior block boundary lands on a multiple of the unit
    size, so that no two blocks ever write the same chunk and a lock
    is unnecessary. The check is conservative. Anything it cannot
    prove aligned counts as not aligned.
    """
    unit = tuple(unit)
    if len(dask_chunks) != len(unit):
        return False
    for axis_blocks, size in zip(dask_chunks, unit):
        if not size:
            return False
        offset = 0
        # The final boundary is the array end. A partial last chunk there
        # is still written by a single block, so it need not align.
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
    shape: tx.Optional[tz.ShapeLike],
    dtype: tx.Optional[npt.DTypeLike],
    config: tx.Union[ArrayConfig, ArrayOptions, None],
    options: ArrayOptions,
    version: tz.ZarrVersion,
    data: tx.Optional[npt.ArrayLike] = None,
) -> ArrayConfig:
    """Build the resolved [ArrayConfig][abczarr.api.config.ArrayConfig] a
    `create_array` call describes.

    *config*, an `ArrayConfig` or a mapping of its fields, is the base.
    *shape*, *dtype* and the per-call *options* are layered on top of it, and
    the array takes the group's format version. A *shape* or *dtype* left out
    falls back to *data* when *data* is given, so an explicit value always
    takes precedence over the data. `"auto"` chunking and sharding are
    resolved here, so a driver receives concrete values.
    """
    base = config if isinstance(config, ArrayConfig) else ArrayConfig(
        **dict(config or {})
    )
    merged = dict(options)
    merged["zarr_version"] = version
    if shape is not None:
        merged["shape"] = shape
    if dtype is not None:
        merged["dtype"] = dtype
    return evolve(base, **merged).resolve(data)


class ZarrGroup(ZarrNode):
    """A Zarr group: a container of arrays and subgroups.

    A `ZarrGroup` is indexed like a mapping. Each member, whether an
    array or a subgroup, is reached by its name.

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
        """Create a subgroup named *name*, or open it if one already
        exists.

        Parameters
        ----------
        name : str
            The subgroup's name.
        overwrite : bool, optional
            Replace an existing member named *name* instead of
            raising an error.

        Returns
        -------
        ZarrGroup
            The created or opened subgroup.
        """
        ...

    def create_array(
        self,
        name: str,
        shape: tx.Optional[tz.ShapeLike] = None,
        dtype: tx.Optional[npt.DTypeLike] = None,
        *,
        data: tx.Optional[npt.ArrayLike] = None,
        config: tx.Union[ArrayConfig, ArrayOptions, None] = None,
        **options: tx.Unpack[ArrayOptions],
    ) -> ZarrArray:
        """Create a new array named *name* within this group.

        The array is created from existing *data* when *data* is given. The
        array's shape and dtype then default to the data's, and the data is
        written into the new array. A *shape* or *dtype* passed explicitly, or
        one carried by *config*, takes precedence over the data.

        Parameters
        ----------
        name : str
            The array's name.
        shape : tuple of int, optional
            The array's shape. Required unless *data* or *config* supplies one.
        dtype : numpy dtype, optional
            The array's data type. Required unless *data* or *config* supplies
            one.
        data : array-like, optional
            Existing data to size the array from and write into it.
        config : ArrayConfig or mapping, optional
            A reusable [ArrayConfig][abczarr.api.config.ArrayConfig], or a
            mapping of the same fields. Individual fields may also be passed as
            keyword arguments, which override the config.
        **options
            Individual [ArrayConfig][abczarr.api.config.ArrayConfig] fields,
            such as `chunks` or `compressor`. Any field passed here overrides
            the same field on *config*.

        Returns
        -------
        ZarrArray
            The newly created array.
        """
        if data is not None and (
            getattr(data, "shape", None) is None
            or getattr(data, "dtype", None) is None
        ):
            data = np.asarray(data)
        resolved = _resolve_array_config(
            shape, dtype, config, options, self.zarr_version, data
        )
        array = self._create_array(name, resolved)
        if data is not None:
            array.store(data)
        return array

    @abstractmethod
    def _create_array(self, name: str, config: ArrayConfig) -> ZarrArray:
        """Create the array named *name* from a resolved *config*, with the
        backend's own creation, so the backend writes its own metadata."""
        ...

    def as_async(self) -> "AsyncZarrGroup":
        """The coroutine twin of this group, over the same backend
        handle.

        By default, the returned twin runs this group's navigation
        and creation in a bounded thread pool, and its `"async"`
        capability reports `Support.SYNTHESIZED`. A group that stores
        its members as directories, such as a
        [PathGroup][abczarr.abc.sync.PathGroup], and a backend with
        its own async group both return a `Support.NATIVE` twin
        instead.
        """
        from .asynchronous import ThreadedAsyncGroup

        return ThreadedAsyncGroup(self)


class PathGroup(ZarrGroup):
    """A group backed by nothing but a key-value store.

    Some backends do not provide a group object of their own.
    TensorStore opens only arrays, and a plain key-value store holds
    only keys. A `PathGroup` supplies the full group surface over such
    a backend. In that setting, a group is a location that carries
    Zarr group metadata. The `PathGroup` reads that metadata, lists
    the group's members, and navigates into its subgroups and arrays.
    It can also create a subgroup, which requires only writing new
    group metadata.

    A backend adapts a `PathGroup` by subclassing it and overriding
    `_open_array` to open a child array through the backend. A
    subclass that also creates arrays overrides `_create_array`. A
    subgroup needs no override, because a subgroup is another
    `PathGroup` of the same subclass. An entire hierarchy is therefore
    reachable from a single opened group.

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
        """This group's Zarr metadata."""
        # Loaded from the store once, then kept in memory, since the I/O is
        # the open. An attribute write updates this cache in place.
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
        """The kind and version of *store_path* when it is a member of
        this group, else `None`.

        A member is a Zarr node written in this group's own format
        version. A node of a different version is not treated as a
        child, because a Zarr hierarchy is written in a single
        version.
        """
        detected = _node_at(store_path)
        if detected is None or detected[1] != self.zarr_version:
            return None
        return detected

    def keys(self) -> tx.Iterator[str]:
        """The names of this group's members, in store order.

        A member may be a subgroup or an array. Both kinds are
        included.
        """
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
        """Create a subgroup named *name*, or open it if one already
        exists.

        Parameters
        ----------
        name : str
            The subgroup's name.
        overwrite : bool, optional
            Replace an existing member named *name* instead of
            raising an error.

        Returns
        -------
        PathGroup
            The created or opened subgroup.
        """
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

        This method is the fallback for a backend with no native
        creation. It writes the config's metadata to the child
        directory and opens it through `_open_array`. A backend that
        creates natively, such as zarr-python or TensorStore,
        overrides this method.
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
    # rest of the surface (listing, navigation, subgroups, and writing an
    # array's metadata) is backend-independent.

    def _open_array(self, store_path: tz.PathLike) -> ZarrArray:
        """Open the child array at *store_path*.

        `PathGroup` does not know how to open an array on its own. A
        driver overrides this method to open one with its own
        backend.
        """
        raise UnsupportedZarrOperation("open an array")

    def as_async(self) -> "AsyncPathGroup":
        """The coroutine twin of this group: a native async path
        group.

        The returned
        [AsyncPathGroup][abczarr.abc.asynchronous.AsyncPathGroup]
        lists and navigates its members directly through an
        [AsyncStore][abczarr.abc.store.AsyncStore] over the group's
        location, rather than running the synchronous listing in a
        thread. Its array children are opened in the async color as
        well. Its `"async"` capability reports `Support.NATIVE`.
        """
        from .asynchronous import AsyncPathGroup

        return AsyncPathGroup(self)
