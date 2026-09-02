"""The Zarr group interface: a container of arrays and subgroups."""

__all__ = [
    "ZarrGroup",
    "PathGroup",
]

# stdlib
from abc import abstractmethod

# dependencies
import numpy.typing as npt
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr.metadata.base import (
    GroupMetadataV2,
    GroupMetadataV3,
    NodeMetadata,
    node_at,
    node_type_at,
)

from .array import ZarrArray, ZarrArrayConfig
from .errors import UnsupportedZarrOperation

# locals
from .node import ZarrNode

#: The group-metadata class to write for each Zarr format version. Zarr v1
#: has no groups, so it is absent.
_GROUP_METADATA = {
    2: GroupMetadataV2,
    3: GroupMetadataV3,
}


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

    @abstractmethod
    def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        config: tx.Optional[ZarrArrayConfig] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
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
        config : ZarrArrayConfig, optional
            Chunking, sharding, and compression options. May also be
            passed as individual keyword arguments -- see
            [ZarrArrayConfig][abczarr.abc.array.ZarrArrayConfig].
        """
        ...


class PathGroup(ZarrGroup):
    """A [ZarrGroup][abczarr.abc.group.ZarrGroup] for a backend with no
    group object of its own.

    Some backends never construct a "group" -- TensorStore opens arrays
    only, and a bare key-value store holds nothing but keys. For those, a
    group is just a directory that carries Zarr group metadata, and
    `PathGroup` provides the whole group surface over it: reading its own
    metadata, listing its members, and navigating into subgroups and
    arrays -- using nothing but abczarr's own path and metadata layers.
    It can also create subgroups on its own, since that only means writing
    group metadata.

    A driver subclasses `PathGroup` and overrides
    [_open_array][abczarr.abc.group.PathGroup._open_array] (and, to support
    creating arrays too,
    [_create_array][abczarr.abc.group.PathGroup._create_array]) to say how a
    child array is opened and created with its own backend. Subgroups need
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
        return NodeMetadata.from_file(self._store_path)

    # attrs is inherited from ZarrNode: a write-through mapping over the
    # group's metadata file.

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
        detected = node_at(store_path)
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
        if node_type_at(child) is not None and not overwrite:
            raise UnsupportedZarrOperation(
                "create a group where a member already exists"
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

    def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        config: tx.Optional[ZarrArrayConfig] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrArray:
        return self._create_array(
            name, shape, dtype, config=config, **kwargs
        )

    # -- backend hooks -----------------------------------------------------
    # A driver overrides these to open and create arrays with its backend.
    # The rest of the surface -- listing, navigation, subgroups -- is
    # backend-independent and needs no override.

    def _open_array(self, store_path: tz.PathLike) -> ZarrArray:
        """Open the child array at *store_path*.

        `PathGroup` does not know how to open an array on its own; a driver
        overrides this method to open one with its own backend.
        """
        raise UnsupportedZarrOperation("open an array")

    def _create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike,
        *,
        config: tx.Optional[ZarrArrayConfig] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrArray:
        """Create a child array, for a driver that can.

        `PathGroup` can create subgroups on its own, but not arrays; a
        driver overrides this method to create one with its own backend.
        """
        raise UnsupportedZarrOperation("create_array")
