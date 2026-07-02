"""
Metadata handling for Zarr.

This file contains code from the Zarr project
https://github.com/zarr-developers/zarr-python
"""
__all__ = [
    "Metadata",
    "FlexibleMetadata",
    "NodeMetadata",
    "GroupMetadata",
    "ArrayMetadata",
    "NodeMetadataV1",
    "ArrayMetadataV1",
    "NodeMetadataV2",
    "GroupMetadataV2",
    "ArrayMetadataV2",
    "NodeMetadataV3",
    "GroupMetadataV3",
    "ArrayMetadataV3",
]

# stdlib
import json
import os
import re
import tempfile
from collections import abc

# dependencies
import typing_extensions as tx

# locals
from abczarr._core import typing as tz
from abczarr._core import constants
from abczarr._core.attrs import (
    Converter, register_converter, autofrozen, fields, evolve
)


# ======================================================================
#
#                                 BASES
#
# ======================================================================


def register_subclass(
    match: tx.Tuple[tx.Tuple[str, tx.Any], ...] = (),
    **other_matches
) -> tx.Callable[[tx.Type["Metadata"]], tx.Type["Metadata"]]:
    """
    Register a subclass of Metadata for a given match dictionary.

    The base class' `__new__` method will return an instance of the
    registered subclass if its input parameters match the given dictionary.
    """

    if isinstance(match, abc.Mapping):
        match = match.items()
    match = dict(map(tuple, match))
    match.update(other_matches)
    match = tuple(match.items())

    def decorator(cls: tx.Type[Metadata]) -> tx.Type[Metadata]:
        for base in cls.__mro__[1:]:
            if not issubclass(base, Metadata):
                continue
            if base is Metadata:
                continue
            if "_REGISTRY" not in base.__dict__:
                base._REGISTRY = {}
            base._REGISTRY[match] = cls
        return cls

    return decorator


@autofrozen
class Metadata:
    """Frozen, recursive, JSON-serializable metadata class."""

    # --- Subclass registry --------------------------------------------

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # Some subclasses register themselves with their base class,
        # so that the base class can return an instance of the subclass
        # if the input parameters match the subclass' fields.
        # This allows for polymorphic behavior when creating instances
        # of the base class.

        for match, subcls in cls._registry().items():

            # Not a subclass -> pass
            if not issubclass(subcls, cls):
                continue

            # Check if the match dictionary matches the input arguments
            match_copy = dict(match)
            args_copy = list(args)
            kwargs_copy = dict(kwargs)
            for f in fields(subcls):
                if not f.init:
                    continue
                if not f.kw_only and args_copy:
                    kwargs_copy[f.name] = args_copy.pop(0)
                if f.name not in kwargs_copy:
                    kwargs_copy[f.name] = f.default
                if f.name in match_copy:
                    kwargs_value = kwargs_copy.get(f.name)
                    match_value = match_copy.get(f.name)
                    if isinstance(match_value, re.Pattern):
                        if not match_value.match(kwargs_value):
                            break
                    elif kwargs_value != match_value:
                        break
                    match_copy.pop(f.name)
            if not match_copy:
                return super().__new__(subcls)

        return super().__new__(cls)

    @classmethod
    def _registry(cls):
        # Return the dictionary of registered subclasses.
        return {
            match: subcls
            for match, subcls in getattr(cls, "_REGISTRY", {}).items()
            if issubclass(subcls, cls)
        }

    # --- Dict-like interface ------------------------------------------
    # NOTE: Metadata is not a subclass of abc.Mapping, but it implements
    # `__getitem__` and `keys()` and can therefore be unpacked as a dict.

    def __getitem__(self, key: str) -> tx.Any:
        if any(f.name == key for f in fields(self)):
            return getattr(self, key)
        if hasattr(self, "extra_items"):
            extra = getattr(self, "extra_items") or {}
            return extra[key]

    def __iter__(self) -> tx.Iterator[tx.Tuple[str, tx.Any]]:
        for f in fields(self):
            if f.name == "extra_items":
                continue
            yield f.name
        if hasattr(self, "extra_items"):
            yield from getattr(self, "extra_items") or {}

    def keys(self) -> tx.Tuple[str, ...]:
        return tuple(self)

    # --- JSON conversion ----------------------------------------------

    def to_dict(self) -> tz.JSONDict:
        """Convert this metadata to a JSON-serializable dict."""
        return _to_json(self)

    @classmethod
    def from_dict(cls, data: tz.JSONDict) -> tx.Self:
        """Create an instance from a JSON-serializable dict."""

        # If not a dict, try to interpret it as a positional argument
        if not isinstance(data, abc.Mapping):
            for f in fields(cls):
                if f.init and not f.kw_only:
                    data = {f.name: data}
                    break

        # If no positional argument -> error
        if not isinstance(data, abc.Mapping):
            raise TypeError(
                f"Cannot create {cls.__name__} from non-mapping data: {data}"
            )

        # Try to find a matching subclass
        for match, subcls in reversed(cls._registry().items()):
            if not issubclass(subcls, cls):
                continue

            match_copy = dict(match)
            data_copy = dict(data)

            for f in fields(subcls):
                if not f.init:
                    continue
                if f.name not in data_copy:
                    data_copy[f.name] = f.default
                if f.name in match_copy:
                    data_value = data_copy.get(f.name)
                    match_value = match_copy.get(f.name)
                    if isinstance(match_value, re.Pattern):
                        if not match_value.match(data_value):
                            break
                    elif data_value != match_value:
                        break
                    match_copy.pop(f.name)

            if not match_copy:
                cls = subcls
                break

        # Split known fields from extra fields
        filtered_data = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            value = data.pop(f.name)
            if not f.init:
                if value != f.default:
                    raise ValueError(
                        f"Field {f.name} is not initable and has a "
                        f"default value of {f.default}, but got {value}"
                    )
            else:
                filtered_data[f.name] = value

        # Assign extra fields
        if data:
            filtered_data["extra_items"] = data

        return cls(**filtered_data)


_JSONMetadata = tx.Union[tz._JSONScalar, Metadata, tx.Tuple["_JSONMetadata", ...]]
JSONMetadata = tx.TypeVar("JSONMetadata", bound=_JSONMetadata, default=_JSONMetadata)


@autofrozen(extra_items=JSONMetadata)
class FlexibleMetadata(Metadata):
    """A flexible metadata class that allows extra fields."""
    ...


# ======================================================================
#
#                             NODE BASE
#
# ======================================================================


@autofrozen
class NodeMetadata(Metadata):

    attributes: tz.JSONDict
    zarr_format: tz.ZarrVersion = 3
    node_type: tz.NodeType = "group"

    # Convenience updaters (immutably return new metadata)
    def update_attributes(self, attributes: tz.JSONDict) -> tx.Self:
        """Return a new Metadata with updated attributes."""
        return evolve(self, attributes=dict(attributes))

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""
        zarr_json = root / constants.Z3_JSON
        if zarr_json.exists():
            return NodeMetadataV3.from_file(root)
        zgroup = root / constants.Z2GROUP_JSON
        zarrays = root / constants.Z2ARRAY_JSON
        if zgroup.exists() or zarrays.exists():
            return NodeMetadataV2.from_files(root)
        zmeta = root / constants.Z1META_JSON
        if zmeta.exists():
            return NodeMetadataV1.from_files(root)
        raise FileNotFoundError(
            f"No metadata found in {root}.Expected one of: "
            f"{constants.Z3_JSON}, "
            f"{constants.Z2GROUP_JSON}, "
            f"{constants.Z2ARRAY_JSON}, "
            f"{constants.Z1META_JSON}"
        )


@register_subclass(node_type="group")
@autofrozen
class GroupMetadata(NodeMetadata):

    node_type: tx.Literal["group"] = "group"


@register_subclass(node_type="array")
@autofrozen
class ArrayMetadata(NodeMetadata):

    node_type: tx.Literal["array"] = "array"


# ======================================================================
#
#                                   V1
#
# ======================================================================


@register_subclass(zarr_format=1)
@autofrozen
class NodeMetadataV1(NodeMetadata):

    zarr_format: tx.Literal[1] = 1

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""
        attrs = {}
        zattrs = root / constants.Z1ATTRS_JSON
        if zattrs.exists():
            with zattrs.open("r", encoding="utf-8") as f:
                attrs = json.load(f)

        meta = {}
        zmeta = root / constants.Z1META_JSON
        if zmeta.exists():
            with zmeta.open("r", encoding="utf-8") as f:
                meta = json.load(f)

        meta.setdefault("zarr_format", 1)

        if cls is NodeMetadataV1:
            # There are no groups in Zarr v1
            cls = getattr(ArrayMetadataV1, "_IMPL", ArrayMetadataV1)

        return cls.from_dict({**meta, "attributes": attrs})

    @classmethod
    def from_dict(cls, data: tz.JSONDict) -> tx.Self:
        if cls is NodeMetadataV1:
            # There are no groups in Zarr v1
            cls = getattr(ArrayMetadataV1, "_IMPL", ArrayMetadataV1)
        return super().from_dict(cls, data)


@register_subclass(zarr_format=1, node_type="array")
@autofrozen
class ArrayMetadataV1(NodeMetadataV1, ArrayMetadata):
    ...


# ======================================================================
#
#                                   V2
#
# ======================================================================


@register_subclass(zarr_format=2)
@autofrozen
class NodeMetadataV2(NodeMetadata):

    zarr_format: tx.Literal[2] = 2

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""

        # --- Detect node type ---

        if cls is NodeMetadataV2:

            if (root / constants.Z2ARRAY_JSON).exists():
                return ArrayMetadataV2.from_files(root)

            if (root / constants.Z2GROUP_JSON).exists():
                return GroupMetadataV2.from_files(root)

            raise FileNotFoundError(
                f"No Zarr v2 metadata found in {root}. Expected one of: "
                f"{constants.Z2ARRAY_JSON}, {constants.Z2GROUP_JSON}"
            )

        # --- We know our node type ---

        if issubclass(cls, ArrayMetadataV2):
            META_JSON = constants.Z2ARRAY_JSON
        elif issubclass(cls, GroupMetadataV2):
            META_JSON = constants.Z2GROUP_JSON
        else:
            raise ValueError(
                f"Cannot determine metadata type for {cls.__name__}"
            )

        meta = {}
        zgroup = root / META_JSON
        if zgroup.exists():
            with zgroup.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.setdefault("zarr_format", 2)

        attrs = {}
        zattrs = root / constants.Z2ATTRS_JSON
        if zattrs.exists():
            with zattrs.open("r", encoding="utf-8") as f:
                attrs = json.load(f)

        return cls.from_dict({**meta, "attributes": attrs})

    @classmethod
    def to_file(self, root: os.PathLike) -> None:
        """Write this metadata to the specified root directory."""
        new_meta = self.to_dict()
        new_attrs = new_meta.pop("attributes", {})

        META_JSON = {
            "array": constants.Z2ARRAY_JSON,
            "group": constants.Z2GROUP_JSON,
        }[self.node_type]

        meta = {}
        mpath = root / META_JSON
        if mpath.exists():
            with mpath.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        meta.update(new_meta)

        apath = root / constants.Z2ATTRS_JSON
        if apath.exists():
            with apath.open("r", encoding="utf-8") as f:
                attrs = json.load(f)
        else:
            attrs = {}
        attrs.update(new_attrs)

        _atomic_write(mpath, meta)
        _atomic_write(apath, attrs)


@register_subclass(zarr_format=2, node_type="group")
@autofrozen
class GroupMetadataV2(NodeMetadataV2, GroupMetadata):
    ...


@register_subclass(zarr_format=2, node_type="array")
@autofrozen
class ArrayMetadataV2(NodeMetadataV2, ArrayMetadata):
    ...


# ======================================================================
#
#                                   V3
#
# ======================================================================


@register_subclass(zarr_format=3)
@autofrozen
class NodeMetadataV3(NodeMetadata):
    """Metadata for Zarr v3, including attributes and format version."""

    zarr_format: tx.Literal[3] = 3

    @classmethod
    def from_file(cls, root: os.PathLike) -> tx.Self:
        """Load metadata from the specified root directory."""
        zarr_json = root / constants.Z3_JSON
        if zarr_json.exists():
            with zarr_json.open("r", encoding="utf-8") as f:
                d = json.load(f)
        return cls.from_dict(d)

    def to_file(self, root: os.PathLike) -> None:
        """Write this metadata to the specified root directory."""
        path = root / constants.Z3_JSON
        data = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        data.update(self.to_dict())
        _atomic_write(path, data)


@register_subclass(zarr_format=3, node_type="group")
@autofrozen
class GroupMetadataV3(NodeMetadataV3, GroupMetadata):
    ...


@register_subclass(zarr_format=3, node_type="array")
@autofrozen
class ArrayMetadataV3(NodeMetadataV3, ArrayMetadata):
    ...


# ======================================================================
#
#                                 UTILS
#
# ======================================================================


def _atomic_write(path: os.PathLike, data: tz.JSONDict) -> None:
    """Write JSON data to path atomically."""
    PathType = type(path)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".meta_tmp_", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        PathType(tmp).replace(path)
    finally:
        try:
            if PathType(tmp).exists():
                PathType(tmp).unlink()
        except Exception:
            pass


def _to_json(obj: tx.Any) -> tz.JSON:

    def _serialize_list(x: tx.Iterable) -> tx.List[tz.JSON]:
        return [_to_json(v) for v in x]

    def _serialize_dict(x: tx.Mapping) -> tx.Dict[str, tz.JSON]:
        if not callable(getattr(x, "items", None)):
            x = dict(**x)
        return {k: _to_json(v) for k, v in x.items()}

    def _serialize_meta(x: Metadata) -> tx.Dict[str, tz.JSON]:
        extra = getattr(x, "extra_items", False)
        out = {
            f.name: _to_json(getattr(x, f.name))
            for f in fields(x)
            if f.name != "extra_items"
        }
        if extra:
            out.update(_serialize_dict(extra))
        return out

    def _serialize_item(x: tx.Any) -> None:
        if _is_metadata(x):
            return _serialize_meta(x)
        elif _is_mapping(x):
            return _serialize_dict(x)
        elif _is_iterable(x):
            return _serialize_list(x)
        else:
            return x

    return _serialize_item(obj)


def _is_iterable(obj: tx.Any) -> bool:
    """Check if an object is iterable (e.g., list, tuple, set, dict)."""
    str_like = (str, bytes, bytearray)
    return hasattr(obj, "__iter__") and not isinstance(obj, str_like)


def _is_mapping(obj: tx.Any) -> bool:
    """Check if an object is a mapping-like (e.g., dict)."""
    return (
        callable(getattr(obj, "keys", None)) and
        callable(getattr(obj, "__getitem__", None))
    )


def _is_metadata(obj: tx.Any) -> bool:
    """Check if an object is an instance of Metadata."""
    return isinstance(obj, Metadata)


_METADATALIKE = tx.Union[Metadata, tz.JSON]
METADATA = tx.TypeVar("METADATA", bound=Metadata, default=Metadata)
METADATALIKE = tx.TypeVar("METADATALIKE", bound=_METADATALIKE, default=_METADATALIKE)


@register_converter(Metadata)
class MetadataConverter(Converter[METADATA, METADATALIKE]):

    DEFAULT = Metadata
    FALLBACK = Metadata

    @classmethod
    def like(cls, hint: tx.Any = METADATALIKE) -> tx.Any:
        hints = (hint, tz.JSONDict)
        if (
            isinstance(hint, type) and
            issubclass(hint, Metadata)
        ):
            for f in fields(hint):
                if f.init and not f.kw_only:
                    hints += (f.type,)
                    break
        return tx.Union[hints]

    def __call__(self, value: METADATALIKE) -> METADATA:
        fallback = self.fallback
        if isinstance(fallback, type) and isinstance(value, fallback):
            return value
        elif isinstance(value, abc.Mapping):
            return fallback.from_dict(value)
        else:
            return fallback(value)
