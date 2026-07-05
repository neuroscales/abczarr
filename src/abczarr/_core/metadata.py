# stdlib
import re
from collections import abc

# dependencies
import typing_extensions as tx

# locals
from abczarr._core import typing as tz
from abczarr._core.auto import (
    Converter,
    autofrozen,
    fields,
    register_converter,
)


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
    def _registry(cls) -> dict:
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
            extra = self.extra_items or {}
            return extra[key]

    def __iter__(self) -> tx.Iterator[tx.Tuple[str, tx.Any]]:
        for f in fields(self):
            if f.name == "extra_items":
                continue
            yield f.name
        if hasattr(self, "extra_items"):
            yield from self.extra_items or {}

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


_JSONMetadata = tx.Union[
    tz._JSONScalar, Metadata, tx.Tuple["_JSONMetadata", ...]
]
JSONMetadata = tx.TypeVar(
    "JSONMetadata", bound=_JSONMetadata, default=_JSONMetadata
)


@autofrozen(extra_items=JSONMetadata)
class FlexibleMetadata(Metadata):
    """A flexible metadata class that allows extra fields."""
    ...


# ======================================================================
#
#                                 UTILS
#
# ======================================================================


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
METADATALIKE = tx.TypeVar(
    "METADATALIKE", bound=_METADATALIKE, default=_METADATALIKE
)


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
