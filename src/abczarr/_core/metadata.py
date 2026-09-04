# stdlib
import re
from collections import abc

# dependencies
import numpy as np
import typing_extensions as tx

# locals
from abczarr._core import typing as tz
from abczarr._core.auto import (
    Converter,
    autofrozen,
    fields,
    register_converter,
)
from abczarr._core.auto.attrs import json_key


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

    def to_json(self) -> tz.JsonDict:
        """Convert this metadata to a JSON-serializable dict.

        Serializes this object's own fields. A nested metadata value is
        serialized through its own `to_json`, so a subclass that overrides it
        (an [Extension][abczarr.metadata.v3.extensions.Extension] that writes
        itself as a bare name) is respected.
        """
        return _serialize_meta(self)

    @classmethod
    def from_json(cls, data: tz.JsonDict) -> tx.Self:
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

        # Find the most specific subclass whose discriminator keys are all
        # satisfied. ``typing.Any`` means "this key is present in the document
        # with any value"; a regex means "present and matching"; a plain value
        # means "equal" -- taken from the document, or, when absent there, from
        # what this class's own defaults already imply (``defaults``). A
        # discriminator names a field by its Python name; its value is read out
        # of the document under that field's JSON key (its ``json=`` alias, so
        # ``image_label`` reads ``image-label`` and ``bioformats2raw_layout``
        # reads ``bioformats2raw.layout``).
        defaults = {f.name: f.default for f in fields(cls) if f.init}
        best = None
        best_score = ()
        for match, subcls in reversed(cls._registry().items()):
            if not issubclass(subcls, cls):
                continue
            score = _match_score(match, data, defaults, subcls)
            if score is not None and score > best_score:
                best, best_score = subcls, score
        if best is not None:
            cls = best

        # Split known fields from extra fields (on a copy -- from_json must
        # not mutate the caller's dict). Each field is read under its JSON key
        # (its ``json=`` alias, or its name) and that key is consumed, so an
        # aliased key like ``bioformats2raw.layout`` populates its typed field
        # and does not also land in ``extra_items``.
        data = dict(data)
        filtered_data = {}
        for f in fields(cls):
            key = json_key(f)
            if key not in data:
                continue
            value = data.pop(key)
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
    tz.JsonScalar, Metadata, tx.Tuple["_JSONMetadata", ...]
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


def _match_score(
    match: tx.Tuple[tx.Tuple[str, tx.Any], ...],
    data: tx.Mapping[str, tx.Any],
    defaults: tx.Mapping[str, tx.Any],
    subcls: type,
) -> tx.Optional[tx.Tuple[int, int, int]]:
    """Score how well *match* fits the data, or ``None`` if it does not.

    *data* is the document as written (keyed by its JSON keys); *defaults* is
    what the class ``from_json`` was called on already implies -- its own
    fields' defaults, keyed by field name. A discriminator names a field by its
    Python name and its value is read from *data* under that field's JSON key
    (its ``json=`` alias, or its name). A ``typing.Any`` key must appear in
    *data* itself (a discriminator that is only ever implied is not a
    discriminator). A value constraint (a literal, or a regex) is satisfied by
    the value in *data*, or, when the key is absent there, by the class's own
    default -- so ``ArrayMetadata.from_json`` still resolves an array document
    that omits the ``node_type`` the class already fixes.

    A discriminator only counts when it names one of *subcls*'s own init
    fields: a value it does not carry as a settable field is not a shape it can
    be told apart by (a codec whose ``id`` is a class attribute is recovered
    another way, not selected here).

    A higher score is a more specific match. The score ranks by number of keys,
    then subclass depth (a derived carrier beats its base), then number of
    value constraints (an exact literal beats ``Any``).
    """
    init_fields = {f.name: f for f in fields(subcls) if f.init}
    concrete = 0
    for name, want in match:
        f = init_fields.get(name)
        if f is None:
            return None
        key = json_key(f)
        if want is tx.Any:
            if key not in data:
                return None
            continue
        concrete += 1
        if key in data:
            value = data[key]
        elif name in defaults:
            value = defaults[name]
        else:
            return None
        if isinstance(want, re.Pattern):
            if not (isinstance(value, str) and want.match(value)):
                return None
        elif value != want:
            return None
    return (len(match), len(subcls.__mro__), concrete)


def _serialize_dict(x: tx.Mapping) -> tx.Dict[str, tz.Json]:
    if not callable(getattr(x, "items", None)):
        x = dict(**x)
    return {k: _to_json(v) for k, v in x.items()}


def _serialize_meta(x: "Metadata") -> tx.Dict[str, tz.Json]:
    """Serialize a metadata object's own fields (not respecting a to_json
    override on *x* itself -- that is the caller's job)."""
    extra = getattr(x, "extra_items", False)
    out = {
        json_key(f): _to_json(getattr(x, f.name))
        for f in fields(x)
        if f.name != "extra_items"
    }
    if extra:
        out.update(_serialize_dict(extra))
    return out


def _to_json(obj: tx.Any) -> tz.Json:
    if _is_metadata(obj):
        # delegate to the value's own to_json, so a subclass that serializes
        # itself specially (an Extension written as a bare name) is honored
        return obj.to_json()
    elif isinstance(obj, np.dtype):
        # a numpy dtype is not JSON: emit its zarr string form ("<f8")
        return obj.str
    elif isinstance(obj, np.generic):
        return obj.item()
    elif _is_mapping(obj):
        return _serialize_dict(obj)
    elif _is_iterable(obj):
        return [_to_json(v) for v in obj]
    else:
        return obj


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


_METADATALIKE = tx.Union[Metadata, tz.Json]
METADATA = tx.TypeVar("METADATA", bound=Metadata, default=Metadata)
METADATALIKE = tx.TypeVar(
    "METADATALIKE", bound=_METADATALIKE, default=_METADATALIKE
)


@register_converter(Metadata)
class MetadataConverter(Converter[METADATA, METADATALIKE]):

    DEFAULT = Metadata
    FALLBACK = Metadata

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        if self.hint in __reentrant:
            return self.hint
        __reentrant += (self.hint,)
        hints = (self.hint, tz.JsonDict)
        if (
            isinstance(self.hint, type) and
            issubclass(self.hint, Metadata)
        ):
            for f in fields(self.hint):
                if f.init and not f.kw_only:
                    hints += (f.type,)
                    break
        return tx.Union[hints]

    def __call__(self, value: METADATALIKE) -> METADATA:
        fallback = self.fallback
        if isinstance(fallback, type) and isinstance(value, fallback):
            return value
        elif isinstance(value, abc.Mapping):
            return fallback.from_json(value)
        else:
            return fallback(value)
