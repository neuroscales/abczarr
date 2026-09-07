"""The base class abczarr's typed metadata classes build on: JSON
conversion, dict-like field access, and construction dispatch to the most
specific registered subclass.
"""

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
from abczarr._core.rfc2119 import MISSING


def register_subclass(
    match: tx.Tuple[tx.Tuple[str, tx.Any], ...] = (),
    **other_matches
) -> tx.Callable[[tx.Type["Metadata"]], tx.Type["Metadata"]]:
    """Register a `Metadata` subclass to be returned in place of one of
    its bases, when the base is constructed with matching field values.

    A keyword argument, or an entry in `match`, names one of the
    decorated class's own init fields and the value that field must
    equal (or, for a string field, a compiled pattern it must match) for
    that class to be selected. `Metadata.__new__` checks the registered
    subclasses of whichever class it is called on, in registration
    order, and returns an instance of the first one whose match
    conditions the constructor arguments satisfy. The registered class
    itself, called directly, still builds an instance of itself.

    Parameters
    ----------
    match : tuple of (str, Any), optional
        Field name and required value pairs, as an alternative to
        keyword arguments. This form is needed for a field name that
        is not a valid Python identifier.
    **other_matches : Any
        Field name and required value pairs.

    Returns
    -------
    callable
        A decorator that registers its class and returns it unchanged.
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
    """A frozen, JSON-serializable metadata document.

    An instance's fields are set at construction and never change
    afterward. `to_json` serializes the fields, recursing into any field
    that is itself a `Metadata` instance, and `from_json` rebuilds an
    instance from the result. A field also supports dict-like read
    access: `metadata["field"]` and iteration work alongside the usual
    attribute access.
    """

    # --- Subclass registry --------------------------------------------

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # A subclass registered through `register_subclass` is returned
        # in place of `cls` when the constructor arguments satisfy that
        # subclass's match conditions, so a caller building the base
        # class by name still gets back the specific subclass its
        # arguments describe.

        for match, subcls in cls._registry().items():

            if not issubclass(subcls, cls):
                continue

            # Fill in the constructor call the same way it would be bound
            # to `subcls`'s own fields, so a value supplied positionally
            # or omitted (and defaulted) is checked exactly like one
            # supplied by keyword.
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
                        # A regex discriminator only matches a string value.
                        # A non-string one does not match this subclass,
                        # mirroring ``_match_score``. Falling through here
                        # avoids letting ``re.Pattern.match`` raise
                        # ``TypeError`` on a non-string value.
                        if not (
                            isinstance(kwargs_value, str)
                            and match_value.match(kwargs_value)
                        ):
                            break
                    elif kwargs_value != match_value:
                        break
                    match_copy.pop(f.name)
            if not match_copy:
                return super().__new__(subcls)

        return super().__new__(cls)

    @classmethod
    def _registry(cls) -> dict:
        """The subclasses registered against `cls`, keyed by their match
        conditions.

        Only entries whose registered class is actually a subclass of
        `cls` are included, so a class further up the hierarchy does not
        see a sibling's registrations.
        """
        return {
            match: subcls
            for match, subcls in getattr(cls, "_REGISTRY", {}).items()
            if issubclass(subcls, cls)
        }

    # --- Dict-like interface ------------------------------------------
    # `Metadata` does not subclass `abc.Mapping`, but implementing
    # `__getitem__`, `__iter__` and `keys` lets an instance be unpacked
    # with `dict(metadata)` or `**metadata` like an ordinary mapping.

    def __getitem__(self, key: str) -> tx.Any:
        """Get a field's value by name, or an extra item's value by key
        on a subclass that carries `extra_items`.

        Raises `KeyError` when `key` names neither.
        """
        if any(f.name == key for f in fields(self)):
            return getattr(self, key)
        if hasattr(self, "extra_items"):
            extra = self.extra_items or {}
            return extra[key]
        raise KeyError(key)

    def __iter__(self) -> tx.Iterator[tx.Tuple[str, tx.Any]]:
        """Iterate over the field names, and, on a subclass that carries
        `extra_items`, the keys of those extra items afterward.
        """
        for f in fields(self):
            if f.name == "extra_items":
                continue
            yield f.name
        if hasattr(self, "extra_items"):
            yield from self.extra_items or {}

    def keys(self) -> tx.Tuple[str, ...]:
        """The field names, and any extra item keys, as a tuple."""
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
        """Build an instance from a JSON document.

        `data` is ordinarily a dict, keyed by each field's JSON key.
        When it is not a mapping, it is treated as the value of the
        class's first positional field, provided the class has one.
        Otherwise, `TypeError` is raised. The class actually
        constructed may be a subclass more specific than `cls`, chosen
        by `register_subclass`'s discriminators. Any key in `data` that
        names no field of the chosen class is collected into
        `extra_items`, on a subclass that carries one.

        Parameters
        ----------
        data : dict or Any
            The JSON document, or a single positional value.

        Returns
        -------
        Self
            The constructed instance, of `cls` or one of its registered
            subclasses.

        Raises
        ------
        TypeError
            When `data` is not a mapping and `cls` has no positional
            field to hold it.
        """
        if not isinstance(data, abc.Mapping):
            for f in fields(cls):
                if f.init and not f.kw_only:
                    data = {f.name: data}
                    break

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
    """A `Metadata` subclass that keeps every unrecognized JSON key
    instead of rejecting it.

    A key in a JSON document that names none of a subclass's own fields
    is collected into `extra_items` rather than raised as an error, and
    `to_json` writes those extra items back alongside the typed fields.
    """
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
    """Score how well `match` fits the data, or return `None` when it
    does not fit at all.

    `data` is the document as written, keyed by its JSON keys.
    `defaults` holds the fields' defaults, keyed by field name, of the
    class `from_json` was originally called on. A discriminator names a
    field by its Python name, and that field's value is read from
    `data` under the field's JSON key, its ``json=`` alias when it has
    one or its own name otherwise. A ``typing.Any`` discriminator counts
    only when its key is present in `data` itself, since a
    discriminator that is only ever implied by a default is not a
    discriminator. A literal or regex discriminator is satisfied by the
    value in `data`, or, when the key is absent there, by the class's
    own default. This lets ``ArrayMetadata.from_json`` still resolve an
    array document that omits the ``node_type`` the class already
    fixes.

    A discriminator counts only when it names one of `subcls`'s own
    init fields. A value the class does not carry as a settable field
    is not a shape this function can tell the class apart by. A codec
    whose ``id`` is a class attribute, for instance, is recovered
    another way and is not selected here.

    A higher score is a more specific match. Scores rank first by the
    number of discriminator keys, then by subclass depth, so a derived
    carrier beats its own base, then by the number of value
    constraints, so an exact literal beats a bare ``Any``.

    Parameters
    ----------
    match : tuple of (str, Any)
        The discriminator field names and required values, as
        `register_subclass` records them.
    data : mapping
        The JSON document being matched against, keyed by JSON key.
    defaults : mapping
        The field defaults of the class `from_json` was originally
        called on, keyed by field name.
    subcls : type
        The registered subclass being scored.

    Returns
    -------
    tuple of (int, int, int) or None
        The match's specificity score, ordered by discriminator count,
        subclass depth, and value-constraint count, or `None` when
        `match` does not fit `data` at all.
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
    """Serialize each value of the mapping `x` with `_to_json`."""
    if not callable(getattr(x, "items", None)):
        x = dict(**x)
    return {k: _to_json(v) for k, v in x.items()}


def _serialize_meta(x: "Metadata") -> tx.Dict[str, tz.Json]:
    """Serialize a metadata object's own fields.

    Does not call a `to_json` override on `x` itself. That call is the
    caller's job. An unset `Recommended`/`Optional` field holds the
    `MISSING` sentinel and is omitted from the result entirely, since
    the sentinel itself is not JSON serializable.
    """
    extra = getattr(x, "extra_items", False)
    out = {}
    for f in fields(x):
        if f.name == "extra_items":
            continue
        value = getattr(x, f.name)
        if value is MISSING:
            continue
        out[json_key(f)] = _to_json(value)
    if extra:
        out.update(_serialize_dict(extra))
    return out


def _to_json(obj: tx.Any) -> tz.Json:
    """Render one value as JSON, recursing into a mapping, an iterable,
    or a nested `Metadata` instance.

    A `Metadata` value is serialized through its own `to_json`, so a
    subclass with a custom serialization (an `Extension` written as a
    bare name) is honored. A numpy dtype renders as its Zarr string
    form (``"<f8"``), and a numpy scalar as the equivalent Python
    value. Anything else is returned unchanged.
    """
    if _is_metadata(obj):
        return obj.to_json()
    elif isinstance(obj, np.dtype):
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
    """Whether `obj` iterates like a list or tuple, rather than a single
    scalar value.

    A string, `bytes`, or `bytearray` is excluded even though it is
    iterable, since `_to_json` treats those as scalars.
    """
    str_like = (str, bytes, bytearray)
    return hasattr(obj, "__iter__") and not isinstance(obj, str_like)


def _is_mapping(obj: tx.Any) -> bool:
    """Whether `obj` has the `keys` and `__getitem__` methods a mapping
    provides.
    """
    return (
        callable(getattr(obj, "keys", None)) and
        callable(getattr(obj, "__getitem__", None))
    )


def _is_metadata(obj: tx.Any) -> bool:
    """Whether `obj` is a `Metadata` instance."""
    return isinstance(obj, Metadata)


_METADATALIKE = tx.Union[Metadata, tz.Json]
METADATA = tx.TypeVar("METADATA", bound=Metadata, default=Metadata)
METADATALIKE = tx.TypeVar(
    "METADATALIKE", bound=_METADATALIKE, default=_METADATALIKE
)


@register_converter(Metadata)
class MetadataConverter(Converter[METADATA, METADATALIKE]):
    """Converts a value to a `Metadata` instance of the field's own type.

    A value already of that type is returned as it is. A mapping is
    read through `Metadata.from_json`. Any other value is passed as the
    single positional argument to the target type's constructor, so a
    field typed to a `Metadata` subclass with one positional field
    accepts that field's value directly.
    """

    DEFAULT = Metadata
    FALLBACK = Metadata

    def like(self, __reentrant: tuple = ()) -> tx.Any:
        """The hints this converter accepts as input for its field's
        type.

        Besides the field's own type and a JSON-shaped mapping, a
        `Metadata` type with a positional first field also accepts that
        field's own type directly, matching what `__call__` does with a
        non-mapping value. `__reentrant` guards against a field type
        that refers to itself.
        """
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
        """Convert `value` to the field's `Metadata` type.

        Parameters
        ----------
        value : Metadata, mapping, or Any
            The value to convert.

        Returns
        -------
        Metadata
            `value` unchanged, when it is already an instance of the
            target type. Otherwise, the result of building that type
            from `value`.
        """
        fallback = self.fallback
        if isinstance(fallback, type) and isinstance(value, fallback):
            return value
        elif isinstance(value, abc.Mapping):
            return fallback.from_json(value)
        else:
            return fallback(value)
