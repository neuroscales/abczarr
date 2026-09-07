"""RFC 2119 requirement levels, usable as type-hint metadata on a metadata
field.
"""

import typing_extensions as tx

from .auto.factories import AnnotatedFactory


class Requirement:
    """Base class for an RFC 2119 requirement level.

    One of the five subclasses is used directly, not this base class:
    `Required`, `Recommended`, `Optional`, `Prohibited`, or
    `NotRecommended`. Each subclass has exactly one instance, exported
    as a module-level constant (`MUST`, `SHOULD`, `MAY`, `MUST_NOT`,
    `SHOULD_NOT`). Subscripting a subclass, as in `Required[int]`,
    annotates the field's type hint with that subclass's instance,
    recording the requirement level on the hint.

    `Requirement("MUST")` returns the `MUST` singleton, and likewise for
    each RFC 2119 keyword and its aliases (`"SHALL"` and `"REQUIRED"` both
    return `MUST`).
    """

    def __new__(cls, value: tx.Union[str, "Requirement"] = "") -> tx.Self:
        if cls is Requirement:
            return cls._INSTANCES[value.upper()]
        value = value or cls._STR
        value = str(value).upper()
        if value != cls._STR:
            raise ValueError(f"Invalid value for {cls.__name__}: {value}")
        return cls._INSTANCE

    def __str__(self) -> str:
        return self._STR

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def __class_getitem__(cls, item: tx.Any) -> tx.Self:
        if cls is Requirement:
            raise TypeError("RequirementType cannot be subscripted")
        return tx.Annotated[item, cls()]


class Required(Requirement):
    """RFC 2119 MUST: a field that has to be present for the metadata to
    be valid.

    Metadata missing a `Required` field is invalid, and a validator
    should report the field's absence as an error.
    """

    _INSTANCE = None
    _STR = "MUST"

    def __bool__(self) -> bool:
        return True


class Recommended(Requirement):
    """RFC 2119 SHOULD: a field whose absence leaves the metadata valid
    but incomplete.

    Metadata missing a `Recommended` field is still valid. A validator
    may report the absence as a warning, but must not treat it as an
    error.
    """

    _INSTANCE = None
    _STR = "SHOULD"

    def __bool__(self) -> bool:
        return False


class Optional(Requirement):
    """RFC 2119 MAY: a field whose absence has no bearing on the
    metadata's validity.

    Metadata missing an `Optional` field is valid, and a validator
    should neither warn nor error over its absence.
    """
    _INSTANCE = None
    _STR = "MAY"

    def __bool__(self) -> bool:
        return False


class Prohibited(Requirement):
    """RFC 2119 MUST NOT: a field that has to be absent for the metadata
    to be valid.

    Metadata carrying a `Prohibited` field is invalid, and a validator
    should report the field's presence as an error.
    """
    _INSTANCE = None
    _STR = "MUST-NOT"

    def __bool__(self) -> bool:
        return True


class NotRecommended(Requirement):
    """RFC 2119 SHOULD NOT: a field whose presence leaves the metadata
    valid but discouraged.

    Metadata carrying a `NotRecommended` field is still valid. A
    validator may report the presence as a warning, but must not treat
    it as an error.
    """
    _INSTANCE = None
    _STR = "SHOULD-NOT"

    def __bool__(self) -> bool:
        return True


class MissingType:
    """The sentinel value of a field that is genuinely absent.

    Distinguishes a field with no value at all from one explicitly set
    to `None`, which is itself a value. The single instance is exported
    as `MISSING`.
    """

    def __new__(cls) -> tx.Self:
        return cls._INSTANCE

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MISSING"


MUST = Required._INSTANCE = object.__new__(Required)
SHOULD = Recommended._INSTANCE = object.__new__(Recommended)
MAY = Optional._INSTANCE = object.__new__(Optional)
MUST_NOT = Prohibited._INSTANCE = object.__new__(Prohibited)
SHOULD_NOT = NotRecommended._INSTANCE = object.__new__(NotRecommended)
MISSING = MissingType._INSTANCE = object.__new__(MissingType)

Requirement._INSTANCES = {
    "MUST": MUST, "SHALL": MUST, "REQUIRED": MUST,
    "SHOULD": SHOULD, "RECOMMENDED": SHOULD,
    "MAY": MAY, "OPTIONAL": MAY,
    "MUST-NOT": MUST_NOT, "SHALL-NOT": MUST_NOT, "PROHIBITED": MUST_NOT,
    "SHOULD-NOT": SHOULD_NOT, "NOT-RECOMMENDED": SHOULD_NOT,
}


class RequirementMixin:
    """Reads the `Requirement` instance a resolver was registered against,
    caching the result.
    """

    @property
    def requirement(self) -> Requirement:
        """The `Requirement` instance this resolver was matched against."""
        if getattr(self, "_requirement", None) is None:
            self._requirement = self._get_requirement()
        return self._requirement

    def _get_requirement(self) -> Requirement:
        # The Requirement instance is the resolver's own metadata (`hint`
        # when the field carries no other type args), or one of the type
        # args alongside another metadata value.
        if isinstance(self.hint, Requirement):
            return self.hint
        for arg in self.args:
            if isinstance(arg, Requirement):
                return arg
        raise TypeError("No Requirement instance found")


@AnnotatedFactory.register_metadata(Requirement)
class RequirementFactory(RequirementMixin, AnnotatedFactory):
    """Builds the default value for a field annotated with a `Requirement`.

    A `Required` field has no default to build, since an absent required
    field makes the metadata invalid rather than merely incomplete.
    Building one raises `TypeError`. Every other requirement level
    resolves to `MISSING`, marking the field as absent.
    """

    def __call__(self) -> Requirement:
        requirement = self.requirement
        if requirement is MUST:
            raise TypeError(
                "Cannot instantiate a Required field without a default value"
            )
        return MISSING
