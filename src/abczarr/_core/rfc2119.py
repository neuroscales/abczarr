import typing_extensions as tx

from .auto.converters import Converter, register_converter
from .auto.factories import AnnotatedFactory, register_factory


class Requirement:
    """
    Base class for RFC 2119 requirement levels.

    It is recommended to use the subclasses Required, Recommended,
    Optional, Prohibited, and NotRecommended instead of this class directly.

    Instances of these subclasses are singletons and can be used as type
    annotations to indicate the requirement level of a field in a metadata
    class.

    Singletons can also be instantiated from a string value, e.g.
    `Requirement("MUST")` will return the singleton instance of the
    Required class.
    """

    def __new__(cls, value: tx.Union[str, "Requirement"]) -> tx.Self:
        if cls is Requirement:
            return cls._INSTANCES[value.upper()]
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
    """
    A field marked as REQUIRED under RFC 2119 MUST be present in the
    metadata. If it is not present, the metadata is invalid.

    Validation tools should issue an error if a REQUIRED field is missing,
    and they should consider the metadata invalid.
    """

    _INSTANCE = None
    _STR = "MUST"

    def __bool__(self) -> bool:
        return True


class Recommended(Requirement):
    """
    A field marked as RECOMMENDED under RFC 2119 SHOULD be present in the
    metadata. If it is not present, the metadata is still valid, but it may
    be missing important information.

    Validation tools may issue a warning if a RECOMMENDED field is missing,
    but they should not consider the metadata invalid.
    """

    _INSTANCE = None
    _STR = "SHOULD"

    def __bool__(self) -> bool:
        return False


class Optional(Requirement):
    """
    A field marked as OPTIONAL under RFC 2119 MAY be present in the
    metadata. If it is not present, the metadata is still valid, and it may
    be missing information that is not critical to the interpretation of the
    metadata.

    Validation tools should not issue a warning if an OPTIONAL field is
    missing, and they should not consider the metadata invalid.
    """
    _INSTANCE = None
    _STR = "MAY"

    def __bool__(self) -> bool:
        return False


class Prohibited(Requirement):
    """
    A field marked as PROHIBITED under RFC 2119 MUST NOT be present in the
    metadata. If it is present, the metadata is invalid.

    Validation tools should issue an error if a PROHIBITED field is present,
    and they should consider the metadata invalid.
    """
    _INSTANCE = None
    _STR = "MUST-NOT"

    def __bool__(self) -> bool:
        return True


class NotRecommended(Requirement):
    """
    A field marked as NOT RECOMMENDED under RFC 2119 SHOULD NOT be present in
    the metadata. If it is present, the metadata is still valid, but it may
    be present in a way that is not recommended.

    Validation tools may issue a warning if a NOT RECOMMENDED field is
    present, but they should not consider the metadata invalid.
    """
    _INSTANCE = None
    _STR = "SHOULD-NOT"

    def __bool__(self) -> bool:
        return True


class MissingType:
    """
    Special value to indicate that a field is missing.
    This is used to distinguish between a field that is explicitly set
    to None and a field that is not present at all.
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


@register_factory(Requirement)
class RequirementFactory(AnnotatedFactory):
    """
    Factory for types annotated with a Requirement instance.
    """

    @property
    def requirement(self) -> Requirement:
        for arg in self.args:
            if isinstance(arg, Requirement):
                return arg

    def __call__(self) -> Requirement:
        requirement = self.requirement
        if requirement is MUST:
            raise TypeError(
                "Cannot instantiate a Required field without a default value"
            )
        return MISSING
