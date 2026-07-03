import typing_extensions as tx


class Requirement:

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
    _INSTANCE = None
    _STR = "MUST"

    def __bool__(self) -> bool:
        return True


class Recommended(Requirement):
    _INSTANCE = None
    _STR = "SHOULD"

    def __bool__(self) -> bool:
        return False


class Optional(Requirement):
    _INSTANCE = None
    _STR = "MAY"

    def __bool__(self) -> bool:
        return False


class Prohibited(Requirement):
    _INSTANCE = None
    _STR = "MUST-NOT"

    def __bool__(self) -> bool:
        return True


class NotRecommended(Requirement):
    _INSTANCE = None
    _STR = "SHOULD-NOT"

    def __bool__(self) -> bool:
        return True


MUST = Required._INSTANCE = object.__new__(Required)
SHOULD = Recommended._INSTANCE = object.__new__(Recommended)
MAY = Optional._INSTANCE = object.__new__(Optional)
MUST_NOT = Prohibited._INSTANCE = object.__new__(Prohibited)
SHOULD_NOT = NotRecommended._INSTANCE = object.__new__(NotRecommended)
