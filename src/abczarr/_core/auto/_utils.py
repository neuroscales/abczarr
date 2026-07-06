__all__ = [
    "HintMagic",
    "get_concrete_type",
    "get_default",
    "get_from_registry",
    "get_origin_uw",
    "get_args_uw",
    "safe_get_origin",
    "safe_get_args",
    "safe_isinstance",
    "safe_issubclass",
    "ishintstance",
    "issubhint",
    "unwrap",
    "UNSET",
    "UNION_TYPES",
]

# stdlib
import inspect
import math
import numbers
from collections import abc

# dependencies
import numpy as np
import typing_extensions as tx

# locals
from ._typing import NoneType, T, UnionType

# constants
UNION_TYPES = (tx.Union, UnionType)


class _UNSET:

    def __new__(cls, *args, **kwargs) -> tx.Self:
        if not hasattr(cls, "_INSTANCE"):
            cls._INSTANCE = object.__new__(cls)
        return cls._INSTANCE

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "<UNSET>"

    def __str__(self) -> str:
        return "<UNSET>"


UNSET = _UNSET()


class HintMagic(tx.Generic[T]):
    """Base class for magic objects (factories, converters)."""

    BOUND = DEFAULT = tx.Any
    FALLBACK = UNSET

    def __init__(self, hint: tx.Any = UNSET) -> None:
        """
        Parameters
        ----------
        hint : Any, optional
            The type hint to use for this magic object.
            If not provided, the default hint for the class is used.
        """
        if hint is UNSET:
            hint = self.DEFAULT
        self.hint = hint
        self.__post_init__()

    def __post_init__(self) -> None:
        if not issubhint(self.hint, self.BOUND):
            raise TypeError(
                f"Hint {self.hint} is not a valid subhint for {self.BOUND}"
            )

    @property
    def unwrapped(self) -> tx.Any:
        """The unwrapped type hint, with any `Annotated` wrappers removed."""
        if getattr(self, "_unwrapped", None) is None:
            self._unwrapped = self._get_unwrapped()
        return self._unwrapped

    def _get_unwrapped(self) -> tx.Any:
        return unwrap(self.hint)

    @property
    def origin(self) -> tx.Any:
        """
        The "safe" origin of the type hint

        * Any `Annotated` wrappers are removed.
        * If the origin is `None`, the hint itself is returned.
        """
        if getattr(self, "_origin", None) is None:
            self._origin = self._get_origin()
        return self._origin

    def _get_origin(self) -> tx.Any:
        return get_origin_uw(self.hint)

    @property
    def args(self) -> tx.Tuple[tx.Any, ...]:
        """
        The "safe" arguments of the type hint

        * Any `Annotated` wrappers are removed.
        * If the origin is `None`, returns an empty tuple.
        """
        if getattr(self, "_args", None) is None:
            self._args = self._get_args()
        return self._args

    def _get_args(self) -> tx.Tuple[tx.Any, ...]:
        return get_args_uw(self.hint)

    @property
    def fallback(self) -> tx.Any:
        """A "concrete" fallback type for the type hint, if possible."""
        if getattr(self, "_fallback", None) is None:
            self._fallback = self._get_fallback()
        return self._fallback

    def _get_fallback(self) -> tx.Any:
        try:
            return get_concrete_type(self.hint, self.FALLBACK)
        except TypeError:
            return self.hint

    def __call__(self, *args, **kwargs) -> T:
        """Do some magic!"""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement __call__"
        )

    def __repr__(self) -> str:
        hint_arg = self.hint if self.hint != self.DEFAULT else ""
        return f"{type(self).__name__}({hint_arg})"

    def __str__(self) -> str:
        return repr(self)

    def error(
        self, message: str, value: tx.Any = UNSET, **kwargs
    ) -> "MagicError":
        """Raise a MagicError with the given value and message."""
        type = kwargs.pop("type", MagicError)
        kwargs.setdefault("this", self)
        kwargs.setdefault("value", value)
        raise type(message, **kwargs)


class MultipleCauses(Exception):
    """A wrapper exception that contains multiple causes."""

    def __init__(self, causes: tx.Iterable[Exception]) -> None:
        super().__init__()
        self.__all_causes__ = tuple(causes)


class MagicError(Exception):
    """An exception raised by magic objects (factories, converters)."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Other Parameters
        ----------------
        this : HintMagic
            The HintMagic instance that raised the error.
        value : Any
            The value that caused the error.
        """
        this = kwargs.pop("this", None)
        value = kwargs.pop("value", UNSET)
        self.this = this
        self.value = value
        if args:
            msg, *args = args
        else:
            msg = ""
        self.message = msg
        msg = self._make_message(msg, this=True, value=True, causes=False)
        super().__init__(msg, *args)

    @property
    def nice_message(self) -> str:
        return getattr(self, "args", ("",))[0]

    @property
    def causes(self) -> tx.Tuple[Exception, ...]:
        if hasattr(self, "__all_causes__"):
            return self.__all_causes__
        if self.__cause__ is not None:
            return (self.__cause__,)
        return ()

    @property
    def depth(self) -> int:
        return 1 + max(getattr(p, "depth", 0) for p in self.causes)

    @property
    def best_cause(self) -> tx.Optional[tx.Self]:
        return max(
            self.causes,
            key=lambda p: getattr(p, "depth", 0),
            default=None
        )

    def _make_message(
        self,
        message: tx.Optional[str] = None,
        this: bool = True,
        value: bool = True,
        causes: bool = True
    ) -> str:
        if message is None:
            message = self.nice_message or ""

        if this:
            if message:
                message = f"{self.this!r}: {message}"
            else:
                message = f"{self.this!r}"

        if value:
            message = f"{message}\n|> value = {self.value!r}"

        if causes and self.causes:
            arrow = "?> " if len(self.causes) > 1 else "->"
            value = len(self.causes) == 1
            for cause in self.causes:
                if hasattr(cause, "_make_message"):
                    cause_message = cause._make_message(
                        this=this, value=value
                    )
                    message = f"{message}\n{arrow} {cause_message}"
        return message

    # def __str__(self) -> str:
    #     return self._make_str()


def get_concrete_type(hint: tx.Any, fallback: type = UNSET) -> tx.Type[tx.Any]:
    """Get a valid concrete type from a type hint."""
    origin = safe_get_origin(hint, unwrap=(tx.Annotated, tx.TypeVar))
    if safe_isinstance(origin, type) and not inspect.isabstract(origin):
        return origin
    if safe_isinstance(fallback, type):
        return fallback
    raise TypeError(
        f"Cannot get concrete type for hint {hint} (of type {type(hint)}) "
        f"and fallback {fallback} (of type {type(fallback)})."
    )


def get_default(hint: tx.Any) -> tx.Any:
    """
    Get a default value from a type hint.

    * If the hint is a `Union` that contains `NoneType`, `None` is returned.
    * If the hint is a `Literal`, the first value in the literal is returned.
    * Otherwise, if the hint is a `Union`, we recurse through its sub-hints.
    * If no default value can be found, a `TypeError` is raised.
      A factory should then be used.
    """
    origin = safe_get_origin(hint, unwrap=tx.Annotated)
    args = safe_get_args(hint, unwrap=tx.Annotated)
    if origin is tx.Literal:
        if None in args:
            return None
        return args[0]
    if origin in (tx.Union, UnionType):
        if NoneType in args:
            return None
        for arg in args:
            try:
                return get_default(arg)
            except TypeError:
                continue
    raise TypeError(f"Cannot get default for hint {hint}")


def get_from_registry(hint: tx.Any, registry: dict) -> tx.Any:
    """
    Get the best matching value from a registry whose keys are types or
    type hints.
    """
    # First naive pass
    best_match, best_dist = _get_best_match(hint, registry)

    # Second pass, where Annotated hints are unwrapped.
    # We only use the resulting match if it is better than the first pass.
    if best_dist != 0 and safe_get_origin(hint) is tx.Annotated:
        hint = safe_get_origin(hint, unwrap=tx.Annotated)
        better_match, better_dist = _get_best_match(hint, registry)
        if better_dist < best_dist:
            best_match, best_dist = better_match, better_dist

    if best_match is not None:
        return registry[best_match]

    return None


def _get_best_match(hint: tx.Any, registry: dict) -> tx.Tuple[tx.Any, float]:
    """
    Get the best matching value from a registry whose keys are types or
    type hints, and return the key and value as a tuple.
    """
    hint = safe_get_origin(hint)

    best_match, best_dist = None, float("inf")
    for key in registry:

        dist = _type_dist(hint, key)

        if dist == 0:
            # Perfect match -> stop here
            best_match, best_dist = key, dist
            break

        elif dist < best_dist:
            if _is_typeddict(best_match) and not _is_typeddict(key):
                # Prefer typeddict over other types if they are compatible
                continue
            else:
                # Update best match
                best_match, best_dist = key, dist

        elif dist < float('inf') and _is_typeddict(key):
            # Prefer typeddict over other types if they are compatible
            best_match, best_dist = key, dist

        elif dist == best_dist:
            if safe_issubclass(key, best_match):
                # Prefer more specific subclass
                best_match = key

    return best_match, best_dist


def _type_dist(subcls: type, cls: type) -> int:
    """Distance between two types, based on their inheritance hierarchy."""
    if safe_isinstance(subcls, tx.TypeVar):
        subcls = tx.TypeVar
    if subcls is cls:
        return 0
    if not _issubclassable(subcls) or not _issubclassable(cls):
        return float("inf")
    if not safe_issubclass(subcls, cls):
        return float("inf")
    if tx.is_typeddict(cls):
        bases = _all_orig_bases(subcls)
    else:
        bases = subcls.__mro__
    distance = 0
    for base in bases:
        if base is cls:
            return distance
        distance += 1
    return 1000


def _issubclassable(cls: tx.Any) -> bool:
    if cls is tx.TypedDict:
        return True
    return isinstance(cls, type)


def _is_typeddict(cls: tx.Any) -> bool:
    if cls is tx.TypedDict:
        return True
    return tx.is_typeddict(cls)


def _all_orig_bases(cls: type, _self: bool = True) -> tx.Tuple[type, ...]:
    """Get all original bases of a type, including the type itself."""
    if not _is_typeddict(cls):
        return ()
    bases = (cls,) if _self else ()
    bases += getattr(cls, '__orig_bases__', ())
    for base in getattr(cls, '__orig_bases__', ()):
        bases += _all_orig_bases(base, _self=False)
    return bases


def safe_issubclass(subcls: tx.Any, cls: tx.Any) -> bool:
    """Safe subclass (does not fail if arguments are not types)."""
    if _is_typeddict(cls):
        return cls in _all_orig_bases(subcls) or subcls is dict
    if isinstance(subcls, type) and isinstance(cls, type):
        return issubclass(subcls, cls)
    return False


def safe_isinstance(obj: tx.Any, cls: tx.Any) -> bool:
    """Safe isinstance (does not fail if second argument is not a type)."""
    if _is_typeddict(cls):
        return safe_issubclass(type(obj), cls)
    if isinstance(cls, type) and cls is not tx.Any:
        return isinstance(obj, cls)
    return False


def ishintstance(obj: tx.Any, hint: tx.Any) -> bool:
    """Like isinstance, but the second argument can be a type hint."""
    origin_uw = get_origin_uw(hint)
    if origin_uw is type:
        return _ishintstance_type(obj, hint)
    return issubhint(type(obj), hint)


def _ishintstance_type(obj: tx.Any, hint: tx.Any) -> bool:
    """Like isinstance, but the second argument can be a type hint."""
    hint_uw = get_origin_uw(hint)
    if safe_get_origin(hint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"Hint {hint} is not a type[]")
    args_uw = tx.get_args(hint_uw)
    if not args_uw:
        # hint is `type` (or `tx.Type`), so any type is valid
        return isinstance(obj, type)
    # hint is `type[T]` (or `tx.Type[T]`), so check that obj is a subclass of T
    return isinstance(obj, type) and safe_issubclass(obj, args_uw[0])


def issubhint(hint: tx.Any, superhint: tx.Any) -> bool:
    """
    Check that a hint is a sub-hint for another hint.

    A hint is a valid subhint if all values that are valid for the hint
    are also valid for the superhint.
    """

    # shortcircuits
    if superhint is tx.Any:
        return True

    if hint is superhint:
        return True

    # Unwrap superhint origin
    origin_uw = get_origin_uw(superhint)

    if origin_uw is tx.Any:
        return True

    if isinstance(origin_uw, tx.TypeVar):
        return _isubtypevar(hint, superhint)

    if isinstance(hint, tx.TypeVar):
        # Unwrap typevar so that its bound can be checked against the
        # superhint. We've already taken care of the case where the
        # superhint is a typevar.

        # For constraints, each constraint must be a subhint of the
        # superhint
        constraints = getattr(hint, "__constraints__", ())
        if constraints:
            return all(
                issubhint(constraint, superhint)
                for constraint in constraints
            )

        # For bounds, the bound must be a subhint of the superhint
        return issubhint(unwrap(hint, tx.TypeVar), superhint)

    if origin_uw in UNION_TYPES:
        return _issubunion(hint, superhint)

    if origin_uw is tx.Literal:
        return _issubliteral(hint, superhint)

    if origin_uw is type(None):
        return _issubnone(hint, superhint)

    if origin_uw is type:
        return _issubtype(hint, superhint)

    if isinstance(origin_uw, type):
        return safe_issubclass(hint, origin_uw)

    return False


def _issubnone(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for NoneType."""
    none_uw = get_origin_uw(superhint)
    if none_uw is not type(None):
        raise TypeError(f"nonehint {superhint} is not a NoneType")
    origin_uw = get_origin_uw(hint)
    return origin_uw is type(None)


def _issubliteral(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a Literal."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) is not tx.Literal:
        # Superhint is not a literal -> error
        raise TypeError(f"Super-hint {superhint} is not a Literal")
    if safe_get_origin(hint_uw) is not tx.Literal:
          # Hint is not a Literal, cannot be a subhint
        return False
    # !! We use tx.get_origin instead of _get_origin
    # !! to differentiate tx.Literal (origin is None)
    # !! from tx.Literal[()] (origin is tx.Literal)
    if not tx.get_origin(superhint_uw):
        # All literals are subhints of `tx.Literal`
        return True
    if not tx.get_origin(hint_uw):
        # # tx.Literal is not a subhint of tx.Literal[...]
        return False
    # Check that all args of hint are in superhint
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return all(arg in superargs for arg in args)


def _isubtypevar(hint: tx.Any, superhint: tx.TypeVar) -> bool:
    """Check that a hint is a sub-hint for a TypeVar."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if not isinstance(superhint_uw, tx.TypeVar):
        # Invalid superhint -> error
        raise TypeError(f"Super-hint {superhint} is not a TypeVar")
    if hint_uw is superhint_uw:
        # Exact match
        return True
    if getattr(superhint_uw, "__constraints__", ()):
        # If constraints, check that hint is a subhint of one of them
        constraints = superhint_uw.__constraints__
        for constraint in constraints:
            if issubhint(hint, constraint):
                return True
        # Else, if hint is a TypeVar, check that all its constraints are
        # subhints of one of the superhint's constraints
        if isinstance(hint_uw, tx.TypeVar):
            subconstraints = getattr(hint_uw, "__constraints__", ())
            if not subconstraints:
                return False
            return all(
                any(
                    issubhint(subconstraint, constraint)
                    for constraint in constraints
                )
                for subconstraint in subconstraints
            )
        # Otherwise, constraints do not match
        return False
    elif getattr(superhint_uw, "__bound__", None) is not None:
        # If bound, check that hint is a subhint of the bound
        bound = superhint_uw.__bound__
        if issubhint(hint, bound):
            return True
        # Else, if hint is a TypeVar, check that all its constraints are
        # subhints of one of the superhint's constraints
        if isinstance(hint_uw, tx.TypeVar):
            # If hint is a TypeVar, check that all its bound is a
            # subhint of the superhint's bound
            subbound = getattr(hint_uw, "__bound__", None)
            if subbound is None:
                return False
            return issubhint(subbound, bound)
        # Otherwise, bound does not match
        return False
    else:
        # Unconstrained TypeVar -> any hint is a subhint
        return True


def _issubunion(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a Union."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) not in UNION_TYPES:
        # Invalid superhint -> error
        raise TypeError(f"union {superhint} is not a Union type")
    # Unwrap hint only if it gets us a union type
    if safe_get_origin(hint_uw) in UNION_TYPES:
        hint = hint_uw
    hint = tx.Union[hint]
    # !! We use tx.get_origin instead of _get_origin
    # !! to differentiate tx.Union (origin is None)
    # !! from tx.Union[...] (origin is tx.Union)
    if not tx.get_origin(superhint_uw):
        # All unions are subhints of `tx.Union`
        return True
    if not tx.get_origin(hint_uw):
        # # tx.Union is not a subhint of tx.Union[...]
        return False
    # Check that all args of hint are subhints of one of the superhint's args
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return all(
        any(issubhint(arg, superarg) for superarg in superargs)
        for arg in args
    )


def _issubtype(hint: tx.Any, superhint: tx.Any) -> bool:
    """Check that a hint is a sub-hint for a type[...] hint."""
    hint_uw = unwrap(hint)
    superhint_uw = unwrap(superhint)
    if safe_get_origin(superhint_uw) is not type:
        # Invalid superhint -> error
        raise TypeError(f"superhint {superhint} is not a type")
    if safe_get_origin(hint_uw) is not type:
        # Hint is not a type, cannot be a subhint
        return False
    if not tx.get_args(superhint_uw):
        # All types are subhints of `tx.Type`
        return True
    if not tx.get_args(hint_uw):
        # # tx.Type is not a subhint of tx.Type[...]
        return False
    # Check that the hint's arg is a subclass of the superhint's arg
    args = safe_get_args(hint_uw)
    superargs = safe_get_args(superhint_uw)
    return safe_issubclass(args[0], superargs[0])


def unwrap(hint: tx.Any, origin: tx.Any = (tx.Annotated,)) -> tx.Any:
    """
    Unwrap a type hint from its origin, if it is in the unwrap list.

    !!! example
        ```python
        unwrap(Annotated[int, "metadata"], unwrap=Annotated)   # returns int
        unwrap(Optional[int], unwrap=Optional)                 # returns int
        ```
    """
    if origin is None:
        origin = ()
    if not isinstance(origin, abc.Sequence):
        origin = (origin,)
    if safe_get_origin(hint) in origin:
        return unwrap(tx.get_args(hint)[0], origin=origin)
    if tx.TypeVar in origin and safe_isinstance(hint, tx.TypeVar):
        return unwrap(_unwrap_typevar(hint), origin=origin)
    return hint


_unwrap = unwrap  # alias for convenience


def _unwrap_typevar(hint: tx.Any, __rentrant: tuple = ()) -> tx.Any:
    origin = get_origin_uw(hint)
    if origin in __rentrant:
        return origin
    __rentrant += (origin,)
    if not safe_isinstance(origin, tx.TypeVar):
        return hint
    if getattr(origin, "__default__", tx.NoDefault) is not tx.NoDefault:
        return _unwrap_typevar(origin.__default__, __rentrant=__rentrant)
    if getattr(origin, "__constraints__", ()):
        return tx.Union[origin.__constraints__]
    if getattr(origin, "__bound__", None) is not None:
        return _unwrap_typevar(origin.__bound__, __rentrant=__rentrant)
    return tx.Any


def safe_get_origin(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Any:
    """
    Safe `get_origin`.

    Returns the input type, instead of `None`, if the input is not a
    generic type. Can also unwrap some hints (e.g. `Annotated`) if asked.
    """
    if unwrap:
        hint = _unwrap(hint, origin=unwrap)
    origin = tx.get_origin(hint)
    if origin is None:
        return hint
    return origin


def get_origin_uw(hint: tx.Any) -> tx.Any:
    """
    Safe `get_origin` that unwraps `Annotated` hints.

    Returns the input type, instead of `None`, if the input is not a
    generic type.
    """
    return safe_get_origin(hint, unwrap=tx.Annotated)


def safe_get_args(hint: tx.Any, unwrap: tx.Any = ()) -> tx.Tuple[tx.Any, ...]:
    """
    Safe `get_args`.

    Returns an empty tuple if the input is not a generic type.
    Can also unwrap some hints (e.g. `Annotated`) if asked.
    """
    hint = _unwrap(hint, origin=unwrap)
    return tx.get_args(hint)


def get_args_uw(hint: tx.Any) -> tx.Tuple[tx.Any, ...]:
    """
    Safe `get_args` that unwraps `Annotated` hints.

    Returns an empty tuple if the input is not a generic type.
    """
    return safe_get_args(hint, unwrap=tx.Annotated)


def eq_safenan(x: tx.Any) -> bool:
    """
    Safe equality comparison that treats NaN as equal to NaN.
    """
    if isinstance(x, (numbers.Real, np.floating)) and math.isnan(x):
        return "NaN"
    return x


def is_subscriptable(x: tx.Any) -> bool:
    if isinstance(x, type) and hasattr(x, "__class_getitem__"):
        return True
    if not isinstance(x, type) and hasattr(x, "__getitem__"):
        return True
    return False


def type2hint(x: tx.Any) -> tx.Any:
    """
    Convert a type to a type hint.

    * If the input is a type, and it does not have `__class_getitem__`,
      it is returned as is, we try to find its corresponding type hint.
    * Otherwise, the value is returned as is.
    """
    if is_subscriptable(x):
        return x

    # Look for a type hint with the same name as the type
    name = x.__name__.split(".")[-1]
    name = name.capitalize()
    if hasattr(tx, name):
        # Type / List / Tuple / ...
        return getattr(tx, name)

    # Otherwise, return the value as is
    return x
