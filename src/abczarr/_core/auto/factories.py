__all__ = [
    "Factory",
    "register_factory",
    "get_factory",
    "NoneFactory",
    "UnionFactory",
    "LiteralFactory",
    "TypeVarFactory",
    "SequenceFactory",
    "MappingFactory",
    "AnnotatedFactory",
]

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

# internals
from ._typing import (
    MAPPING,
    NONE,
    SEQUENCE,
    ClassDecorator,
    MagicRegistry,
    NoneType,
    T,
    UnionType,
)
from ._utils import (
    HintMagic,
    get_from_registry,
    safe_get_args,
    safe_get_origin,
    safe_isinstance,
    safe_issubclass,
)

# ======================================================================
#       BASE
# ======================================================================


class Factory(HintMagic[T]):
    """Base class for magic factories."""

    def __call__(self) -> T:
        return self.fallback()


_FACTORIES: MagicRegistry[Factory] = {}


def register_factory(*hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:
    """
    Decorator to register a factory class for one or more type hints.

    !!! example
        ```python
        @register_factory(int)
        class IntFactory(Factory[int]):
            def __call__(self) -> int:
                return 42
        ```
    """
    def decorator(cls: tx.Type[Factory]) -> tx.Type[Factory]:
        for hint in hints:
            _FACTORIES[hint] = cls
        return cls

    return decorator


def get_factory(
    hint: tx.Any,
    registry: MagicRegistry[Factory] = _FACTORIES,
    fallback: tx.Optional[tx.Type[Factory]] = Factory
) -> tx.Callable[[], T]:
    """
    Get the best-matching factory function for a given type hint.
    """
    factory_cls = get_factory_class(hint, registry, fallback)
    return factory_cls(hint)


def get_factory_class(
    hint: tx.Any,
    registry: MagicRegistry[Factory] = _FACTORIES,
    fallback: tx.Optional[tx.Type[Factory]] = Factory
) -> tx.Type[Factory]:
    """
    Get the best-matching factory class for a given type hint.
    """
    factory_cls = get_from_registry(hint, registry) or fallback
    if hasattr(factory_cls, "__class_getitem__"):
        factory_cls = factory_cls[hint]
    return factory_cls


# ======================================================================
#       IMPL
# ======================================================================


@register_factory(NoneType)
class NoneFactory(Factory[NONE]):

    DEFAULT = NoneType

    def __call__(self) -> NONE:
        return None


@register_factory(tx.Union, UnionType)
class UnionFactory(Factory[T]):

    DEFAULT = tx.Union

    def __call__(self) -> T:
        if NoneType in self.args:
            return None
        for arg in self.args:
            try:
                factory = get_factory(arg)
                return factory()
            except TypeError:
                continue
        raise TypeError(
            "Cannot create an instance of any of the union types: "
            f"{' | '.join(str(arg) for arg in self.args)}"
        )


@register_factory(tx.Literal)
class LiteralFactory(Factory[T]):

    DEFAULT = tx.Literal

    def __call__(self) -> T:
        if not self.args:
            raise TypeError("Cannot create an instance of an empty literal")
        if None in self.args:
            return None
        return self.args[0]


@register_factory(tx.TypeVar)
class TypeVarFactory(Factory[T]):

    DEFAULT = tx.TypeVar("T")

    def __call__(self) -> T:
        return get_factory(self.fallback)()


@register_factory(abc.Sequence)
class SequenceFactory(Factory[SEQUENCE]):

    DEFAULT = abc.Sequence
    FALLBACK = list


@register_factory(abc.Mapping)
class MappingFactory(Factory[MAPPING]):

    DEFAULT = abc.Mapping
    FALLBACK = dict


@register_factory(tx.Annotated)
class AnnotatedFactory(Factory[T]):

    _REGISTRY: MagicRegistry[Factory] = {}

    @classmethod
    def register(cls, *hints: tx.Unpack[tx.Tuple[tx.Any]]) -> ClassDecorator:

        def decorator(factory_cls: tx.Type[Factory]) -> tx.Type[Factory]:
            for hint in hints:
                cls._REGISTRY[hint] = factory_cls
            return factory_cls

        return decorator

    @classmethod
    def _get_factory(cls, hint: tx.Any) -> tx.Optional[tx.Type[Factory]]:
        return get_factory(hint, registry=cls._REGISTRY, fallback=None)

    @property
    def factories(self) -> tx.Tuple[Factory, ...]:
        origin = safe_get_origin(self.hint, unwrap=tx.Annotated)

        factories = []
        for arg in safe_get_args(self.hint):
            if safe_issubclass(arg, Factory):
                arg = arg(origin)
            if not isinstance(arg, Factory):
                # Look into annotation registry
                arg = self._get_factory(arg)
            if safe_isinstance(arg, Factory):
                factories.append(arg)

        factories.insert(0, get_factory(origin))
        return tuple(factories)

    def __call__(self) -> T:
        for factory in reversed(self.factories):
            return factory()
        raise TypeError(f"Cannot instantiate value for {self.hint}")
