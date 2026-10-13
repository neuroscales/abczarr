"""The shape of a 0.6 dataset's coordinate transformations.

From ``0.6.dev2`` on, each dataset of a multiscale carries exactly one
coordinate transformation, which maps the dataset's array into a coordinate
system of the multiscale. That transformation is a ``scale``, an ``identity``
(from ``0.6.dev3`` on), or a ``sequence`` of exactly two steps, each of which
is a ``scale`` or a ``translation``.

The spec text describes the sequence as a scale followed by a translation,
but the NGFF schema does not constrain the order of the two steps. The model
follows the schema, so a translation-before-scale sequence found in the wild
still loads.

The version packages annotate a dataset's transformation with
`ToDatasetTransformation`, so the rule is checked whenever a dataset is built,
whether from JSON, from keyword arguments, or by a conversion between
versions.
"""

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

# core
from abczarr._core.metadata import MetadataConverter

#: The step types a dataset's ``sequence`` may hold.
_SEQUENCE_STEPS = ("scale", "translation")

#: For each 0.6 version, the single-transformation types a dataset may carry
#: besides ``"sequence"``. ``None`` marks a version that puts no restriction
#: on a dataset's transformations. The version packages spell the same types
#: in their ``DatasetTransformation`` annotation.
TYPES: tx.Dict[str, tx.Optional[tx.Tuple[str, ...]]] = {
    "0.6.dev1": None,
    "0.6.dev2": ("scale",),
    "0.6.dev3": ("scale", "identity"),
    "0.6.dev4": ("scale", "identity"),
    "0.6rc0": ("scale", "identity"),
    "0.6": ("scale", "identity"),
}


class ToDatasetTransformation(MetadataConverter):
    """Converts a dataset's coordinate transformation and checks its shape.

    The value is first converted to the annotated transformation class, as
    any metadata field is. A transformation whose type is not one of `types`
    or ``"sequence"``, and a ``sequence`` that does not hold exactly two
    ``scale`` or ``translation`` steps, raise a conversion error.
    """

    def __init__(
        self, types: tx.Sequence[str], *args: tx.Any, **kwargs: tx.Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.types = tuple(types)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.types!r})"

    def __call__(self, value: tx.Any) -> tx.Any:
        value = super().__call__(value)
        problem = check(value, self.types)
        if problem is not None:
            raise self.value_error(value, problem)
        return value


def check(transform: tx.Any, types: tx.Sequence[str]) -> tx.Optional[str]:
    """Describe why `transform` cannot be a dataset's transformation.

    `transform` is a transformation object or its JSON form. `types` lists
    the single-transformation types the version allows besides
    ``"sequence"``. `None` is returned when the transformation is allowed.
    """
    allowed = tuple(types) + ("sequence",)
    kind = _get(transform, "type")
    if kind not in allowed:
        return (
            f"A dataset's coordinate transformation must be of type "
            f"{_spell(allowed)}, not {kind!r}."
        )
    if kind != "sequence":
        return None
    steps = _get(transform, "transformations")
    kinds = [_get(step, "type") for step in steps or ()]
    if len(kinds) != 2 or any(k not in _SEQUENCE_STEPS for k in kinds):
        return (
            f"A dataset's sequence must hold exactly two transformations, "
            f"each of type {_spell(_SEQUENCE_STEPS)}, not {kinds!r}."
        )
    return None


def check_dataset(
    transforms: tx.Sequence[tx.Any], version: str
) -> tx.Optional[str]:
    """Describe why `transforms` cannot be a dataset's transformations.

    `transforms` lists a dataset's transformations, as objects or in their
    JSON form. `None` is returned when `version` allows them.
    """
    types = TYPES[version]
    if types is None:
        return None
    if len(transforms) != 1:
        return (
            f"A dataset must carry exactly one coordinate transformation, "
            f"not {len(transforms)}."
        )
    return check(transforms[0], types)


def _get(obj: tx.Any, key: str) -> tx.Any:
    if isinstance(obj, abc.Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _spell(kinds: tx.Sequence[str]) -> str:
    quoted = [repr(k) for k in kinds]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]
