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

A dataset's transformation also names the array it maps from in ``input``
and the coordinate system it maps to in ``output``. The vendored schemas
require these references in two forms:

* ``0.6.dev1`` and ``0.6.dev3`` require both references on every dataset
  transformation, as strings. The ``input`` string is the dataset path, and
  the ``output`` string is the name of a coordinate system.
* ``0.6.dev2`` requires both strings on a ``sequence`` only. A lone ``scale``
  may omit them.
* From ``0.6.dev4`` on, both references are required objects. ``input``
  carries the dataset path in ``path``, and ``output`` carries the name of a
  coordinate system in ``name``.

The version packages annotate a dataset's transformation with
`ToDatasetTransformation`, so the rules are checked whenever a dataset is
built, whether from JSON, from keyword arguments, or by a conversion between
versions.
"""

# stdlib
from collections import abc

# dependencies
import typing_extensions as tx

# core
from abczarr._core.metadata import MetadataConverter
from abczarr._core.rfc2119 import MISSING

#: The step types a dataset's ``sequence`` may hold.
_SEQUENCE_STEPS = ("scale", "translation")

#: For each 0.6 version, the single-transformation types a dataset may carry
#: besides ``"sequence"``. ``None`` marks a version that puts no restriction
#: on the number or the types of a dataset's transformations. The version
#: packages spell the same types in their ``DatasetTransformation``
#: annotation.
TYPES: tx.Dict[str, tx.Optional[tx.Tuple[str, ...]]] = {
    "0.6.dev1": None,
    "0.6.dev2": ("scale",),
    "0.6.dev3": ("scale", "identity"),
    "0.6.dev4": ("scale", "identity"),
    "0.6rc0": ("scale", "identity"),
    "0.6": ("scale", "identity"),
}

#: The references are strings, and every dataset transformation needs them.
STRING = "string"
#: The references are strings, and only a ``sequence`` needs them.
SEQUENCE_STRING = "sequence-string"
#: The references are ``{"path": ...}`` and ``{"name": ...}`` objects, and
#: every dataset transformation needs them.
OBJECT = "object"

#: For each 0.6 version, how a dataset transformation spells its ``input``
#: and ``output`` references. The version packages pass the same form to
#: their ``DatasetTransformation`` annotation.
REFS: tx.Dict[str, str] = {
    "0.6.dev1": STRING,
    "0.6.dev2": SEQUENCE_STRING,
    "0.6.dev3": STRING,
    "0.6.dev4": OBJECT,
    "0.6rc0": OBJECT,
    "0.6": OBJECT,
}


class ToDatasetTransformation(MetadataConverter):
    """Converts a dataset's coordinate transformation and checks its shape.

    The value is first converted to the annotated transformation class, as
    any metadata field is. A conversion error is raised for a
    transformation whose type is not one of `types` or ``"sequence"``, and
    for a ``sequence`` that does not hold exactly two ``scale`` or
    ``translation`` steps. When `types` is `None`, any type is accepted.
    A conversion error is also raised when an ``input`` or ``output``
    reference that `refs` requires is missing or has the wrong form.
    """

    def __init__(
        self,
        types: tx.Optional[tx.Sequence[str]],
        refs: str,
        *args: tx.Any,
        **kwargs: tx.Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.types = None if types is None else tuple(types)
        self.refs = refs

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.types!r}, {self.refs!r})"

    def __call__(self, value: tx.Any) -> tx.Any:
        value = super().__call__(value)
        problem = check(value, self.types, self.refs)
        if problem is not None:
            raise self.value_error(value, problem)
        return value


def check(
    transform: tx.Any,
    types: tx.Optional[tx.Sequence[str]],
    refs: tx.Optional[str] = None,
) -> tx.Optional[str]:
    """Describe why `transform` cannot be a dataset's transformation.

    `transform` is a transformation object or its JSON form. `types` lists
    the single-transformation types the version allows besides
    ``"sequence"``, or is `None` when any type is allowed. `refs` is the
    form of the ``input`` and ``output`` references, as listed in `REFS`.
    `None` is returned when the transformation is allowed.
    """
    problem = _check_type(transform, types)
    if problem is None and refs is not None:
        problem = check_refs(transform, refs)
    return problem


def _check_type(
    transform: tx.Any, types: tx.Optional[tx.Sequence[str]]
) -> tx.Optional[str]:
    if types is None:
        return None
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


def check_refs(transform: tx.Any, refs: str) -> tx.Optional[str]:
    """Describe why the references of `transform` do not have form `refs`.

    `transform` is a transformation object or its JSON form. `None` is
    returned when its ``input`` and ``output`` references have the form
    that `refs` requires.
    """
    if refs == SEQUENCE_STRING and _get(transform, "type") != "sequence":
        return None
    if refs in (STRING, SEQUENCE_STRING):
        for side in ("input", "output"):
            ref = _get(transform, side)
            if not isinstance(ref, str):
                shown = "nothing" if _missing(ref) else repr(_json(ref))
                return (
                    f"A dataset's coordinate transformation must name its "
                    f"{side} with a string, not {shown}."
                )
        return None
    if not isinstance(_get(_get(transform, "input"), "path"), str):
        return (
            "A dataset's coordinate transformation must give the path of "
            "its array in input.path."
        )
    if not isinstance(_get(_get(transform, "output"), "name"), str):
        return (
            "A dataset's coordinate transformation must give the name of "
            "its coordinate system in output.name."
        )
    return None


def check_dataset(
    transforms: tx.Sequence[tx.Any], version: str, refs: bool = True
) -> tx.Optional[str]:
    """Describe why `transforms` cannot be a dataset's transformations.

    `transforms` lists a dataset's transformations, as objects or in their
    JSON form. `None` is returned when `version` allows them. The
    ``input`` and ``output`` references are checked only when `refs` is
    true.
    """
    types = TYPES[version]
    if types is not None and len(transforms) != 1:
        return (
            f"A dataset must carry exactly one coordinate transformation, "
            f"not {len(transforms)}."
        )
    form = REFS[version] if refs else None
    for transform in transforms:
        problem = check(transform, types, form)
        if problem is not None:
            return problem
    return None


def dataset_refs(
    version: str, kind: str, path: str, system: str
) -> tx.Dict[str, tx.Any]:
    """Return the ``input`` and ``output`` references of a dataset.

    The references point from the array at `path` to the coordinate system
    named `system`, in the form that `version` uses for a transformation of
    type `kind`. An empty mapping is returned when the version needs no
    references on a transformation of that type.
    """
    form = REFS[version]
    if form == SEQUENCE_STRING and kind != "sequence":
        return {}
    if form == OBJECT:
        return {"input": {"path": path}, "output": {"name": system}}
    return {"input": path, "output": system}


def _get(obj: tx.Any, key: str) -> tx.Any:
    if isinstance(obj, abc.Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _missing(value: tx.Any) -> bool:
    return value is None or value is MISSING


def _json(value: tx.Any) -> tx.Any:
    to_json = getattr(value, "to_json", None)
    return to_json() if callable(to_json) else value


def _spell(kinds: tx.Sequence[str]) -> str:
    quoted = [repr(k) for k in kinds]
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]
