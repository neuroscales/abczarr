"""Offline JSON-schema validation of OME-Zarr metadata.

The official OME-NGFF JSON schemas are vendored under `_ngff/<version>/`
(see `_ngff/README.md` for provenance). This module compiles them into
fast validators with [fastjsonschema](https://pypi.org/project/fastjsonschema/)
and resolves every cross-file `$ref` **locally** -- no network access, and
no dependency on an upstream schema package.

Reach a validator by version and document kind::

    >>> from abczarr.ome import schemas
    >>> validate = schemas.get_validator("0.4", "image")
    >>> validate({"multiscales": [ ... ]})           # doctest: +SKIP

or validate in one call::

    >>> schemas.validate(doc, "0.6rc0", "image")     # doctest: +SKIP

A document that does not conform raises
[SchemaValidationError][abczarr.errors.SchemaValidationError].

Versions accept either the `abczarr` spelling (`"v0_6rc0"`) or the
official NGFF string (`"0.6rc0"`, `"0.6.dev1"`). NGFF 0.2 never published
a distinct schema, so its validators use the reconstruction described in
`_ngff/README.md`.

fastjsonschema (2.x) does not implement the `minContains`/`maxContains`
bounds the schemas place on `contains` (the "2-3 space axes" rule, 0.4 on;
"at most one scale transform"), so a second pass in `_contains.py` enforces
them after the compiled validator runs -- a multiscales image with too many
or too few space axes is rejected here. That count bound is scoped to image
axes: the 0.6 schemas reuse `axes.schema` for a coordinate system's axes
too, but a general N-D coordinate system is not held to the image rule. The
one bound fastjsonschema handles
*differently* rather than ignoring is `minContains: 0` (0.6.dev1's axes
schema, permitting zero space axes): its built-in `contains` check requires
at least one match regardless, so such a document is rejected -- an
over-strict corner of a dev pre-release, independent of that second pass.
"""

__all__ = [
    "VERSIONS",
    "documents",
    "get_validator",
    "validate",
]

# stdlib
import functools
import json
import pathlib

# dependencies
import fastjsonschema
import typing_extensions as tx

# core
from abczarr.errors import SchemaValidationError
from abczarr.ome.schemas import _contains

_HERE = pathlib.Path(__file__).parent
_DATA = _HERE / "_ngff"
_HOST = "https://ngff.openmicroscopy.org"

#: the `abczarr` package suffix for each version, and the official version
#: segment that its schemas' ``$id`` carries.
_SEGMENT = {
    "v0_1": "0.1",
    "v0_2": "0.2",
    "v0_3": "0.3",
    "v0_4": "0.4",
    "v0_5": "0.5",
    "v0_6dev1": "0.6.dev1",
    "v0_6dev2": "0.6.dev2",
    "v0_6dev3": "0.6.dev3",
    "v0_6dev4": "0.6.dev4",
    "v0_6rc0": "0.6rc0",
}

#: the versions this module can validate, in the `abczarr` spelling.
VERSIONS = tuple(_SEGMENT)

# every spelling a caller might pass -> the canonical `abczarr` suffix.
_ALIASES = {}
for _suffix, _seg in _SEGMENT.items():
    _ALIASES[_suffix] = _suffix  # "v0_6rc0"
    _ALIASES[_seg] = _suffix  # "0.6rc0", "0.6.dev1"
    _ALIASES[_seg.replace(".", "_")] = _suffix  # "0_6rc0", "0_6_dev1"
del _suffix, _seg


def _canonical(version: str) -> str:
    """Resolve any accepted version spelling to its `abczarr` suffix."""
    key = str(version).strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        known = ", ".join(_SEGMENT.values())
        raise ValueError(
            f"unknown OME-NGFF version {version!r}; expected one of: {known}"
        ) from None


def _lift_misplaced_required(node: tx.Any) -> tx.Any:  # noqa: ANN401
    """Correct a known upstream typo in the 0.6.dev1/dev2 schemas.

    Their ``coordinate_transformation(s).schema`` nests a ``required``
    array *inside* a ``properties`` object (``mapAxis`` -- and, in dev1,
    ``affine``/``rotation``). A property's value can never be a bare
    array, so an eager compiler rejects it. Lift any such ``required``
    to its correct position as a sibling of ``properties``. The vendored
    files keep the upstream bytes; this runs on the in-memory copy only.
    """
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict) and isinstance(
            properties.get("required"), list
        ):
            node.setdefault("required", properties.pop("required"))
        for value in node.values():
            _lift_misplaced_required(value)
    elif isinstance(node, list):
        for value in node:
            _lift_misplaced_required(value)
    return node


@functools.lru_cache(maxsize=None)
def _registry(suffix: str) -> tx.Dict[str, tx.Any]:
    """Load one version's schemas into a ``{$ref target -> schema}`` map.

    Each file is registered under three keys so that both absolute and
    relative ``$ref``\\ s resolve: its declared ``$id``, its canonical
    ``.../<segment>/schemas/<file>`` URI (a few dev1 files declare a
    ``/latest/`` ``$id`` while their siblings reference them by path),
    and the ``/latest/`` alias.
    """
    segment = _SEGMENT[suffix]
    registry: tx.Dict[str, tx.Any] = {}
    for path in sorted((_DATA / suffix).glob("*.schema")):
        schema = _lift_misplaced_required(json.loads(path.read_text("utf-8")))
        if isinstance(schema.get("$id"), str):
            registry[schema["$id"]] = schema
        registry[f"{_HOST}/{segment}/schemas/{path.name}"] = schema
        registry[f"{_HOST}/latest/schemas/{path.name}"] = schema
    return registry


def documents(version: str) -> tx.Tuple[str, ...]:
    """The document kinds a version can validate (schema file stems).

    For example ``"image"``, ``"label"``, ``"plate"``, ``"well"``, and --
    from 0.4 on -- ``"ome"``/``"ome_zarr"``, plus the ``strict_*`` variants.
    Pass one of these as the ``document`` argument to
    [get_validator][abczarr.ome.schemas.get_validator].
    """
    suffix = _canonical(version)
    # `_`-prefixed stems are internal helper schemas referenced by `$ref`
    # (e.g. `_version`, an enum of the version string), not validatable
    # documents, so they are left out of the listing.
    stems = sorted(
        p.stem
        for p in (_DATA / suffix).glob("*.schema")
        if not p.stem.startswith("_")
    )
    return tuple(stems)


def get_validator(
    version: str, document: str
) -> tx.Callable[[tx.Any], tx.Any]:
    """Return a compiled validator for one version's *document* schema.

    Parameters
    ----------
    version : str
        The OME-NGFF version, in either spelling (``"v0_6rc0"`` or
        ``"0.6rc0"``; ``"0.6.dev1"``).
    document : str
        The document kind -- a schema file stem from
        [documents][abczarr.ome.schemas.documents], e.g. ``"image"``.

    Returns
    -------
    callable
        A function that takes a document and returns it when it conforms,
        or raises
        [SchemaValidationError][abczarr.errors.SchemaValidationError]. The
        same validator is returned for every spelling of a version.
    """
    return _compile(_canonical(version), document)


@functools.lru_cache(maxsize=None)
def _compile(
    suffix: str, document: str
) -> tx.Callable[[tx.Any], tx.Any]:
    """Compile (and cache) the validator for a canonical version + document."""
    registry = _registry(suffix)
    segment = _SEGMENT[suffix]
    root_uri = f"{_HOST}/{segment}/schemas/{document}.schema"
    if root_uri not in registry:
        available = ", ".join(documents(suffix))
        raise ValueError(
            f"no {document!r} schema for OME-NGFF {segment}; "
            f"available: {available}"
        )

    def handler(uri: str) -> tx.Any:  # noqa: ANN401
        try:
            return registry[uri]
        except KeyError:
            raise ValueError(
                f"OME-NGFF {segment} {document}: cannot resolve schema "
                f"reference {uri!r}"
            ) from None

    compiled = fastjsonschema.compile(
        registry[root_uri], handlers={"https": handler}
    )
    root_schema = registry[root_uri]
    label = f"OME-NGFF {segment} {document}"

    def _validate(instance: tx.Any) -> tx.Any:  # noqa: ANN401
        try:
            compiled(instance)
        except fastjsonschema.JsonSchemaException as exc:
            raise SchemaValidationError(
                f"{label}: {exc.message}",
                schema=label,
                path=getattr(exc, "name", None),
            ) from exc
        # fastjsonschema ignores minContains/maxContains; enforce them here.
        _contains.enforce(instance, root_schema, registry, label)
        return instance

    return _validate


def validate(instance: tx.Any, version: str, document: str) -> tx.Any:  # noqa: ANN401,E501
    """Validate *instance* against a version's *document* schema.

    Returns the instance when it conforms; raises
    [SchemaValidationError][abczarr.errors.SchemaValidationError] otherwise.
    A thin wrapper over
    [get_validator][abczarr.ome.schemas.get_validator].
    """
    return get_validator(version, document)(instance)
