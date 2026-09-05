"""Offline JSON-schema validation of Zarr array/group metadata.

abczarr authors the Zarr v1/v2/v3 *core* metadata schemas (the Zarr spec
publishes none) under `_zarr/{v1,v2,v3/core}/`, and vendors the official
Zarr v3 *extension* schemas under `_zarr/v3/extensions/` (see
`_zarr/README.md`). The v3 array schema composes the vendored codec and
data-type schemas, so a v3 `zarr.json` is validated against the authored
core *and* the official extension definitions -- all offline, with no
network and no upstream package.

Reach a validator by version and document kind::

    >>> from abczarr import schemas
    >>> validate = schemas.get_validator("v3", "array")
    >>> validate({"zarr_format": 3, "node_type": "array"})  # doctest: +SKIP

or in one call::

    >>> schemas.validate(doc, "v2", "array")            # doctest: +SKIP

A document that does not conform raises
[SchemaValidationError][abczarr.errors.SchemaValidationError].

Versions accept `"v1"`/`"v2"`/`"v3"` or the bare number `"1"`/`"2"`/`"3"`.
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

_HERE = pathlib.Path(__file__).parent
_DATA = _HERE / "_zarr"
_EXTENSIONS = _DATA / "v3" / "extensions"
_RAW = (
    "https://raw.githubusercontent.com/zarr-developers/"
    "zarr-extensions/refs/heads/main/"
)

#: the version this module validates -> the directory holding its authored
#: array/group schemas.
_VERSION_DIR = {
    "v1": _DATA / "v1",
    "v2": _DATA / "v2",
    "v3": _DATA / "v3" / "core",
}

#: the versions this module can validate.
VERSIONS = tuple(_VERSION_DIR)

_ALIASES = {}
for _v in _VERSION_DIR:
    _ALIASES[_v] = _v  # "v3"
    _ALIASES[_v[1:]] = _v  # "3"
del _v


def _canonical(version: str) -> str:
    """Resolve any accepted version spelling to its `abczarr` suffix."""
    key = str(version).strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        known = ", ".join(VERSIONS)
        raise ValueError(
            f"unknown Zarr version {version!r}; expected one of: {known}"
        ) from None


def _normalize_extension(node: tx.Any) -> tx.Any:  # noqa: ANN401
    """Correct three known upstream defects in the vendored extension schemas.

    The vendored bytes are untouched; this runs on the in-memory copy:

    - draft-2020-12 ``prefixItems`` (which fastjsonschema does not implement
      and would silently ignore) becomes the equivalent draft-07 tuple form
      -- ``items`` as a list, plus ``additionalItems`` for any ``items``
      "rest" schema -- so the tuple constraint is actually enforced;
    - a ``type`` whose value is a JSON pointer or URL (a plain typo for
      ``$ref``; meaningless as a type) is read as the ``$ref`` it was meant
      to be;
    - the custom ``"format": "uint"`` annotation (the rectilinear chunk
      grid), which older fastjsonschema rejects at *compile* time as an
      unknown format. A ``format`` is only an annotation, and the field's
      ``"type": "integer"`` with ``"minimum"`` already carries the
      unsignedness, so dropping it enforces the same constraint everywhere.
    """
    if isinstance(node, dict):
        if "prefixItems" in node:
            tuple_items = node.pop("prefixItems")
            rest = node.pop("items", None)
            node["items"] = tuple_items
            if rest is not None:
                node.setdefault("additionalItems", rest)
        kind = node.get("type")
        if isinstance(kind, str) and (
            kind.startswith("#") or kind.startswith("http")
        ):
            node.pop("type")
            node.setdefault("$ref", kind)
        if node.get("format") == "uint":
            node.pop("format")
        for value in node.values():
            _normalize_extension(value)
    elif isinstance(node, list):
        for value in node:
            _normalize_extension(value)
    return node


@functools.lru_cache(maxsize=None)
def _extension_registry() -> tx.Dict[str, tx.Any]:
    """The vendored Zarr v3 extension schemas, keyed by their raw-URL ``$ref``.

    Each schema's ``$id``-less cross-references use absolute
    ``raw.githubusercontent.com/.../<category>/<name>/schema.json`` URLs; the
    key mirrors that path so both the authored v3 array schema and the
    extensions themselves resolve against the vendored files.
    """
    registry: tx.Dict[str, tx.Any] = {}
    for path in _EXTENSIONS.rglob("schema.json"):
        rel = path.relative_to(_EXTENSIONS).as_posix()
        registry[_RAW + rel] = _normalize_extension(
            json.loads(path.read_text("utf-8"))
        )
    return registry


def documents(version: str) -> tx.Tuple[str, ...]:
    """The document kinds a version validates: ``"array"`` and ``"group"``."""
    suffix = _canonical(version)
    stems = sorted(p.stem for p in _VERSION_DIR[suffix].glob("*.schema"))
    return tuple(stems)


def get_validator(
    version: str, document: str
) -> tx.Callable[[tx.Any], tx.Any]:
    """Return a compiled validator for a version's *document* schema.

    Parameters
    ----------
    version : str
        The Zarr version -- ``"v1"``/``"v2"``/``"v3"``, or the bare
        number ``"1"``/``"2"``/``"3"``.
    document : str
        ``"array"`` or ``"group"`` (see
        [documents][abczarr.schemas.documents]).

    Returns
    -------
    callable
        A function that returns a conforming document, or raises
        [SchemaValidationError][abczarr.errors.SchemaValidationError]. The
        same validator is returned for every spelling of a version.
    """
    return _compile(_canonical(version), document)


@functools.lru_cache(maxsize=None)
def _compile(
    suffix: str, document: str
) -> tx.Callable[[tx.Any], tx.Any]:
    """Compile (and cache) the validator for a canonical version + document."""
    path = _VERSION_DIR[suffix] / f"{document}.schema"
    if not path.exists():
        available = ", ".join(documents(suffix))
        raise ValueError(
            f"no {document!r} schema for Zarr {suffix}; "
            f"available: {available}"
        )
    registry = _extension_registry()

    def handler(uri: str) -> tx.Any:  # noqa: ANN401
        try:
            return registry[uri]
        except KeyError:
            raise ValueError(
                f"Zarr {suffix} {document}: cannot resolve schema "
                f"reference {uri!r}"
            ) from None

    compiled = fastjsonschema.compile(
        json.loads(path.read_text("utf-8")), handlers={"https": handler}
    )
    label = f"Zarr {suffix} {document}"

    def _validate(instance: tx.Any) -> tx.Any:  # noqa: ANN401
        try:
            return compiled(instance)
        except fastjsonschema.JsonSchemaException as exc:
            raise SchemaValidationError(
                f"{label}: {exc.message}",
                schema=label,
                path=getattr(exc, "name", None),
            ) from exc

    return _validate


def validate(instance: tx.Any, version: str, document: str) -> tx.Any:  # noqa: ANN401,E501
    """Validate *instance* against a version's *document* schema.

    Returns the instance when it conforms; raises
    [SchemaValidationError][abczarr.errors.SchemaValidationError] otherwise.
    A thin wrapper over [get_validator][abczarr.schemas.get_validator].
    """
    return get_validator(version, document)(instance)
