"""Enforce the ``contains`` count bounds fastjsonschema does not implement.

fastjsonschema (2.x) checks that an array ``contains`` at least one item
matching a subschema, but silently ignores the ``minContains`` /
``maxContains`` bounds on that count. The OME-NGFF schemas rely on those
bounds for the "2-3 space axes" rule (0.4 on) and "at most one scale
transform", so a document with the wrong number of axes would otherwise
pass. This module runs a second pass, after fastjsonschema has validated
structure, that enforces exactly those bounds.

It is **not** a general JSON-schema engine. It walks ``(schema, instance)``
in parallel -- resolving ``$ref`` (local pointer or vendored registry URI)
and descending ``properties`` / ``items`` / ``allOf`` -- and acts only on
the two shapes the vendored schemas actually use to bound a count:

* a node carrying ``contains`` together with ``minContains`` / ``maxContains``
  (0.4/0.5 ``image``: 2-3 space axes, at most one scale transform); and
* a ``oneOf`` / ``anyOf`` whose every branch is a bare count-constraint
  (``contains`` + bounds and nothing else), where the branch a document
  satisfies is decided *by* those bounds (0.6 ``axes``).

Any other combinator is left untouched -- fastjsonschema has already checked
it structurally, so leaving it alone can never reject a valid document; at
worst a count bound in a shape not listed above would stay unenforced, and
the ``test_no_new_unsupported_keywords`` guard flags a new one creeping in.

**Scope.** The "2-3 space axes" bound is the image-axis rule, but the
vendored 0.6 schemas reuse the shared ``axes.schema`` file for a *coordinate
system*'s axes too -- so a faithful reading of them (and the reference
validator's) rejects a 1-D or 4-D coordinate system in a transformation
document. RFC-5 scopes the 2-3 rule to axes "inside multiscales metadata"
only and leaves a general coordinate system's dimensionality unbounded, so
this is an upstream over-constraint (see ``_ngff/README.md``). The
enforcement is therefore suppressed once a ``$ref`` crosses into the shared
``axes.schema`` file -- that file *is* a coordinate system's axes. An image's
own axis bound lives in ``image.schema``'s local ``$defs/axes`` (0.4/0.5) and
is reached without crossing into ``axes.schema``, so it stays enforced. Only
the definition site of the bound decides this, not the instance property name
-- so ``arrayCoordinateSystem`` axes and any future container are covered
too.
"""

import functools
import json
from urllib.parse import urljoin

import fastjsonschema
import typing_extensions as tx

from abczarr.errors import SchemaValidationError

# The keywords a "bare count-constraint" branch may carry. A branch with
# anything else (its own ``properties``, a nested ``oneOf``, ...) is not one
# whose selection is decided purely by the count, so it is left to
# fastjsonschema rather than second-guessed here.
_COUNT_ONLY = frozenset(
    {"contains", "minContains", "maxContains", "$comment", "title",
     "description"}
)


@functools.lru_cache(maxsize=None)
def _matcher(subschema_json: str) -> tx.Callable[[tx.Any], tx.Any]:
    """Compile a ``contains`` subschema into a match predicate.

    The vendored ``contains`` subschemas carry no ``$ref``, so each compiles
    standalone. Cached on the schema's canonical JSON text.
    """
    return fastjsonschema.compile(json.loads(subschema_json))


def _matches(subschema: tx.Any, item: tx.Any) -> bool:
    try:
        _matcher(json.dumps(subschema, sort_keys=True))(item)
        return True
    except fastjsonschema.JsonSchemaException:
        return False


def _count(subschema: tx.Any, array: tx.List[tx.Any]) -> int:
    return sum(1 for item in array if _matches(subschema, item))


def _in_bounds(node: tx.Mapping, array: tx.List[tx.Any]) -> bool:
    """Whether *array*'s matching-item count satisfies *node*'s bounds."""
    count = _count(node["contains"], array)
    low = node.get("minContains", 1)
    high = node.get("maxContains")
    return count >= low and (high is None or count <= high)


def _is_count_branch(node: tx.Any) -> bool:
    return (
        isinstance(node, dict)
        and "contains" in node
        and set(node).issubset(_COUNT_ONLY)
        and ("minContains" in node or "maxContains" in node)
    )


def _pointer(doc: tx.Any, fragment: str) -> tx.Any:
    """Resolve a JSON-pointer *fragment* (``/$defs/axes``) within *doc*."""
    node = doc
    for token in fragment.split("/"):
        if not token:
            continue
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            node = node.get(token)
        elif isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def _deref(
    ref: str, registry: tx.Mapping[str, tx.Any], base: tx.Mapping
) -> tx.Tuple[tx.Any, tx.Mapping]:
    """Resolve a ``$ref`` to ``(schema, new_base_document)``.

    A local ``#/...`` pointer resolves within *base*; an absolute or
    relative URI resolves through the vendored *registry* (relative refs are
    joined onto *base*'s ``$id``). An unresolved ref yields ``(None, base)``
    -- the walk simply stops there rather than guessing.
    """
    uri, _, fragment = ref.partition("#")
    if not uri:
        return _pointer(base, fragment), base
    doc = registry.get(uri)
    if doc is None and base.get("$id"):
        doc = registry.get(urljoin(base["$id"], uri))
    if doc is None:
        return None, base
    return (_pointer(doc, fragment) if fragment else doc), doc


def _fail(label: str, message: str) -> "tx.NoReturn":
    raise SchemaValidationError(f"{label}: {message}", schema=label)


def _is_axes_ref(new_base: tx.Mapping, base: tx.Mapping) -> bool:
    """Whether a ``$ref`` crossed into the shared ``axes.schema`` file.

    That file defines a *coordinate system*'s axes. Its 2-3 space-axis bound
    is the image rule (``image.schema``'s own ``$defs/axes`` in 0.4/0.5) and
    does not apply to a general coordinate system, so the count bound is not
    enforced below a ref that reaches ``axes.schema``. A local ``#/...`` ref
    stays within the same document (``new_base is base``) and so an image's
    own inline axes bound is unaffected.
    """
    if new_base is base:
        return False
    uri = new_base.get("$id", "")
    return isinstance(uri, str) and uri.rsplit("/", 1)[-1] == "axes.schema"


def _walk(
    schema: tx.Any,
    instance: tx.Any,
    registry: tx.Mapping[str, tx.Any],
    base: tx.Mapping,
    label: str,
    seen: tx.FrozenSet[tx.Tuple[int, int]],
    enforce_counts: bool = True,
) -> None:
    if not isinstance(schema, dict):
        return

    ref = schema.get("$ref")
    if isinstance(ref, str):
        target, new_base = _deref(ref, registry, base)
        key = (id(target), id(instance))
        if target is not None and key not in seen:
            # Crossing into the shared axes.schema means these are a
            # coordinate system's axes, not an image's, so the count bound
            # (the image rule) is not enforced below here.
            child = enforce_counts and not _is_axes_ref(new_base, base)
            _walk(target, instance, registry, new_base, label,
                  seen | {key}, child)

    if enforce_counts and isinstance(instance, list):
        if "contains" in schema and (
            "minContains" in schema or "maxContains" in schema
        ):
            if not _in_bounds(schema, instance):
                count = _count(schema["contains"], instance)
                low = schema.get("minContains", 1)
                high = schema.get("maxContains")
                if count < low:
                    _fail(label, "too few items match a required subschema "
                                 f"(need at least {low}, found {count})")
                _fail(label, "too many items match a bounded subschema "
                             f"(allowed at most {high}, found {count})")
        for keyword, want_one in (("oneOf", True), ("anyOf", False)):
            branches = schema.get(keyword)
            if isinstance(branches, list) and all(
                _is_count_branch(b) for b in branches
            ):
                matched = sum(_in_bounds(b, instance) for b in branches)
                ok = matched == 1 if want_one else matched >= 1
                if not ok:
                    expected = ("exactly one" if want_one else "at least one")
                    _fail(label, "the axis counts satisfy "
                                 f"{matched} of the {keyword} constraints "
                                 f"({expected} is required)")

    properties = schema.get("properties")
    if isinstance(properties, dict) and isinstance(instance, dict):
        for name, subschema in properties.items():
            if name in instance:
                _walk(subschema, instance[name], registry, base, label,
                      seen, enforce_counts)

    items = schema.get("items")
    if isinstance(items, dict) and isinstance(instance, list):
        for item in instance:
            _walk(items, item, registry, base, label, seen, enforce_counts)

    for subschema in schema.get("allOf", []):
        _walk(subschema, instance, registry, base, label, seen, enforce_counts)


def enforce(
    instance: tx.Any,
    root: tx.Mapping,
    registry: tx.Mapping[str, tx.Any],
    label: str,
) -> None:
    """Enforce the vendored schemas' ``contains`` count bounds on *instance*.

    Raises
    ------
    SchemaValidationError
        When a ``minContains`` / ``maxContains`` bound is violated. Returns
        ``None`` when every bound is satisfied (or the schema declares none).
    """
    _walk(root, instance, registry, root, label, frozenset())
