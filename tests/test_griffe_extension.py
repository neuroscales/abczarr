"""The documentation griffe extension synthesizes correct signatures.

The extension in ``docs/extensions/griffe_abczarr.py`` rebuilds the constructor
signature of every attrs class so the API reference renders each field with its
real default. These tests load the package through griffe with the extension
and check each synthesized ``__init__`` against the class's real constructor.

Griffe is a documentation-only dependency, so these tests are skipped where it
is not installed.
"""

import importlib.util
import inspect
import pathlib

import pytest

griffe = pytest.importorskip("griffe")

import attrs  # noqa: E402  (after the griffe skip guard)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_EXTENSION = _ROOT / "docs" / "extensions" / "griffe_abczarr.py"

# The inspect parameter kind each griffe ParameterKind value maps back to.
_INSPECT_KIND = {
    "positional-only": inspect.Parameter.POSITIONAL_ONLY,
    "positional or keyword": inspect.Parameter.POSITIONAL_OR_KEYWORD,
    "keyword-only": inspect.Parameter.KEYWORD_ONLY,
    "variadic positional": inspect.Parameter.VAR_POSITIONAL,
    "variadic keyword": inspect.Parameter.VAR_KEYWORD,
}


def _load_extension() -> object:
    spec = importlib.util.spec_from_file_location(
        "griffe_abczarr", _EXTENSION
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package(extension: object) -> object:
    return griffe.load(
        "abczarr",
        search_paths=[str(_ROOT / "src")],
        extensions=griffe.Extensions(extension.AttrsExtension()),
    )


def _synthesized(module: object) -> list:
    """Every class the extension gave a synthesized ``__init__``."""
    classes = []
    seen: set = set()

    def walk(obj: object) -> None:
        if obj.canonical_path in seen:
            return
        seen.add(obj.canonical_path)
        if obj.is_class:
            init = obj.members.get("__init__")
            if isinstance(init, griffe.Function) and init.lineno == 0:
                classes.append(obj)
        for member in obj.members.values():
            if not member.is_alias and (member.is_module or member.is_class):
                walk(member)

    walk(module)
    return classes


def test_every_synthesized_signature_matches_the_real_init() -> None:
    extension = _load_extension()
    package = _package(extension)
    classes = _synthesized(package)
    assert classes, "the extension synthesized no signatures"
    for class_ in classes:
        cls = extension._runtime_object(class_)
        if cls is None or not attrs.has(cls):
            continue
        real = [
            (param.name, param.kind)
            for param in inspect.signature(cls.__init__).parameters.values()
        ]
        ours = [
            (param.name, _INSPECT_KIND[param.kind.value])
            for param in class_.members["__init__"].parameters
        ]
        assert ours == real, class_.path


def test_a_keyword_only_field_keeps_its_position() -> None:
    # Regression: the extension once emitted parameters in attrs-field order
    # rather than in the constructor's real order and kind.
    extension = _load_extension()
    package = _package(extension)
    init = package["metadata.v3.array.ArrayMetadata"].members["__init__"]
    kinds = {p.name: p.kind.value for p in init.parameters}
    assert kinds["shape"] == "keyword-only"
    # extra_items is the keyword-only catch-all, and sorts to the end.
    assert kinds["extra_items"] == "keyword-only"
    assert list(init.parameters)[-1].name == "extra_items"
    # the catch-all field is typed rather than left blank
    extra = next(p for p in init.parameters if p.name == "extra_items")
    assert extra.annotation == "Mapping[str, Any]"


def test_a_real_default_is_rendered() -> None:
    extension = _load_extension()
    package = _package(extension)
    init = package["api.config.ArrayConfig"].members["__init__"]
    defaults = {p.name: p.default for p in init.parameters}
    assert defaults["chunks"] == "'auto'"
    assert defaults["max_chunk_bytes"] == "8388608"
