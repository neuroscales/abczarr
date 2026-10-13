"""The input and output references of a 0.6 dataset's transformation (#202).

The vendored ``image.schema`` of each 0.6 version requires a dataset's
transformation to name the array it maps from and the coordinate system it
maps to:

* 0.6.dev1 and 0.6.dev3 require both references, as strings, on every
  dataset transformation.
* 0.6.dev2 requires both strings on a ``sequence`` only.
* 0.6.dev4, 0.6rc0 and 0.6 require ``input.path`` and ``output.name``.

Up to 0.6.dev3, the ``input`` string is the dataset path and the ``output``
string is the coordinate system name. Conversions rewrite one form into the
other, and the writers emit the form of the version they write.
"""

import importlib
import types
import warnings

import pytest
import typing_extensions as tx

from abczarr.ome import schemas, v0_5
from abczarr.ome.pyramid import _level_dataset, _RefStyle

VERSIONS = ["0.6.dev1", "0.6.dev2", "0.6.dev3", "0.6.dev4", "0.6rc0", "0.6"]
STRING_FORM = VERSIONS[:3]
OBJECT_FORM = VERSIONS[3:]

_PACKAGES = {
    "0.6.dev1": "v0_6dev1",
    "0.6.dev2": "v0_6dev2",
    "0.6.dev3": "v0_6dev3",
    "0.6.dev4": "v0_6dev4",
    "0.6rc0": "v0_6rc0",
    "0.6": "v0_6",
}

_AXES = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
_SCALE = {"type": "scale", "scale": [2.0, 2.0]}
_TRANSLATION = {"type": "translation", "translation": [0.5, 0.5]}
_STRINGS = {"input": "0", "output": "physical"}
_OBJECTS = {"input": {"path": "0"}, "output": {"name": "physical"}}


def _pkg(version: str) -> types.ModuleType:
    return importlib.import_module("abczarr.ome." + _PACKAGES[version])


def _refs(version: str) -> tx.Dict[str, tx.Any]:
    return dict(_STRINGS if version in STRING_FORM else _OBJECTS)


def _sequence(**refs: tx.Any) -> tx.Dict[str, tx.Any]:
    return {
        "type": "sequence",
        "transformations": [_SCALE, _TRANSLATION],
        **refs,
    }


def _multiscale_doc(
    transform: tx.Any, systems: tx.Sequence[str] = ("physical",)
) -> tx.Dict[str, tx.Any]:
    return {
        "coordinateSystems": [{"name": s, "axes": _AXES} for s in systems],
        "datasets": [{"path": "0", "coordinateTransformations": [transform]}],
    }


def _multiscale(version: str, transform: tx.Any, **kw: tx.Any) -> tx.Any:
    return _pkg(version).Multiscale.from_json(_multiscale_doc(transform, **kw))


def _transform(ms: tx.Any) -> tx.Dict[str, tx.Any]:
    (dataset,) = ms.datasets
    (transform,) = dataset.coordinateTransformations
    return transform.to_json()


# --------------------------------------------------------------------------
#   the model agrees with the vendored schema
# --------------------------------------------------------------------------

#: Reference spellings, each tried on a lone scale and on a sequence.
_SPELLINGS = {
    "strings": _STRINGS,
    "objects": _OBJECTS,
    "no references": {},
    "no input": {"output": _STRINGS["output"]},
    "no output": {"input": _STRINGS["input"]},
    "no input object": {"output": _OBJECTS["output"]},
    "no output object": {"input": _OBJECTS["input"]},
    "input names a system": {
        "input": {"name": "array"},
        "output": {"name": "physical"},
    },
    "output gives a path only": {
        "input": {"path": "0"},
        "output": {"path": "physical"},
    },
    "string input, object output": {
        "input": "0",
        "output": {"name": "physical"},
    },
}


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("kind", ["scale", "sequence"])
@pytest.mark.parametrize("spelling", list(_SPELLINGS))
def test_model_agrees_with_the_vendored_schema(
    version: str, kind: str, spelling: str
) -> None:
    refs = _SPELLINGS[spelling]
    transform = (
        _sequence(**refs)
        if kind == "sequence"
        else {
            **_SCALE,
            **refs,
        }
    )
    doc = {
        "ome": {
            "version": version,
            "multiscales": [_multiscale_doc(transform)],
        }
    }
    try:
        schemas.validate(doc, version, "image")
    except Exception:
        schema_accepts = False
    else:
        schema_accepts = True
    try:
        _pkg(version).Dataset.from_json(
            {"path": "0", "coordinateTransformations": [transform]}
        )
    except ValueError:
        model_accepts = False
    else:
        model_accepts = True
    assert model_accepts == schema_accepts


@pytest.mark.parametrize("version", ["0.6rc0", "0.6"])
def test_release_dataset_without_input_path_is_rejected(version: str) -> None:
    with pytest.raises(ValueError, match=r"input\.path"):
        _pkg(version).Dataset.from_json(
            {
                "path": "0",
                "coordinateTransformations": [
                    dict(_SCALE, output={"name": "physical"})
                ],
            }
        )


@pytest.mark.parametrize("version", ["0.6rc0", "0.6"])
def test_release_dataset_without_output_name_is_rejected(version: str) -> None:
    with pytest.raises(ValueError, match=r"output\.name"):
        _pkg(version).Dataset.from_json(
            {
                "path": "0",
                "coordinateTransformations": [
                    dict(_SCALE, input={"path": "0"})
                ],
            }
        )


def test_dev2_lone_scale_may_omit_its_references() -> None:
    dataset = _pkg("0.6.dev2").Dataset.from_json(
        {"path": "0", "coordinateTransformations": [_SCALE]}
    )
    assert dataset.to_json()["coordinateTransformations"] == [_SCALE]


# --------------------------------------------------------------------------
#   conversion from 0.5
# --------------------------------------------------------------------------


def _multiscale_05(**extra: tx.Any) -> tx.Any:
    return v0_5.Multiscale.from_json(
        {
            "name": "example",
            "axes": _AXES,
            "datasets": [
                {"path": "0", "coordinateTransformations": [_SCALE]},
                {
                    "path": "1",
                    "coordinateTransformations": [_SCALE, _TRANSLATION],
                },
            ],
            **extra,
        }
    )


_GLOBAL = {
    "scale": [{"type": "scale", "scale": [3.0, 3.0]}],
    "scale and translation": [
        {"type": "scale", "scale": [3.0, 3.0]},
        {"type": "translation", "translation": [1.0, 2.0]},
    ],
}


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("global_", [None] + list(_GLOBAL))
def test_05_converts_to_a_schema_valid_06_image(
    version: str, global_: tx.Optional[str]
) -> None:
    extra = {}
    if global_ is not None:
        extra["coordinateTransformations"] = _GLOBAL[global_]
    original = _multiscale_05(**extra)
    multiscale = original.to_version(version)
    doc = {"ome": {"version": version, "multiscales": [multiscale.to_json()]}}
    assert schemas.validate(doc, version, "image") is doc
    for dataset in multiscale.to_json()["datasets"]:
        (transform,) = dataset["coordinateTransformations"]
        if version in STRING_FORM:
            assert transform["input"] == dataset["path"]
            assert transform["output"] == "example"
        else:
            assert transform["input"] == {"path": dataset["path"]}
            assert transform["output"] == {"name": "example"}
    assert multiscale.to_version("0.5") == original


@pytest.mark.parametrize("version", ["0.6.dev3", "0.6"])
def test_05_global_transform_maps_onto_a_second_system(version: str) -> None:
    ms = _multiscale_05(
        coordinateTransformations=_GLOBAL["scale and translation"]
    ).to_version(version)
    assert [s.name for s in ms.coordinateSystems] == [
        "example",
        "example_transformed",
    ]
    (transform,) = [t.to_json() for t in ms.coordinateTransformations]
    assert transform["type"] == "sequence"
    if version in STRING_FORM:
        refs = ("example", "example_transformed")
    else:
        refs = ({"name": "example"}, {"name": "example_transformed"})
    assert (transform["input"], transform["output"]) == refs


# --------------------------------------------------------------------------
#   conversion within 0.6
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", VERSIONS)
@pytest.mark.parametrize("target", VERSIONS)
def test_dataset_references_take_the_target_form(
    source: str, target: str
) -> None:
    ms = _multiscale(source, _sequence(**_refs(source)))
    assert _transform(ms.to_version(target)) == _sequence(**_refs(target))


def test_dev2_lone_scale_gains_references_from_its_context() -> None:
    ms = _multiscale("0.6.dev2", _SCALE)
    assert _transform(ms.to_version("0.6.dev3")) == dict(_SCALE, **_STRINGS)
    assert _transform(ms.to_version("0.6")) == dict(_SCALE, **_OBJECTS)
    # A lone scale needs no references in 0.6.dev2, and keeps none.
    assert _transform(ms.to_version("0.6.dev2")) == _SCALE


def test_dev2_lone_scale_with_two_systems_raises_clearly() -> None:
    ms = _multiscale("0.6.dev2", _SCALE, systems=("physical", "other"))
    with pytest.raises(ValueError, match="exactly one coordinate system"):
        ms.to_version("0.6.dev3")


def test_dataset_output_in_another_group_cannot_become_a_string() -> None:
    transform = dict(
        _SCALE, input={"path": "0"}, output={"name": "physical", "path": "a"}
    )
    ms = _multiscale("0.6", transform)
    with pytest.raises(ValueError, match="single string"):
        ms.to_version("0.6.dev3")


def _multiscale_with_global(version: str, refs: tx.Any) -> tx.Any:
    doc = _multiscale_doc(
        dict(_SCALE, **_refs(version)), ("physical", "world")
    )
    doc["coordinateTransformations"] = [dict(_SCALE, **refs)]
    return _pkg(version).Multiscale.from_json(doc)


def _global(ms: tx.Any) -> tx.Dict[str, tx.Any]:
    (transform,) = ms.coordinateTransformations
    return transform.to_json()


def test_multiscale_references_are_names_in_every_form() -> None:
    names = {"input": "physical", "output": "world"}
    objects = {"input": {"name": "physical"}, "output": {"name": "world"}}
    ms = _multiscale_with_global("0.6.dev3", names)
    assert _global(ms.to_version("0.6")) == dict(_SCALE, **objects)
    back = ms.to_version("0.6").to_version("0.6.dev1")
    assert _global(back) == dict(_SCALE, **names)


def test_multiscale_reference_with_a_path_stops_at_dev2() -> None:
    refs = {
        "input": {"name": "physical"},
        "output": {"name": "w", "path": "a"},
    }
    ms = _multiscale_with_global("0.6", refs)
    # 0.6.dev2 and 0.6.dev3 accept an object, so it is kept.
    kept = _global(ms.to_version("0.6.dev2"))
    assert kept["input"] == "physical"
    assert kept["output"] == {"name": "w", "path": "a"}
    with pytest.raises(ValueError, match="coordinate system name"):
        ms.to_version("0.6.dev1")


def test_standalone_string_reference_cannot_become_an_object() -> None:
    t = _pkg("0.6.dev3").transformations.CoordinateTransformation.from_json(
        dict(_SCALE, input="a", output="b")
    )
    with pytest.raises(ValueError, match="names a coordinate system"):
        t.to_version("0.6.dev4")


def test_scene_string_reference_becomes_a_name() -> None:
    scene = _pkg("0.6.dev3").scenes.Scene.from_json(
        {
            "coordinateTransformations": [
                dict(_SCALE, input={"name": "a"}, output="b")
            ]
        }
    )
    (transform,) = scene.to_version("0.6").coordinateTransformations
    assert transform.to_json()["output"] == {"name": "b"}


# --------------------------------------------------------------------------
#   the pyramid writer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("offset", [0.0, 0.5])
def test_pyramid_level_uses_the_reference_form_of_its_version(
    version: str, offset: float
) -> None:
    base = _refs(version)
    style = _RefStyle(version, (True, base["output"]), {})
    doc = _level_dataset("1", [2.0, 2.0], [offset, offset], style)
    (transform,) = doc["coordinateTransformations"]
    expected = _refs(version)
    expected["input"] = "1" if version in STRING_FORM else {"path": "1"}
    assert {k: transform[k] for k in ("input", "output")} == expected
    _pkg(version).Dataset.from_json(doc)


def test_pyramid_level_in_dev2_needs_references_only_on_a_sequence() -> None:
    multiscale = _multiscale_doc(_SCALE)
    style = _RefStyle("0.6.dev2", (False, None), multiscale)
    lone = _level_dataset("1", [2.0, 2.0], [0.0, 0.0], style)
    assert lone["coordinateTransformations"] == [
        {"type": "scale", "scale": [2.0, 2.0]}
    ]
    pair = _level_dataset("1", [2.0, 2.0], [0.5, 0.5], style)
    (sequence,) = pair["coordinateTransformations"]
    assert (sequence["input"], sequence["output"]) == ("1", "physical")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _pkg("0.6.dev2").Dataset.from_json(pair)
