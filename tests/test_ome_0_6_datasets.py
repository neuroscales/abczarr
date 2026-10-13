"""The coordinate transformation a 0.6 dataset may carry.

From 0.6.dev2 on, the NGFF schema gives each dataset of a multiscale exactly
one coordinate transformation: a ``scale``, an ``identity`` (from 0.6.dev3
on), or a ``sequence`` of two steps, each a ``scale`` or a ``translation``.
0.6.dev1 puts no restriction on a dataset's transformations. The schema does
not order the two steps of the sequence, so a translation-before-scale
sequence, which some datasets in the wild use, still loads (see #169).
"""

import importlib
import types

import pytest
import typing_extensions as tx

from abczarr.ome import schemas, v0_5, v0_6dev1

#: The versions that restrict a dataset's transformation.
RESTRICTED = ["0.6.dev2", "0.6.dev3", "0.6.dev4", "0.6rc0", "0.6"]
#: The restricted versions that also allow an identity.
WITH_IDENTITY = RESTRICTED[1:]

_PACKAGES = {
    "0.6.dev1": "v0_6dev1",
    "0.6.dev2": "v0_6dev2",
    "0.6.dev3": "v0_6dev3",
    "0.6.dev4": "v0_6dev4",
    "0.6rc0": "v0_6rc0",
    "0.6": "v0_6",
}

_SCALE = {"type": "scale", "scale": [2.0, 2.0]}
_TRANSLATION = {"type": "translation", "translation": [0.5, 0.5]}
_AFFINE = {"type": "affine", "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
_AXES = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]


def _pkg(version: str) -> types.ModuleType:
    return importlib.import_module("abczarr.ome." + _PACKAGES[version])


def _refs(version: str) -> tx.Dict[str, tx.Any]:
    """The dataset's input and output references, spelled for `version`."""
    if version in ("0.6.dev2", "0.6.dev3"):
        return {"input": "0", "output": "physical"}
    return {"input": {"path": "0"}, "output": {"name": "physical"}}


def _sequence(version: str, *steps: tx.Any) -> tx.Dict[str, tx.Any]:
    refs = _refs(version)
    return {"type": "sequence", "transformations": list(steps), **refs}


def _dataset(version: str, *transforms: tx.Any) -> tx.Any:
    return _pkg(version).Dataset.from_json(
        {"path": "0", "coordinateTransformations": list(transforms)}
    )


def _valid(version: str) -> tx.Dict[str, tx.Any]:
    cases = {
        "scale": dict(_SCALE, **_refs(version)),
        "scale then translation": _sequence(version, _SCALE, _TRANSLATION),
        # The order of the steps is not constrained by the schema, and a
        # translation-before-scale dataset found in the wild must load.
        "translation then scale": _sequence(version, _TRANSLATION, _SCALE),
    }
    if version in WITH_IDENTITY:
        cases["identity"] = {"type": "identity", **_refs(version)}
    return cases


def _invalid(version: str) -> tx.Dict[str, tx.List[tx.Any]]:
    cases = {
        "affine": [dict(_AFFINE, **_refs(version))],
        "translation": [dict(_TRANSLATION, **_refs(version))],
        "two transformations": [
            dict(_SCALE, **_refs(version)),
            dict(_TRANSLATION, **_refs(version)),
        ],
        "no transformation": [],
        "one-step sequence": [_sequence(version, _SCALE)],
        "three-step sequence": [
            _sequence(version, _SCALE, _TRANSLATION, _SCALE)
        ],
        "sequence with an affine": [_sequence(version, _SCALE, _AFFINE)],
    }
    if version not in WITH_IDENTITY:
        cases["identity"] = [{"type": "identity", **_refs(version)}]
    return cases


_VALID = [(v, name) for v in RESTRICTED for name in _valid(v)]
_INVALID = [(v, name) for v in RESTRICTED for name in _invalid(v)]


# --------------------------------------------------------------------------
#   the typed model
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("version", "name"), _VALID)
def test_allowed_transformation_loads_and_round_trips(
    version: str, name: str
) -> None:
    transform = _valid(version)[name]
    dataset = _dataset(version, transform)
    assert len(dataset.coordinateTransformations) == 1
    assert dataset.to_json()["coordinateTransformations"] == [
        dataset.coordinateTransformations[0].to_json()
    ]
    assert type(dataset).from_json(dataset.to_json()) == dataset


@pytest.mark.parametrize(("version", "name"), _INVALID)
def test_disallowed_transformations_are_rejected(
    version: str, name: str
) -> None:
    with pytest.raises(ValueError):
        _dataset(version, *_invalid(version)[name])


@pytest.mark.parametrize("version", RESTRICTED)
def test_keyword_construction_is_checked_too(version: str) -> None:
    pkg = _pkg(version)
    affine = pkg.transformations.CoordinateTransformation.from_json(
        dict(_AFFINE, **_refs(version))
    )
    with pytest.raises(ValueError, match="'affine'"):
        pkg.Dataset(path="0", coordinateTransformations=[affine])


def test_dev1_dataset_is_not_restricted() -> None:
    refs = {"input": "0", "output": "physical"}
    dataset = _dataset("0.6.dev1", dict(_SCALE, **refs), dict(_AFFINE, **refs))
    assert [t.type for t in dataset.coordinateTransformations] == [
        "scale",
        "affine",
    ]


@pytest.mark.parametrize("version", ["0.6rc0", "0.6"])
@pytest.mark.parametrize(
    "transforms",
    [[_SCALE], [_SCALE, _TRANSLATION], [_AFFINE], [_SCALE, _SCALE]],
    ids=["scale", "sequence", "affine", "two scales"],
)
def test_model_agrees_with_the_vendored_schema(
    version: str, transforms: tx.List[tx.Any]
) -> None:
    # Each case is a whole image document, checked against the schema and
    # against the model. A lone transformation stands alone, and a pair is
    # wrapped in a sequence.
    if len(transforms) == 1:
        transform = dict(transforms[0], **_refs(version))
    else:
        transform = _sequence(version, *transforms)
    doc = {
        "ome": {
            "version": version,
            "multiscales": [
                {
                    "coordinateSystems": [{"name": "physical", "axes": _AXES}],
                    "datasets": [
                        {"path": "0", "coordinateTransformations": [transform]}
                    ],
                }
            ],
        }
    }
    try:
        schemas.validate(doc, version, "image")
    except Exception:
        schema_accepts = False
    else:
        schema_accepts = True
    try:
        _dataset(version, transform)
    except ValueError:
        model_accepts = False
    else:
        model_accepts = True
    assert model_accepts == schema_accepts


# --------------------------------------------------------------------------
#   conversion
# --------------------------------------------------------------------------


def _image_05() -> tx.Any:
    return v0_5.OMEImage.from_json(
        {
            "version": "0.5",
            "multiscales": [
                {
                    "name": "physical",
                    "axes": _AXES,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [_SCALE],
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [
                                _SCALE,
                                _TRANSLATION,
                            ],
                        },
                    ],
                }
            ],
        }
    )


@pytest.mark.parametrize("version", ["0.6rc0", "0.6"])
def test_05_converts_to_schema_valid_06_datasets(version: str) -> None:
    image = _image_05().to_version(version)
    doc = {"ome": image.to_json()}
    assert schemas.validate(doc, version, "image") is doc
    (multiscale,) = image.multiscales
    kinds = [d.coordinateTransformations[0].type for d in multiscale.datasets]
    assert kinds == ["scale", "sequence"]
    assert image.to_version("0.5") == _image_05()


def _dev1_multiscale(*transforms: tx.Any) -> tx.Any:
    return v0_6dev1.Multiscale.from_json(
        {
            "coordinateSystems": [{"name": "physical", "axes": _AXES}],
            "datasets": [
                {"path": "0", "coordinateTransformations": list(transforms)}
            ],
        }
    )


@pytest.mark.parametrize("version", RESTRICTED)
def test_dev1_dataset_with_several_transformations_raises_clearly(
    version: str,
) -> None:
    refs = {"input": "0", "output": "physical"}
    ms = _dev1_multiscale(dict(_SCALE, **refs), dict(_TRANSLATION, **refs))
    with pytest.raises(
        ValueError,
        match=r"cannot convert dataset '0' from OME 0\.6\.dev1 to 0\.6\.dev2",
    ):
        ms.to_version(version)


@pytest.mark.parametrize("source", ["0.6.dev1", "0.6.dev3", "0.6"])
def test_identity_becomes_a_scale_of_ones_in_dev2(source: str) -> None:
    refs = _refs(source)
    ms = _pkg(source).Multiscale.from_json(
        {
            "coordinateSystems": [{"name": "physical", "axes": _AXES}],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "identity", **refs}
                    ],
                }
            ],
        }
    )
    (dataset,) = ms.to_version("0.6.dev2").datasets
    (scale,) = dataset.coordinateTransformations
    assert scale.type == "scale"
    assert scale.scale == [1.0, 1.0]
    # The reference keeps the spelling of the source version.
    output = scale.output
    if isinstance(output, dict):
        output = output["name"]
    assert output == "physical"


def test_identity_without_a_declared_system_raises_clearly() -> None:
    dataset = _dataset(
        "0.6.dev3", {"type": "identity", "input": "0", "output": "elsewhere"}
    )
    with pytest.raises(ValueError, match="number of axes"):
        dataset.to_version("0.6.dev2")


def test_pyramid_level_without_offset_is_a_lone_scale() -> None:
    # A new pyramid level whose offset is zero carries a single scale. A
    # one-step sequence would be rejected by the schema.
    from abczarr.ome import v0_6
    from abczarr.ome.pyramid import _level_dataset

    doc = _level_dataset(
        "1", [2.0, 2.0], [0.0, 0.0], (True, {"name": "physical"})
    )
    assert doc["coordinateTransformations"] == [
        {
            "type": "scale",
            "scale": [2.0, 2.0],
            "input": {"path": "1"},
            "output": {"name": "physical"},
        }
    ]
    assert v0_6.Dataset.from_json(doc).to_json() == doc
