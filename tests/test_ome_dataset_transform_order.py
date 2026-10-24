"""The order of a 0.4 or 0.5 dataset's scale and translation.

The 0.4 and 0.5 specifications place a dataset's translation after its
scale, but their schemas do not enforce the order, and some datasets in
the wild list the translation first (see #169). The typed models load
such a dataset, keep its order, and convert it to and from 0.6 without
changing where its voxels land.
"""

import importlib
import types

import pytest
import typing_extensions as tx

from abczarr._core.attrs import ConversionError
from abczarr.ome import schemas

VERSIONS = ["0.4", "0.5"]

_SCALE = {"type": "scale", "scale": [2.0, 4.0]}
_TRANSLATION = {"type": "translation", "translation": [0.5, 1.5]}
_AXES = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]


def _pkg(version: str) -> types.ModuleType:
    return importlib.import_module("abczarr.ome.v" + version.replace(".", "_"))


def _image_doc(version: str, *transforms: tx.Any) -> tx.Dict[str, tx.Any]:
    multiscale = {
        "axes": _AXES,
        "datasets": [
            {"path": "0", "coordinateTransformations": list(transforms)}
        ],
    }
    if version == "0.4":
        multiscale["version"] = "0.4"
        return {"multiscales": [multiscale]}
    return {"version": "0.5", "multiscales": [multiscale]}


def _image(version: str, *transforms: tx.Any) -> tx.Any:
    return _pkg(version).OMEImage.from_json(_image_doc(version, *transforms))


@pytest.mark.parametrize("version", VERSIONS)
def test_translation_before_scale_loads_and_round_trips(version: str) -> None:
    dataset = _pkg(version).Dataset.from_json(
        {"path": "0", "coordinateTransformations": [_TRANSLATION, _SCALE]}
    )
    kinds = [t.type for t in dataset.coordinateTransformations]
    assert kinds == ["translation", "scale"]
    assert dataset.to_json()["coordinateTransformations"] == [
        _TRANSLATION,
        _SCALE,
    ]
    assert type(dataset).from_json(dataset.to_json()) == dataset


@pytest.mark.parametrize("version", VERSIONS)
def test_image_with_translation_first_is_schema_valid(version: str) -> None:
    image = _image(version, _TRANSLATION, _SCALE)
    doc = image.to_json()
    if version == "0.5":
        doc = {"ome": doc}
    assert schemas.validate(doc, version, "image") is doc
    assert type(image).from_json(image.to_json()) == image


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    "transforms",
    [
        [_TRANSLATION],
        [_SCALE, _SCALE],
        [_TRANSLATION, _TRANSLATION],
        [_TRANSLATION, _SCALE, _TRANSLATION],
    ],
    ids=["translation", "two scales", "two translations", "three"],
)
def test_other_transform_lists_are_still_rejected(
    version: str, transforms: tx.List[tx.Any]
) -> None:
    with pytest.raises(ConversionError):
        _pkg(version).Dataset.from_json(
            {"path": "0", "coordinateTransformations": transforms}
        )


def test_translation_first_keeps_its_meaning_through_0_6() -> None:
    image = _image("0.5", _TRANSLATION, _SCALE)
    converted = image.to_version("0.6")
    (dataset,) = converted.multiscales[0].datasets
    (sequence,) = dataset.coordinateTransformations
    assert [t.type for t in sequence.transformations] == [
        "translation",
        "scale",
    ]
    # Back in 0.5, the steps are written scale first. The translation
    # applied before a scale of 2 and 4 becomes an offset in physical
    # units, multiplied by that scale.
    back = converted.to_version("0.5")
    (dataset,) = back.multiscales[0].datasets
    assert dataset.to_json()["coordinateTransformations"] == [
        _SCALE,
        {"type": "translation", "translation": [1.0, 6.0]},
    ]
