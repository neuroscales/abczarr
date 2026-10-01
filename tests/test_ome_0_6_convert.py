"""Coordinate transformations convert correctly between the 0.6 previews.

The 0.6 previews disagree on how a transformation addresses single axes:

* ``mapAxis`` maps output axis names to input axis names in 0.6.dev1, and is
  a permutation of input axis indices from 0.6.dev2 on.
* a ``byDimension`` child names its axes in ``input``/``output`` (dev1) or
  ``input_axes``/``output_axes`` (dev2), and addresses them by index in a
  wrapping object from dev3 on (``inputAxes``/``outputAxes`` from rc0 on).
* ``projectAxis`` exists from 0.6rc0 on.

Name/index translation reads the axes of the coordinate systems a
transformation refers to, so most tests convert a whole multiscale that
declares them. Where an upstream example exists for both versions, the
converted transformation is compared against the vendored example itself.
"""

import importlib
import json
import types
from pathlib import Path

import pytest
import typing_extensions as tx

TESTDIR = Path(__file__).parent

#: package suffix -> version string, oldest to newest.
VERSIONS = {
    "v0_6dev1": "0.6.dev1",
    "v0_6dev2": "0.6.dev2",
    "v0_6dev3": "0.6.dev3",
    "v0_6dev4": "0.6.dev4",
    "v0_6rc0": "0.6rc0",
    "v0_6": "0.6",
}
LATER_THAN_DEV1 = list(VERSIONS.values())[1:]
BEFORE_RC0 = list(VERSIONS.values())[:4]


def _pkg(version: str) -> types.ModuleType:
    for suffix, name in VERSIONS.items():
        if name == version:
            return importlib.import_module("abczarr.ome." + suffix)
    raise KeyError(version)


def _load(suffix: str, name: str) -> dict:
    path = TESTDIR / "data" / "ome" / suffix / (name + ".json")
    with path.open("r") as f:
        return json.load(f)


def _multiscale(version: str, systems: list, transforms: list) -> tx.Any:
    """A multiscale declaring `systems` and carrying `transforms`."""
    return _pkg(version).Multiscale.from_json(
        {
            "coordinateSystems": systems,
            "coordinateTransformations": transforms,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {
                            "type": "identity",
                            "input": {"path": "0"},
                            "output": {"name": systems[0]["name"]},
                        }
                    ],
                }
            ],
        }
    )


def _from_example(suffix: str, name: str) -> tx.Any:
    doc = _load(suffix, name)
    return _multiscale(
        VERSIONS[suffix],
        doc["coordinateSystems"],
        doc["coordinateTransformations"],
    )


def _xforms(ms: tx.Any) -> list:
    return [t.to_json() for t in ms.coordinateTransformations]


def _system(name: str, *axes: str) -> dict:
    return {"name": name, "axes": [{"name": a, "type": "space"} for a in axes]}


# --------------------------------------------------------------------------
#   mapAxis: names (dev1) <-> indices (dev2 onward)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", LATER_THAN_DEV1)
def test_dev1_permutation_maps_to_indices(version: str) -> None:
    ms = _from_example("v0_6dev1", "mapAxis1").to_version(version)
    assert [t["mapAxis"] for t in _xforms(ms)] == [[0, 1], [1, 0]]


def test_dev1_permutation_matches_the_release_example() -> None:
    converted = _from_example("v0_6dev1", "mapAxis1").to_version("0.6")
    release = _from_example("v0_6", "mapAxis1")
    assert converted.coordinateTransformations == (
        release.coordinateTransformations
    )


@pytest.mark.parametrize("version", LATER_THAN_DEV1)
def test_indices_map_back_to_dev1_names(version: str) -> None:
    original = _from_example("v0_6dev1", "mapAxis1")
    back = original.to_version(version).to_version("0.6.dev1")
    assert [t.mapAxis for t in back.coordinateTransformations] == [
        {"x": "i", "y": "j"},
        {"y": "i", "x": "j"},
    ]


def test_release_indices_map_to_dev1_names() -> None:
    ms = _from_example("v0_6", "mapAxis1").to_version("0.6.dev1")
    assert [t.mapAxis for t in ms.coordinateTransformations] == [
        {"y": "j", "x": "i"},
        {"y": "i", "x": "j"},
    ]


@pytest.mark.parametrize("version", LATER_THAN_DEV1)
def test_dev1_drop_and_duplicate_become_selection_affines(
    version: str,
) -> None:
    # in = (a, b). "projection down" keeps b as x. "projection up" writes
    # b to z and y, and a to x. Neither is a permutation, so each becomes
    # the affine that selects the same input coordinates.
    ms = _from_example("v0_6dev1", "mapAxis2").to_version(version)
    down, up = _xforms(ms)
    assert down["type"] == up["type"] == "affine"
    assert "mapAxis" not in down
    assert down["name"] == "projection down"
    assert down["affine"] == [[0.0, 1.0, 0.0]]
    assert up["affine"] == [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]


@pytest.mark.parametrize("version", ["0.6.dev2", "0.6.dev4", "0.6"])
def test_dev3_non_permutation_indices_become_affines(version: str) -> None:
    # The dev3 example writes the dev1 projections as index lists, which
    # its own schema rejects. Every version from dev2 on requires a
    # permutation, so the projections become the equivalent affines.
    ms = _from_example("v0_6dev3", "mapAxis2").to_version(version)
    down, up = _xforms(ms)
    assert down["affine"] == [[0.0, 1.0, 0.0]]
    assert up["affine"] == [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]


def test_mapaxis_needs_every_output_axis() -> None:
    ms = _multiscale(
        "0.6.dev1",
        [_system("in", "a", "b"), _system("out", "y", "x")],
        [
            {
                "type": "mapAxis",
                "mapAxis": {"x": "a"},
                "input": "in",
                "output": "out",
            }
        ],
    )
    with pytest.raises(ValueError, match=r"output axes \['y'\]"):
        ms.to_version("0.6.dev2")


def test_mapaxis_unknown_input_axis_raises() -> None:
    ms = _multiscale(
        "0.6.dev1",
        [_system("in", "a", "b"), _system("out", "y", "x")],
        [
            {
                "type": "mapAxis",
                "mapAxis": {"x": "a", "y": "q"},
                "input": "in",
                "output": "out",
            }
        ],
    )
    with pytest.raises(ValueError, match=r"input axes \['q'\]"):
        ms.to_version("0.6.dev2")


def test_standalone_named_mapaxis_needs_its_coordinate_systems() -> None:
    t = _pkg("0.6.dev1").transformations.CoordinateTransformation.from_json(
        {
            "type": "mapAxis",
            "mapAxis": {"x": "j", "y": "i"},
            "input": "in",
            "output": "out",
        }
    )
    with pytest.raises(ValueError, match="axis names of its input"):
        t.to_version("0.6.dev2")


def test_standalone_permutation_converts_between_index_versions() -> None:
    t = _pkg("0.6").transformations.CoordinateTransformation.from_json(
        {"type": "mapAxis", "mapAxis": [1, 0]}
    )
    assert t.to_version("0.6.dev2").mapAxis == [1, 0]
    with pytest.raises(ValueError, match="axis names"):
        t.to_version("0.6.dev1")


# --------------------------------------------------------------------------
#   byDimension: names (dev1, dev2) <-> indices (dev3 onward)
# --------------------------------------------------------------------------


# The dev3 `byDimension2` example uses different translation values from
# the dev1 one, so it is not a conversion of it.
@pytest.mark.parametrize(
    ("suffix", "name"),
    [
        ("v0_6dev2", "byDimension1"),
        ("v0_6dev2", "byDimension2"),
        ("v0_6dev3", "byDimension1"),
    ],
)
def test_dev1_bydimension_matches_the_later_example(
    suffix: str, name: str
) -> None:
    converted = _from_example("v0_6dev1", name).to_version(VERSIONS[suffix])
    expected = _from_example(suffix, name)
    assert converted.coordinateTransformations == (
        expected.coordinateTransformations
    )


@pytest.mark.parametrize("version", ["0.6rc0", "0.6"])
def test_dev1_bydimension_uses_camel_case_indices(version: str) -> None:
    ms = _from_example("v0_6dev1", "byDimension1").to_version(version)
    (by_dimension,) = _xforms(ms)
    assert by_dimension["transformations"] == [
        {
            "transformation": {"type": "translation", "translation": [-1.0]},
            "inputAxes": [1],
            "outputAxes": [1],
        },
        {
            "transformation": {"type": "scale", "scale": [2.0]},
            "inputAxes": [0],
            "outputAxes": [0],
        },
    ]


@pytest.mark.parametrize("version", ["0.6.dev1", "0.6.dev2"])
@pytest.mark.parametrize("name", ["byDimension1", "byDimension2"])
def test_release_bydimension_maps_back_to_names(
    name: str, version: str
) -> None:
    ms = _from_example("v0_6", name).to_version(version)
    (by_dimension,) = ms.coordinateTransformations
    in_key = "input" if version == "0.6.dev1" else "input_axes"
    out_key = "output" if version == "0.6.dev1" else "output_axes"
    children = [t.to_json() for t in by_dimension.transformations]
    if name == "byDimension1":
        expected = [(["i"], ["x"]), (["j"], ["y"])]
    else:
        expected = [(["i", "k"], ["y", "x"]), (["j"], ["z"])]
    assert [(c[in_key], c[out_key]) for c in children] == expected


@pytest.mark.parametrize("version", list(VERSIONS.values()))
def test_bydimension_round_trips(version: str) -> None:
    # From 0.6.dev4 on, a coordinate-system reference is an object, and it
    # stays an object on the way back. Only the children are compared.
    original = _from_example("v0_6dev1", "byDimension2")
    back = original.to_version(version).to_version("0.6.dev1")
    assert (
        _xforms(back)[0]["transformations"]
        == (_xforms(original)[0]["transformations"])
    )


def test_nested_mapaxis_uses_the_child_axes() -> None:
    systems = [_system("in", "a", "b", "c"), _system("out", "x", "y", "z")]
    dev1 = _multiscale(
        "0.6.dev1",
        systems,
        [
            {
                "type": "byDimension",
                "input": "in",
                "output": "out",
                "transformations": [
                    {
                        "type": "mapAxis",
                        "mapAxis": {"x": "b", "y": "a"},
                        "input": ["a", "b"],
                        "output": ["x", "y"],
                    },
                    {
                        "type": "scale",
                        "scale": [2.0],
                        "input": ["c"],
                        "output": ["z"],
                    },
                ],
            }
        ],
    )
    dev3 = dev1.to_version("0.6.dev3")
    (child, _) = _xforms(dev3)[0]["transformations"]
    assert child == {
        "transformation": {"type": "mapAxis", "mapAxis": [1, 0]},
        "input_axes": [0, 1],
        "output_axes": [0, 1],
    }
    assert dev3.to_version("0.6.dev1") == dev1


def test_sequence_passes_its_axes_to_its_children() -> None:
    dev1 = _multiscale(
        "0.6.dev1",
        [_system("in", "j", "i"), _system("out", "y", "x")],
        [
            {
                "type": "sequence",
                "input": "in",
                "output": "out",
                "transformations": [
                    {"type": "mapAxis", "mapAxis": {"y": "i", "x": "j"}},
                ],
            }
        ],
    )
    (sequence,) = _xforms(dev1.to_version("0.6"))
    assert sequence["transformations"] == [
        {"type": "mapAxis", "mapAxis": [1, 0]}
    ]


# --------------------------------------------------------------------------
#   sequence: the axes between its steps
# --------------------------------------------------------------------------

_SCALE = {"type": "scale", "scale": [2.0, 3.0]}
_NAMED_MAP_AXIS = {"type": "mapAxis", "mapAxis": {"y": "i", "x": "j"}}
_INDEX_MAP_AXIS = {"type": "mapAxis", "mapAxis": [1, 0]}
_AFFINE = {"type": "affine", "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
_ROTATION = {"type": "rotation", "rotation": [[0.0, 1.0], [1.0, 0.0]]}


def _sequence(version: str, steps: list, systems: tx.Any = None) -> tx.Any:
    """A multiscale with one sequence from system ``in`` to ``out``."""
    if systems is None:
        systems = [_system("in", "j", "i"), _system("out", "y", "x")]
    by_name = version in ("0.6.dev1", "0.6.dev2", "0.6.dev3")
    sequence = {
        "type": "sequence",
        "input": "in" if by_name else {"name": "in"},
        "output": "out" if by_name else {"name": "out"},
        "transformations": steps,
    }
    return _multiscale(version, systems, [sequence])


def _steps(ms: tx.Any) -> list:
    return _xforms(ms)[0]["transformations"]


def _standalone_sequence(steps: list) -> tx.Any:
    return _pkg("0.6").transformations.CoordinateTransformation.from_json(
        {"type": "sequence", "transformations": steps}
    )


@pytest.mark.parametrize("version", LATER_THAN_DEV1)
@pytest.mark.parametrize("first", [True, False])
def test_named_mapaxis_inside_a_sequence_converts(
    first: bool, version: str
) -> None:
    # The mapAxis reads its input names from the sequence input or through
    # the scale before it, and its output names from the sequence output or
    # through the scale after it.
    steps = [_NAMED_MAP_AXIS, _SCALE] if first else [_SCALE, _NAMED_MAP_AXIS]
    dev1 = _sequence("0.6.dev1", steps)
    converted = dev1.to_version(version)
    expected = (
        [_INDEX_MAP_AXIS, _SCALE] if first else [_SCALE, _INDEX_MAP_AXIS]
    )
    assert _steps(converted) == expected
    assert _steps(converted.to_version("0.6.dev1")) == steps


@pytest.mark.parametrize("version", LATER_THAN_DEV1)
@pytest.mark.parametrize("first", [True, False])
def test_index_mapaxis_inside_a_sequence_converts_to_names(
    first: bool, version: str
) -> None:
    steps = [_INDEX_MAP_AXIS, _SCALE] if first else [_SCALE, _INDEX_MAP_AXIS]
    converted = _sequence(version, steps).to_version("0.6.dev1")
    expected = (
        [_NAMED_MAP_AXIS, _SCALE] if first else [_SCALE, _NAMED_MAP_AXIS]
    )
    assert _steps(converted) == expected


@pytest.mark.parametrize("first", [True, False])
def test_index_mapaxis_inside_a_sequence_converts_to_0_5(first: bool) -> None:
    # 0.5 cannot hold a mapAxis, so the conversion drops it with a warning.
    steps = [_INDEX_MAP_AXIS, _SCALE] if first else [_SCALE, _INDEX_MAP_AXIS]
    with pytest.warns(UserWarning, match="mapAxis"):
        converted = _sequence("0.6", steps).to_version("0.5")
    (scale,) = converted.coordinateTransformations
    assert scale.scale == [2.0, 3.0]


@pytest.mark.parametrize("other", [_AFFINE, _ROTATION])
@pytest.mark.parametrize("first", [True, False])
def test_named_mapaxis_next_to_a_matrix_names_the_cause(
    first: bool, other: dict
) -> None:
    steps = [_NAMED_MAP_AXIS, other] if first else [other, _NAMED_MAP_AXIS]
    position = 1 if first else 0
    ms = _sequence("0.6.dev1", steps)
    cause = f"inferred through the {other['type']} at index {position}"
    with pytest.raises(ValueError, match=cause):
        ms.to_version("0.6.dev2")


def test_named_mapaxis_after_coordinates_names_the_cause() -> None:
    steps = [{"type": "coordinates", "path": "c"}, _NAMED_MAP_AXIS]
    ms = _sequence("0.6.dev1", steps)
    with pytest.raises(ValueError) as info:
        ms.to_version("0.6.dev2")
    assert "inferred through the coordinates at index 0" in str(info.value)
    assert "multiscale" not in str(info.value)


_PROJECT_STEP = {"type": "projectAxis", "createdOutputs": [0]}
# in (i, j) -> out (c, y, x): zeros at output 0
_PROJECT_STEP_AFFINE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
_PROJECT_SEQUENCES = {
    "middle": [
        {"type": "scale", "scale": [2.0, 2.0]},
        _PROJECT_STEP,
        {"type": "scale", "scale": [3.0, 3.0, 3.0]},
    ],
    "first": [_PROJECT_STEP, {"type": "scale", "scale": [3.0, 3.0, 3.0]}],
    "last": [{"type": "scale", "scale": [2.0, 2.0]}, _PROJECT_STEP],
}


@pytest.mark.parametrize("version", BEFORE_RC0)
@pytest.mark.parametrize("where", list(_PROJECT_SEQUENCES))
def test_projectaxis_inside_a_sequence_becomes_an_affine(
    where: str, version: str
) -> None:
    steps = _PROJECT_SEQUENCES[where]
    systems = [_system("in", "i", "j"), _system("out", "c", "y", "x")]
    converted = _sequence("0.6", steps, systems).to_version(version)
    (affine,) = [t for t in _steps(converted) if t["type"] == "affine"]
    assert affine["affine"] == _PROJECT_STEP_AFFINE
    # The affine stays an affine on the way back up.
    back = _steps(converted.to_version("0.6"))
    assert [t["type"] for t in back] == [
        "affine" if t is _PROJECT_STEP else t["type"] for t in steps
    ]


def test_projectaxis_inside_a_sequence_counts_from_the_scales() -> None:
    # No coordinate system is in scope. The scale vectors give the counts.
    t = _standalone_sequence(_PROJECT_SEQUENCES["middle"])
    (affine,) = [
        s
        for s in t.to_version("0.6.dev4").to_json()["transformations"]
        if s["type"] == "affine"
    ]
    assert affine["affine"] == _PROJECT_STEP_AFFINE


@pytest.mark.parametrize(
    ("matrix", "n_in"),
    [
        # 2 inputs -> 3 outputs
        ({"type": "affine", "affine": [[1, 0, 0], [0, 1, 0], [1, 1, 0]]}, 3),
        # 2 inputs -> 2 outputs, with the homogeneous row
        ({"type": "affine", "affine": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, 2),
        ({"type": "rotation", "rotation": [[0, 1], [1, 0]]}, 2),
    ],
)
def test_matrix_inside_a_sequence_gives_the_axis_count(
    matrix: dict, n_in: int
) -> None:
    project = {"type": "projectAxis", "droppedInputs": [0]}
    t = _standalone_sequence([matrix, project])
    (_, affine) = t.to_version("0.6.dev4").to_json()["transformations"]
    assert len(affine["affine"]) == n_in - 1
    assert len(affine["affine"][0]) == n_in + 1


def test_matrix_after_projectaxis_gives_the_axis_count() -> None:
    project = {"type": "projectAxis", "droppedInputs": [0]}
    matrix = {"type": "affine", "affine": [[1, 0, 0, 0], [0, 1, 0, 0]]}
    t = _standalone_sequence([project, matrix])
    (affine, _) = t.to_version("0.6.dev4").to_json()["transformations"]
    assert affine["affine"] == [
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]


def test_sequence_axis_count_disagreement_raises_when_needed() -> None:
    # Read forward, the first scale gives 3 axes before the projectAxis.
    # Read backward, the last scale and the projectAxis give 1.
    steps = [
        {"type": "scale", "scale": [1.0, 2.0, 3.0]},
        _PROJECT_STEP,
        {"type": "scale", "scale": [1.0, 2.0]},
    ]
    t = _standalone_sequence(steps)
    with pytest.raises(ValueError, match="3 axes between transformations 0"):
        t.to_version("0.6.dev4")
    # The step to 0.6rc0 needs no axes, so the sequence is kept as it is.
    assert t.to_version("0.6rc0").to_version("0.6") == t


@pytest.mark.parametrize("version", ["0.6.dev2", "0.6.dev3"])
def test_named_bydimension_after_an_affine_inside_a_sequence(
    version: str,
) -> None:
    by_dimension = {
        "type": "byDimension",
        "transformations": [
            {"type": "scale", "scale": [2.0], "input": ["j"], "output": ["y"]},
            {"type": "scale", "scale": [3.0], "input": ["i"], "output": ["x"]},
        ],
    }
    ms = _sequence("0.6.dev1", [_AFFINE, by_dimension])
    if version == "0.6.dev2":
        # Axis names carry over between dev1 and dev2 without being looked up.
        (_, converted) = _steps(ms.to_version(version))
        assert [c["input_axes"] for c in converted["transformations"]] == [
            ["j"],
            ["i"],
        ]
    else:
        with pytest.raises(ValueError, match="through the affine at index 0"):
            ms.to_version(version)


def test_standalone_index_bydimension_converts_without_systems() -> None:
    doc = _load("v0_6", "byDimension2")["coordinateTransformations"][0]
    t = _pkg("0.6").transformations.CoordinateTransformation.from_json(doc)
    dev4 = t.to_version("0.6.dev4").to_json()
    assert [c["input_axes"] for c in dev4["transformations"]] == [[3, 2], [1]]
    assert t.to_version("0.6.dev4").to_version("0.6") == t


# --------------------------------------------------------------------------
#   byDimension coverage: every output axis exactly once
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", LATER_THAN_DEV1)
def test_dev1_bydimension_writing_an_axis_twice_raises(version: str) -> None:
    ms = _from_example("v0_6dev1", "byDimensionInvalid2")
    with pytest.raises(ValueError, match="'x' is written by transformations"):
        ms.to_version(version)


def test_dev1_bydimension_unknown_output_axis_raises() -> None:
    ms = _from_example("v0_6dev1", "byDimensionInvalid1")
    with pytest.raises(ValueError, match=r"output axes \['z'\]"):
        ms.to_version("0.6.dev2")


def test_bydimension_missing_output_axis_raises() -> None:
    ms = _multiscale(
        "0.6.dev1",
        [_system("in", "j", "i"), _system("out", "y", "x")],
        [
            {
                "type": "byDimension",
                "input": "in",
                "output": "out",
                "transformations": [
                    {
                        "type": "scale",
                        "scale": [2.0],
                        "input": ["j"],
                        "output": ["y"],
                    },
                ],
            }
        ],
    )
    with pytest.raises(ValueError, match="no transformation writes .*'x'"):
        ms.to_version("0.6.dev2")


@pytest.mark.parametrize("version", ["0.6rc0", "0.6.dev4", "0.6.dev1"])
def test_index_bydimension_writing_an_axis_twice_raises(version: str) -> None:
    ms = _from_example("v0_6", "byDimensionInvalid2")
    with pytest.raises(ValueError, match="'x' is written by transformations"):
        ms.to_version(version)


def test_index_bydimension_out_of_range_axis_raises() -> None:
    ms = _from_example("v0_6", "byDimensionInvalid1")
    with pytest.raises(ValueError, match=r"output axes \[2\]"):
        ms.to_version("0.6rc0")


def test_standalone_index_bydimension_duplicate_raises() -> None:
    doc = _load("v0_6", "byDimensionInvalid2")["coordinateTransformations"][0]
    t = _pkg("0.6").transformations.CoordinateTransformation.from_json(doc)
    with pytest.raises(ValueError, match="written by transformations 0 and 1"):
        t.to_version("0.6rc0")


_UNWRAPPED = [
    (suffix, version)
    for suffix in ("v0_6dev3", "v0_6dev4")
    for version in VERSIONS.values()
    if version != VERSIONS[suffix]
]


@pytest.mark.parametrize(("suffix", "version"), _UNWRAPPED)
def test_unwrapped_bydimension_entry_raises(suffix: str, version: str) -> None:
    # These upstream examples list each transformation directly instead of
    # wrapping it in an object, which no version from dev3 on allows.
    ms = _from_example(suffix, "byDimensionXarray")
    with pytest.raises(ValueError, match="has no 'transformation' object"):
        ms.to_version(version)


def test_bydimension_entry_that_is_not_an_object_raises() -> None:
    # A parsed byDimension always holds objects, so the JSON is converted
    # directly.
    from abczarr.ome import _transforms06

    doc = {"type": "byDimension", "transformations": ["scale"]}
    with pytest.raises(ValueError, match="entry 0 .* is not an object"):
        _transforms06.convert(doc, "0.6", "0.6rc0", {}, lambda field: None)


# --------------------------------------------------------------------------
#   projectAxis (0.6rc0 onward) -> affine in earlier versions
# --------------------------------------------------------------------------

_PROJECT_AXIS_AFFINE = {
    # in (i, j) -> out (c, z, y, x): zeros at outputs 0 and 1
    "projectAxis": [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    # in (c, i, j) -> out (z, y, x): drop input 0, zero at output 0
    "projectAxis2": [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
}


@pytest.mark.parametrize("version", BEFORE_RC0)
@pytest.mark.parametrize("name", list(_PROJECT_AXIS_AFFINE))
def test_projectaxis_becomes_an_affine_before_rc0(
    name: str, version: str
) -> None:
    (t,) = _xforms(_from_example("v0_6", name).to_version(version))
    assert t["type"] == "affine"
    assert t["name"] == "up-project"
    assert "createdOutputs" not in t and "droppedInputs" not in t
    assert t["affine"] == _PROJECT_AXIS_AFFINE[name]


@pytest.mark.parametrize("name", list(_PROJECT_AXIS_AFFINE))
def test_projectaxis_is_kept_in_rc0(name: str) -> None:
    ms = _from_example("v0_6", name)
    assert ms.to_version("0.6rc0").to_version("0.6") == ms


def test_projectaxis_counts_from_the_output_system() -> None:
    # Only the output system is known: 4 outputs, 2 created -> 2 inputs.
    ms = _multiscale(
        "0.6",
        [_system("out", "c", "z", "y", "x")],
        [
            {
                "type": "projectAxis",
                "createdOutputs": [0, 1],
                "input": {"path": "elsewhere"},
                "output": {"name": "out"},
            }
        ],
    )
    (t,) = _xforms(ms.to_version("0.6.dev4"))
    assert t["affine"] == _PROJECT_AXIS_AFFINE["projectAxis"]


def test_standalone_projectaxis_needs_an_axis_count() -> None:
    t = _pkg("0.6").transformations.CoordinateTransformation.from_json(
        {"type": "projectAxis", "droppedInputs": [0]}
    )
    with pytest.raises(ValueError, match="number of input or output axes"):
        t.to_version("0.6.dev4")


# --------------------------------------------------------------------------
#   other structural changes and schema conformance
# --------------------------------------------------------------------------


def test_inverseof_does_not_exist_from_dev3() -> None:
    doc = _load("v0_6dev2", "inverseOf")
    ms = _multiscale(
        "0.6.dev2", doc["coordinateSystems"], doc["coordinateTransformations"]
    )
    assert ms.to_version("0.6.dev1").to_version("0.6.dev2") == ms
    with pytest.raises(ValueError, match="InverseOf does not exist"):
        ms.to_version("0.6.dev3")


_CONFORMANCE = [
    ("mapAxis1", "0.6.dev2"),
    ("mapAxis2", "0.6.dev2"),
    ("byDimension1", "0.6.dev2"),
    ("byDimension2", "0.6.dev2"),
    ("mapAxis1", "0.6"),
    ("mapAxis2", "0.6"),
    ("byDimension1", "0.6"),
    ("byDimension2", "0.6"),
]


@pytest.mark.parametrize(("name", "version"), _CONFORMANCE)
def test_converted_dev1_example_conforms_to_the_target_schema(
    name: str,
    version: str,
    validate_systems_and_transforms: "tx.Callable[[dict, str], None]",
) -> None:
    source = _load("v0_6dev1", name)
    # Typed axes, which the 0.6 schemas require.
    systems = [
        {
            "name": s["name"],
            "axes": [{"name": a["name"], "type": "space"} for a in s["axes"]],
        }
        for s in source["coordinateSystems"]
    ]
    ms = _multiscale(
        "0.6.dev1", systems, source["coordinateTransformations"]
    ).to_version(version)
    validate_systems_and_transforms(
        {
            "coordinateSystems": [s.to_json() for s in ms.coordinateSystems],
            "coordinateTransformations": _xforms(ms),
        },
        version,
    )


# The 0.6.dev4 schema rejects a reference by name alone, including in the
# upstream dev4 examples, so that version is not checked here.
@pytest.mark.parametrize("version", ["0.6.dev2", "0.6.dev3"])
@pytest.mark.parametrize("name", list(_PROJECT_AXIS_AFFINE))
def test_converted_projectaxis_conforms_to_the_target_schema(
    name: str, version: str
) -> None:
    from abczarr.ome import schemas

    ms = _from_example("v0_6", name).to_version(version)
    xforms = _xforms(ms)
    # The coordinate-system references keep the object spelling of the
    # source, while these versions name the system by a bare string. That
    # spelling is not what is being checked here.
    for t in xforms:
        t["input"], t["output"] = t["input"]["name"], t["output"]["name"]
    schemas.validate(xforms, version, "coordinate_transformations")
