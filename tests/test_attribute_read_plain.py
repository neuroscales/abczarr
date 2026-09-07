"""Reading a container attribute returns plain built-in types.

An array's metadata stores its attributes as immutable, so a nested value it
holds is a [FrozenDict][abczarr._core.frozendict.FrozenDict] or a tuple. A
group's metadata stores plain values. The node attributes mapping normalizes
what it returns, so both node kinds hand back the same plain `dict` or `list`
for the same stored value. These tests pin that, and that a value read back
compares equal to the value that was stored.
"""

import pathlib

import pytest

import abczarr

# a real backend is needed to create nodes and persist their attributes
pytest.importorskip("zarr")


def _nested() -> dict:
    return {"x": {"y": [1, 2]}, "z": 3}


def test_an_array_returns_a_plain_dict(tmp_path: pathlib.Path) -> None:
    array = abczarr.create_array(
        str(tmp_path / "a.zarr"), shape=(2, 2), dtype="int8"
    )
    array.attrs["m"] = _nested()
    reopened = abczarr.open(str(tmp_path / "a.zarr"), mode="r")
    value = reopened.attrs["m"]
    assert type(value) is dict
    assert type(value["x"]) is dict
    assert type(value["x"]["y"]) is list
    assert value == _nested()


def test_a_group_returns_a_plain_dict(tmp_path: pathlib.Path) -> None:
    group = abczarr.create_group(str(tmp_path / "g.zarr"))
    group.attrs["m"] = _nested()
    reopened = abczarr.open(str(tmp_path / "g.zarr"), mode="r")
    value = reopened.attrs["m"]
    assert type(value) is dict
    assert value == _nested()


def test_an_array_and_a_group_agree(tmp_path: pathlib.Path) -> None:
    array = abczarr.create_array(
        str(tmp_path / "a.zarr"), shape=(2, 2), dtype="int8"
    )
    group = abczarr.create_group(str(tmp_path / "g.zarr"))
    array.attrs["m"] = _nested()
    group.attrs["m"] = _nested()
    assert array.attrs["m"] == group.attrs["m"]


def test_asdict_and_dict_are_plain(tmp_path: pathlib.Path) -> None:
    array = abczarr.create_array(
        str(tmp_path / "a.zarr"), shape=(2, 2), dtype="int8"
    )
    array.attrs["m"] = _nested()
    assert array.attrs.asdict() == {"m": _nested()}
    assert dict(array.attrs) == {"m": _nested()}
    # values() and items() route through the same normalization
    assert list(array.attrs.values()) == [_nested()]


def test_the_array_metadata_stays_immutable(tmp_path: pathlib.Path) -> None:
    # Normalizing the attributes view must not un-freeze the metadata itself,
    # which stays hashable.
    array = abczarr.create_array(
        str(tmp_path / "a.zarr"), shape=(2, 2), dtype="int8"
    )
    array.attrs["m"] = _nested()
    reopened = abczarr.open(str(tmp_path / "a.zarr"), mode="r")
    hash(reopened.metadata)  # does not raise
