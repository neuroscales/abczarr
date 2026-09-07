"""The frozen-Json model deep-freezes Json so a frozen metadata object is
genuinely immutable (and, as a consequence, hashable)."""

# stdlib
import json

# dependencies
import pytest

# core
from abczarr._core import typing as tz
from abczarr._core.auto.converters import get_converter
from abczarr._core.frozendict import FrozenDict, unfreeze

# metadata
from abczarr.metadata.v2.filters.base import Filter


def _freeze(value: dict) -> FrozenDict:
    return get_converter(tz.FrozenJsonDict)(value)


def test_deep_freezes_nested_containers() -> None:
    frozen = _freeze({"a": [1, {"b": 2}], "s": "x", "n": 5})
    assert isinstance(frozen, FrozenDict)
    assert isinstance(frozen["a"], tuple)  # a list becomes a tuple
    assert isinstance(frozen["a"][1], FrozenDict)  # a nested dict, frozen
    assert frozen["s"] == "x"  # scalars pass through unchanged
    assert frozen["n"] == 5


def test_frozen_json_is_immutable_and_hashable() -> None:
    frozen = _freeze({"a": [1, 2], "nested": {"b": [3]}})
    hash(frozen)  # immutable, so hashable -- does not raise
    with pytest.raises(TypeError):
        frozen["a"] = 9  # a FrozenDict cannot be mutated
    assert isinstance(frozen["nested"]["b"], tuple)


def test_a_list_value_is_not_coerced_to_a_bool() -> None:
    # Regression: a list- or dict-valued frozen-Json item used to fall
    # through the union to a greedy `bool` branch (`bool([1, 0])` is `True`).
    frozen = Filter.from_json({"id": "transpose", "order": [1, 0]})
    assert frozen.extra_items["order"] == (1, 0)
    assert isinstance(frozen.extra_items["order"], tuple)


def test_serializes_back_to_plain_json() -> None:
    frozen = Filter.from_json({"id": "x", "nested": {"a": [1, 2]}})
    assert frozen.to_json() == {"id": "x", "nested": {"a": [1, 2]}}


def test_metadata_with_container_extra_items_stays_hashable() -> None:
    # The whole frozen attrs object is hashable even when it carries a
    # nested container -- the point of freezing the Json it holds.
    frozen = Filter.from_json({"id": "x", "order": [1, 0], "cfg": {"k": [1]}})
    hash(frozen)


# --- unfreeze: the inverse, back to plain built-in types -------------------


def test_unfreeze_rebuilds_plain_builtin_types() -> None:
    frozen = _freeze({"a": [1, {"b": 2}], "s": "x", "n": 5})
    plain = unfreeze(frozen)
    assert type(plain) is dict
    assert type(plain["a"]) is list  # a frozen tuple becomes a list again
    assert type(plain["a"][1]) is dict  # a nested FrozenDict becomes a dict
    assert plain == {"a": [1, {"b": 2}], "s": "x", "n": 5}


def test_unfreeze_makes_a_frozen_structure_json_serializable() -> None:
    # The regression: a FrozenDict nested in an attribute payload used to
    # reach json.dumps and raise. Unfreezing it first is what fixes the write.
    frozen = _freeze({"ome": {"multiscales": [{"name": "img"}]}})
    with pytest.raises(TypeError):
        json.dumps(frozen)
    assert json.loads(json.dumps(unfreeze(frozen))) == {
        "ome": {"multiscales": [{"name": "img"}]}
    }


def test_unfreeze_leaves_a_plain_value_unchanged() -> None:
    plain = {"a": [1, 2], "b": {"c": 3}}
    assert unfreeze(plain) == plain
    assert unfreeze(5) == 5
    assert unfreeze("x") == "x"
