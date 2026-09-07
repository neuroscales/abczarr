"""Writing container-valued attributes on an array node.

A node's metadata deep-freezes its attributes into
[FrozenDict][abczarr._core.frozendict.FrozenDict] values so the metadata
stays immutable. A backend's attribute write serializes those values to JSON,
and a `FrozenDict` is not JSON serializable. These tests pin that a nested
dictionary attribute, and OME metadata, can be written on an array node and on
a group node, synchronously and asynchronously, for both Zarr formats.
"""

import asyncio
import pathlib

import pytest

import abczarr
from abczarr._core.frozendict import unfreeze

# a real backend is needed to create nodes and persist their attributes
pytest.importorskip("zarr")


_OME_05 = {
    "version": "0.5",
    "multiscales": [
        {
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0]}
                    ],
                }
            ],
        }
    ],
}
_OME_04 = {**_OME_05, "version": "0.4"}


def _array(tmp_path: pathlib.Path, name: str, **kwargs: object) -> object:
    location = str(pathlib.Path(tmp_path) / name)
    return abczarr.create_array(
        location, shape=(2, 2), dtype="int8", **kwargs
    )


# --- nested-dictionary attributes ------------------------------------------


def test_a_nested_dict_attribute_writes_on_an_array(
    tmp_path: pathlib.Path,
) -> None:
    array = _array(tmp_path, "a.zarr")
    # Before the fix this raised: the frozen attribute value reached the
    # backend's json.dumps, which cannot serialize a FrozenDict.
    array.attrs["meta"] = {"a": {"b": [1, 2]}}
    reopened = abczarr.open(str(tmp_path / "a.zarr"), mode="r")
    assert unfreeze(reopened.attrs["meta"]) == {"a": {"b": [1, 2]}}


def test_a_nested_dict_attribute_writes_on_a_v2_array(
    tmp_path: pathlib.Path,
) -> None:
    array = _array(tmp_path, "a.zarr", zarr_version=2)
    array.attrs["meta"] = {"a": {"b": [1, 2]}}
    reopened = abczarr.open(str(tmp_path / "a.zarr"), mode="r")
    assert unfreeze(reopened.attrs["meta"]) == {"a": {"b": [1, 2]}}


# --- OME metadata on an array node -----------------------------------------


def test_ome_0_5_writes_on_an_array(tmp_path: pathlib.Path) -> None:
    array = _array(tmp_path, "a.zarr")
    array.ome = _OME_05
    assert array.ome is not None
    assert array.ome.version == "0.5"
    reopened = abczarr.open(str(tmp_path / "a.zarr"), mode="r")
    assert reopened.ome.version == "0.5"


def test_ome_0_4_writes_on_an_array(tmp_path: pathlib.Path) -> None:
    array = _array(tmp_path, "a.zarr")
    array.ome = _OME_04
    assert array.ome is not None
    assert array.ome.version == "0.4"


def test_ome_writes_on_a_group_too(tmp_path: pathlib.Path) -> None:
    group = abczarr.create_group(str(tmp_path / "g.zarr"))
    group.ome = _OME_05
    assert group.ome is not None
    assert group.ome.version == "0.5"


# --- the async write path --------------------------------------------------


def test_ome_writes_on_an_async_array(tmp_path: pathlib.Path) -> None:
    async def go() -> object:
        return await abczarr.create(
            str(tmp_path / "a.zarr"),
            abczarr.ArrayConfig(shape=(2, 2), dtype="int8"),
            ome=_OME_05,
            asynchronous=True,
        )

    array = asyncio.run(go())
    assert array.ome is not None
    assert array.ome.version == "0.5"
