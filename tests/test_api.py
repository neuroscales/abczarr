"""The top-level open() and driver selection.

The registry tests run without a backend; the end-to-end open() tests need
zarr-python 3.x and run on the coverage leg.
"""

import pathlib

import numpy as np
import pytest

import abczarr
from abczarr import registry
from abczarr.abc.capabilities import Support
from abczarr.abc.errors import UnsupportedZarrOperation
from abczarr.drivers.base import Driver
from abczarr.registry import available_drivers

# --------------------------------------------------------------------------
# registry -- no backend needed
# --------------------------------------------------------------------------


def test_open_is_exported() -> None:
    assert hasattr(abczarr, "open")
    assert "open" in abczarr.__all__


def test_available_drivers_instantiates_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the registry imports and instantiates each known driver; the base
    # Driver is available() by default, so it is returned
    monkeypatch.setattr(
        registry,
        "_KNOWN_DRIVERS",
        [("base", "abczarr.drivers.base", "Driver")],
    )
    names = [type(d).__name__ for d in available_drivers()]
    assert names == ["Driver"]


def test_register_driver_appends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_KNOWN_DRIVERS", [])
    registry.register_driver("abczarr.drivers.base", "Driver", "x")
    assert available_drivers()  # the base Driver is available by default


# --------------------------------------------------------------------------
# end-to-end open() -- needs zarr
# --------------------------------------------------------------------------

zarr = pytest.importorskip("zarr")


def _store(tmp_path: pathlib.Path) -> str:
    root = str(tmp_path / "data.zarr")
    group = zarr.open_group(root, mode="w")
    array = group.create_array(
        "img", shape=(6, 6), chunks=(3, 3), dtype="int32"
    )
    array[:] = np.arange(36).reshape(6, 6)
    return root


def test_open_returns_the_group(tmp_path: pathlib.Path) -> None:
    node = abczarr.open(_store(tmp_path), mode="r")
    assert isinstance(node, abczarr.ZarrGroup)
    assert sorted(node.keys()) == ["img"]


def test_open_array_reads_data(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(_store(tmp_path) + "/img", mode="r")
    assert isinstance(arr, abczarr.ZarrArray)
    assert arr.shape == (6, 6)
    assert np.asarray(arr[0, :3]).tolist() == [0, 1, 2]


def test_open_array_on_a_group_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(UnsupportedZarrOperation, match="open_array"):
        abczarr.open_array(_store(tmp_path), mode="r")


def test_open_group_on_an_array_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(UnsupportedZarrOperation, match="open_group"):
        abczarr.open_group(_store(tmp_path) + "/img", mode="r")


def test_open_with_an_explicit_driver(tmp_path: pathlib.Path) -> None:
    node = abczarr.open(_store(tmp_path), mode="r", driver="zarr-python")
    assert isinstance(node, abczarr.ZarrGroup)


def test_open_with_an_unknown_driver_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(UnsupportedZarrOperation):
        abczarr.open(_store(tmp_path), driver="no-such-driver")


def test_selection_routes_by_the_arrays_features(
    tmp_path: pathlib.Path,
) -> None:
    # two drivers offered; the capable one is chosen even though a driver
    # that supports nothing is listed first
    from abczarr.api import _choose
    from abczarr.drivers.zarr_python import ZarrPythonDriver

    array_path = _store(tmp_path) + "/img"

    class _Blind(Driver):
        name = "blind"

        def capability(self, capability: str) -> Support:
            return Support.NONE

    chosen = _choose(array_path, [_Blind(), ZarrPythonDriver()])
    assert chosen.name == "zarr-python"
