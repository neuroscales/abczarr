"""The top-level open() and driver selection.

The registry tests run without a backend; the end-to-end open() tests need
zarr-python 3.x and run on the coverage leg.
"""

import pathlib

import numpy as np
import pytest

import abczarr
from abczarr._errors import UnsupportedZarrOperation
from abczarr.abc.capabilities import Support
from abczarr.api import _entry
from abczarr.api import _registry as registry
from abczarr.api._registry import available_drivers
from abczarr.drivers.base import Driver

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
    from abczarr.api._entry import _choose
    from abczarr.drivers.zarr_python import ZarrPythonDriver

    array_path = _store(tmp_path) + "/img"

    class _Blind(Driver):
        name = "blind"

        def capability(self, capability: str) -> Support:
            return Support.NONE

    chosen = _choose(array_path, [_Blind(), ZarrPythonDriver()])
    assert chosen.name == "zarr-python"


# --------------------------------------------------------------------------
# open(mode=...) honours create modes, zarr-style -- needs zarr
# --------------------------------------------------------------------------


def test_open_w_with_array_fields_creates_an_array(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "a.zarr")
    arr = abczarr.open(root, mode="w", shape=(4, 4), dtype="int16")
    assert isinstance(arr, abczarr.ZarrArray)
    assert arr.shape == (4, 4)


def test_open_w_without_array_fields_creates_a_group(
    tmp_path: pathlib.Path,
) -> None:
    grp = abczarr.open(str(tmp_path / "g.zarr"), mode="w")
    assert isinstance(grp, abczarr.ZarrGroup)


def test_open_w_overwrites_an_existing_node(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "a.zarr")
    abczarr.open(root, mode="w", shape=(8,), dtype="int8")
    arr = abczarr.open(root, mode="w", shape=(2, 2), dtype="int8")
    assert arr.shape == (2, 2)


def test_open_w_dash_fails_if_it_exists(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g.zarr")
    abczarr.open(root, mode="w")
    with pytest.raises(FileExistsError):
        abczarr.open(root, mode="w-")


def test_open_x_is_w_dash(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "g.zarr")
    grp = abczarr.open(root, mode="x")  # creates when missing
    assert isinstance(grp, abczarr.ZarrGroup)
    with pytest.raises(FileExistsError):
        abczarr.open(root, mode="x")  # fails when it exists


def test_open_a_opens_or_creates(tmp_path: pathlib.Path) -> None:
    root = str(tmp_path / "a.zarr")
    made = abczarr.open(root, mode="a", shape=(3,), dtype="u1")  # created
    assert isinstance(made, abczarr.ZarrArray)
    again = abczarr.open(root, mode="a")  # opened
    assert isinstance(again, abczarr.ZarrArray)
    assert again.shape == (3,)


def test_open_r_and_r_plus_error_when_missing(tmp_path: pathlib.Path) -> None:
    for mode in ("r", "r+"):
        with pytest.raises(FileNotFoundError):
            abczarr.open(str(tmp_path / "missing.zarr"), mode=mode)


def test_open_r_with_creation_fields_is_an_error(
    tmp_path: pathlib.Path,
) -> None:
    root = str(tmp_path / "a.zarr")
    abczarr.open(root, mode="w", shape=(4,), dtype="u1")
    with pytest.raises(TypeError, match="opens an existing node"):
        abczarr.open(root, mode="r", shape=(4,))


def test_open_rejects_an_overwrite_field(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="overwrite"):
        abczarr.open(str(tmp_path / "a.zarr"), mode="w", overwrite=True)


def test_open_array_create_mode_needs_a_shape(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="needs at least a shape"):
        abczarr.open_array(str(tmp_path / "a.zarr"), mode="w")


def test_open_array_create_mode_makes_an_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.open_array(
        str(tmp_path / "a.zarr"), mode="w", shape=(5,), dtype="int32"
    )
    assert isinstance(arr, abczarr.ZarrArray)
    assert arr.shape == (5,)


def test_open_group_create_mode_makes_a_group(tmp_path: pathlib.Path) -> None:
    grp = abczarr.open_group(str(tmp_path / "g.zarr"), mode="w")
    assert isinstance(grp, abczarr.ZarrGroup)


def test_open_group_create_mode_rejects_array_fields(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(TypeError, match="no array fields"):
        abczarr.open_group(str(tmp_path / "g.zarr"), mode="w", shape=(4,))


# --------------------------------------------------------------------------
# create_array / create_group -- the metadata-free create surface
# --------------------------------------------------------------------------


def test_create_array_makes_an_array(tmp_path: pathlib.Path) -> None:
    arr = abczarr.create_array(
        str(tmp_path / "a.zarr"), shape=(5, 5), dtype="int32", chunks=(5, 5)
    )
    assert isinstance(arr, abczarr.ZarrArray)
    assert arr.shape == (5, 5)
    assert arr.dtype == np.dtype("int32")
    arr[:] = np.arange(25).reshape(5, 5)
    assert np.asarray(arr[0, :3]).tolist() == [0, 1, 2]


def test_create_array_from_a_config(tmp_path: pathlib.Path) -> None:
    from abczarr.api._config import ArrayConfig

    arr = abczarr.create_array(
        str(tmp_path / "a.zarr"),
        config=ArrayConfig(shape=(4, 4), dtype="float32"),
    )
    assert isinstance(arr, abczarr.ZarrArray)
    assert arr.shape == (4, 4)
    assert arr.dtype == np.dtype("float32")


def test_create_array_needs_a_shape(tmp_path: pathlib.Path) -> None:
    with pytest.raises(TypeError, match="needs at least a shape"):
        abczarr.create_array(str(tmp_path / "a.zarr"))


def test_create_array_rejects_a_group_shaped_request(
    tmp_path: pathlib.Path,
) -> None:
    from abczarr.api._config import GroupConfig

    with pytest.raises(TypeError, match="needs at least a shape"):
        abczarr.create_array(
            str(tmp_path / "a.zarr"), config=GroupConfig()
        )


def test_exists_detects_a_v1_meta_file(tmp_path: pathlib.Path) -> None:
    # a v1 array is marked by a ``meta`` file; _METADATA_KEYS omitted it, so
    # a location holding one was read as absent (and open(mode="a") on a v1
    # array would try to create rather than open).
    (tmp_path / "meta").write_bytes(b"{}")
    assert _entry._exists(str(tmp_path)) is True
