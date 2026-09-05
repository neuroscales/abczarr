"""Regression tests for ``to_zarr3`` on datetime64/timedelta64 dtypes (#91).

``to_zarr3`` used to call ``str`` methods on a ``numpy.dtype`` object and
required a leading count on the time unit, so every ``datetime64`` and
``timedelta64`` dtype raised on conversion to Zarr v3.
"""

# stdlib
# dependencies
import numpy as np
import pytest

# package
from abczarr._core.dtypes import to_zarr3
from abczarr.metadata import v2


@pytest.mark.parametrize(
    ("spec", "name", "unit", "scale"),
    [
        ("<M8[ns]", "numpy.datetime64", "ns", 1),
        ("<m8[ns]", "numpy.timedelta64", "ns", 1),
        (">M8[s]", "numpy.datetime64", "s", 1),
        ("<m8[us]", "numpy.timedelta64", "us", 1),
        (">M8[10s]", "numpy.datetime64", "s", 10),
        ("<M8", "numpy.datetime64", "generic", 1),
        ("<m8", "numpy.timedelta64", "generic", 1),
    ],
)
def test_to_zarr3_time_dtype(
    spec: str, name: str, unit: str, scale: int
) -> None:
    expected = {
        "name": name,
        "configuration": {"unit": unit, "scale_factor": scale},
    }
    assert to_zarr3(np.dtype(spec)) == expected


@pytest.mark.parametrize(
    ("spec", "name", "unit"),
    [
        ("<M8[ns]", "numpy.datetime64", "ns"),
        ("<m8[ns]", "numpy.timedelta64", "ns"),
    ],
)
def test_v2_time_dtype_to_version_3(spec: str, name: str, unit: str) -> None:
    dt3 = v2.DType(spec).to_version(3)
    assert dt3.to_json() == {
        "name": name,
        "configuration": {"unit": unit, "scale_factor": 1},
    }
