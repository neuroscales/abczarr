"""Regression tests for ``asdtype`` on Zarr v3 extension dtypes (#92).

``asdtype`` read a data type's configuration with ``getattr`` on a mapping
(always ``None``), so any configured extension dtype -- ``numpy.datetime64``,
``struct`` -- was reduced to its bare name and then indexed as a string.
And an extension dtype with no numpy equivalent (``complex_float32``,
``float8_*``) surfaced a raw numpy ``TypeError`` instead of the package's
``UnsupportedConversion``. (``string`` and ``bytes`` do have a conventional
numpy representation -- ``object`` plus a vlen codec -- and are covered by
``test_dtype_v3_vlen``.)
"""

# dependencies
import numpy as np
import pytest

# package
from abczarr._core.dtypes import asdtype
from abczarr.errors import UnsupportedConversion
from abczarr.metadata import v3


def test_configured_datetime_extension_to_numpy() -> None:
    dt = v3.DType.from_json(
        {"name": "numpy.datetime64",
         "configuration": {"unit": "ns", "scale_factor": 1}}
    )
    assert dt.numpy == np.dtype("<M8[ns]")


def test_configured_struct_extension_to_numpy() -> None:
    dt = v3.DType.from_json(
        {"name": "struct",
         "configuration": {"fields": [
             {"name": "a", "data_type": "int32"},
             {"name": "b", "data_type": "float64"},
         ]}}
    )
    assert dt.numpy == np.dtype([("a", "<i4"), ("b", "<f8")])


def test_asdtype_mapping_with_configuration() -> None:
    # the mapping form, indexed directly through asdtype
    got = asdtype(
        {"name": "numpy.timedelta64",
         "configuration": {"unit": "s", "scale_factor": 10}}
    )
    assert got == np.dtype("timedelta64[10s]")


def test_unrepresentable_v3_dtype_to_version_2() -> None:
    # "complex_float32" is a Zarr v3 extension dtype numpy has no scalar for,
    # so it cannot be carried into Zarr v2's numpy model (numpy has no complex
    # type over an extension float, and does not understand the name even when
    # ml_dtypes is installed).
    dt = v3.DType.from_json("complex_float32")
    with pytest.raises(UnsupportedConversion) as info:
        dt.to_version(2)
    assert info.value.field == "complex_float32"
    assert info.value.version == 2


def test_unrepresentable_dtype_via_asdtype() -> None:
    with pytest.raises(UnsupportedConversion):
        asdtype("complex_float32")
