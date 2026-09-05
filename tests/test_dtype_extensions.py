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

# stdlib
import sys

# dependencies
import numpy as np
import pytest

# package
import abczarr._core.dtypes as dtypes_mod
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


# -- ml_dtypes-backed extension floats (#126) --------------------------------
#
# ``bfloat16`` and the ``float8_*`` variants have no numpy scalar on their own,
# but ``ml_dtypes`` registers them with numpy on import, after which
# ``asdtype`` resolves them transparently. When ``ml_dtypes`` is absent the
# error points at the optional extra that supplies them.


def test_asdtype_resolves_ml_dtypes_floats() -> None:
    # ml_dtypes must be importable for its names to resolve; importing it here
    # registers the dtypes with numpy for the rest of the process.
    pytest.importorskip("ml_dtypes")

    assert asdtype("bfloat16") == np.dtype("bfloat16")
    # the mapping form (a bare v3 extension name, no configuration) too
    assert asdtype({"name": "bfloat16"}) == np.dtype("bfloat16")
    # at least one float8 variant
    assert asdtype("float8_e4m3fn") == np.dtype("float8_e4m3fn")
    assert asdtype("float8_e5m2") == np.dtype("float8_e5m2")


def test_pointed_error_when_ml_dtypes_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate ml_dtypes being unavailable without disturbing the global numpy
    # registration (which persists once any test has imported ml_dtypes):
    #   * make numpy fail to resolve the name, and
    #   * make ``import ml_dtypes`` raise ImportError.
    real_dtype = np.dtype

    def fake_dtype(obj: object, *args, **kwargs) -> np.dtype:
        if isinstance(obj, str) and obj == "bfloat16":
            raise TypeError("data type 'bfloat16' not understood")
        return real_dtype(obj, *args, **kwargs)

    monkeypatch.setattr(dtypes_mod.np, "dtype", fake_dtype)
    monkeypatch.setitem(sys.modules, "ml_dtypes", None)

    with pytest.raises(UnsupportedConversion) as info:
        asdtype("bfloat16")
    assert info.value.field == "bfloat16"
    assert "abczarr[ml-dtypes]" in str(info.value)


def test_plain_error_for_non_ml_dtypes_extension() -> None:
    # ml_dtypes provides no scalar for the complex extension floats, so the
    # extra would not help and the message stays plain (no hint).
    with pytest.raises(UnsupportedConversion) as info:
        asdtype("complex_float32")
    assert info.value.hint is None
    assert "ml-dtypes" not in str(info.value)
