"""Foundations the conversion machinery stands on.

Two shared-machinery guarantees that cross-version conversion depends on:

* ``fields`` accepts an instance, so serialization (which walks a live
  object's fields) works on the Python 3.8 floor, where ``attrs.fields``
  rejects instances;
* a converter/validator that fails raises its real error, not a wrapper
  bug that masks it.
"""

import typing_extensions as tx

from abczarr._core.auto import fields
from abczarr._core.auto.converters import get_converter
from abczarr._core.auto.validators import get_validator
from abczarr.metadata import v2


def _v2_array() -> v2.ArrayMetadata:
    return v2.ArrayMetadata.from_json(
        {
            "zarr_format": 2,
            "shape": [4],
            "chunks": [2],
            "dtype": "<f8",
            "compressor": None,
            "filters": [],
            "fill_value": 0,
            "order": "C",
            "dimension_separator": ".",
            "attributes": {},
        }
    )


# --------------------------------------------------------------------------
# fields() on an instance (dead on the 3.8 floor before the shim)
# --------------------------------------------------------------------------


def test_fields_accepts_an_instance() -> None:
    meta = _v2_array()
    from_instance = [f.name for f in fields(meta)]
    from_class = [f.name for f in fields(type(meta))]
    assert from_instance == from_class
    assert "shape" in from_instance


def test_to_dict_on_an_instance_works() -> None:
    # to_json() walks fields(self); this is the operation that was dead on 3.8
    meta = _v2_array()
    data = meta.to_json()
    assert isinstance(data, dict)
    assert v2.ArrayMetadata.from_json(data) == meta


# --------------------------------------------------------------------------
# a failing converter/validator raises its real error, not the wrapper bug
# --------------------------------------------------------------------------


def test_converter_error_is_the_real_exception() -> None:
    convert = get_converter(tx.Literal["a", "b"])
    try:
        convert("z")
    except Exception as e:
        # a real ConversionError, not "exceptions must derive from
        # BaseException" from a wrapper that raised a bound method
        assert type(e).__name__.endswith("ConversionError")
        assert "must derive from BaseException" not in str(e)
    else:
        raise AssertionError("expected the conversion to fail")


def test_validator_error_is_the_real_exception() -> None:
    validate = get_validator(tx.Literal["a", "b"])
    try:
        validate("z")
    except Exception as e:
        assert type(e).__name__.endswith("ValidationError")
        assert "must derive from BaseException" not in str(e)
    else:
        raise AssertionError("expected the validation to fail")
