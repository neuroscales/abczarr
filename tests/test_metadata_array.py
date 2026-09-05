import json

import pytest

from abczarr.metadata import v1, v2, v3
from abczarr.schemas import validate


def test_zarray_v3() -> None:

    EXAMPLE = """
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10000, 1000],
        "dimension_names": ["rows", "columns"],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {
                "chunk_shape": [1000, 100]
            }
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {
                "separator": "/"
            }
        },
        "codecs": [{
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }],
        "fill_value": "NaN",
        "attributes": {
            "foo": 42,
            "bar": "apples",
            "baz": [1, 2, 3, 4]
        }
    }
    """
    EXAMPLE_JSON = json.loads(EXAMPLE)

    EXAMPLE_META = v3.ArrayMetadata(
        zarr_format=3,
        node_type="array",
        shape=(10000, 1000),
        dimension_names=("rows", "columns"),
        data_type=v3.Float64(),
        chunk_grid=v3.RegularChunkGrid(configuration=(1000, 100)),
        chunk_key_encoding=v3.DefaultChunkKeyEncoding(
            configuration={"separator": "/"}
        ),
        codecs=(v3.BytesCodec(configuration={"endian": "little"}),),
        fill_value=float("nan"),
        attributes={"foo": 42, "bar": "apples", "baz": [1, 2, 3, 4]},
    )

    metadata = v3.ArrayMetadata.from_json(EXAMPLE_JSON)

    assert metadata == EXAMPLE_META


def test_zarray_v3_extension() -> None:

    EXAMPLE = """
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10000, 1000],
        "data_type": {
            "name": "urn:example:datetime",
            "configuration": {
                "unit": "ns"
            }
        },
        "chunk_grid": {
            "name": "regular",
            "configuration": {
                "chunk_shape": [1000, 100]
            }
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {
                "separator": "/"
            }
        },
        "codecs": [{
            "name": "bytes",
            "configuration": {
                "endian": "big"
            }
        }],
        "fill_value": null
    }
    """
    EXAMPLE_JSON = json.loads(EXAMPLE)

    EXAMPLE_META = v3.ArrayMetadata(
        zarr_format=3,
        node_type="array",
        shape=(10000, 1000),
        data_type=v3.DType(
            name="urn:example:datetime",
            configuration={"unit": "ns"}
        ),
        chunk_grid=v3.RegularChunkGrid(configuration=(1000, 100)),
        chunk_key_encoding=v3.DefaultChunkKeyEncoding(
            configuration={"separator": "/"}
        ),
        codecs=(v3.BytesCodec(configuration={"endian": "big"}),),
        fill_value=None,
    )

    metadata = v3.ArrayMetadata.from_json(EXAMPLE_JSON)

    assert metadata == EXAMPLE_META


def test_zarray_v2() -> None:

    EXAMPLE = """
    {
        "chunks": [
            1000,
            1000
        ],
        "compressor": {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1
        },
        "dtype": "<f8",
        "fill_value": "NaN",
        "filters": [
            {"id": "delta", "dtype": "<f8", "astype": "<f4"}
        ],
        "order": "C",
        "shape": [
            10000,
            10000
        ],
        "zarr_format": 2
    }
    """

    EXAMPLE_JSON = json.loads(EXAMPLE)

    EXAMPLE_META = v2.ArrayMetadata(
        chunks=(1000, 1000),
        compressor=v2.BloscCodec(cname="lz4", clevel=5, shuffle=1),
        dtype="<f8",
        fill_value=float("nan"),
        filters=(v2.DeltaFilter(dtype="<f8", astype="<f4"),),
        order="C",
        shape=(10000, 10000),
        zarr_format=2,
    )

    metadata = v2.ArrayMetadata.from_json(EXAMPLE_JSON)

    assert metadata == EXAMPLE_META


def test_a_core_data_type_serializes_as_a_bare_string() -> None:
    meta = v3.ArrayMetadata.from_json(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4],
            "data_type": "float32",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [4]},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}}
            ],
            "fill_value": 0,
            "attributes": {},
        }
    )
    out = meta.to_json()
    # a core data type is a bare string, per the Zarr v3 spec
    assert out["data_type"] == "float32"
    # a codec keeps its object form
    assert isinstance(out["codecs"][0], dict)
    assert out["codecs"][0]["name"] == "bytes"


def test_v3_complex_fill_value_accepts_re_im_array() -> None:
    # The Zarr v3 spec encodes a complex fill value as a two-element
    # ``[real, imag]`` array (JSON has no complex literal), which the
    # authored ``array.schema`` allows. The model must accept it.
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "complex64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [4]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "fill_value": [1, 2],
        "attributes": {},
    }

    meta = v3.ArrayMetadata.from_json(document)
    assert meta.fill_value == (1, 2)

    # round-trips back to the ``[real, imag]`` array, which the authored
    # schema accepts -- model and schema agree in both directions.
    out = meta.to_json()
    assert out["fill_value"] == [1, 2]
    validate(out, "v3", "array")


def test_v1_scalar_compression_opts_accepted() -> None:
    # The authored v1 ``array.schema`` allows a scalar ``compression_opts``
    # (an integer or string) alongside the object form; the model must too.
    document = {
        "zarr_format": 1,
        "shape": [10],
        "chunks": [5],
        "dtype": "<f8",
        "compression": "zlib",
        "compression_opts": 1,
        "fill_value": 0,
        "order": "C",
        "attributes": {},
    }

    meta = v1.ArrayMetadata.from_json(document)
    assert meta.compression_opts == 1

    out = meta.to_json()
    assert out["compression_opts"] == 1
    validate(out, "v1", "array")


def test_v1_scalar_compression_opts_converts_to_v2_and_v3() -> None:
    # Expanding a scalar goes through numcodecs, absent on the minimal-deps
    # test leg -- skip there (the metadata layer still imports without it).
    pytest.importorskip("numcodecs")
    # A scalar ``compression_opts`` must still convert: numcodecs defines
    # which parameter the scalar fills, so ``zlib`` ``1`` becomes ``level=1``
    # in the v2 compressor -- and the result is identical to the object form.
    base = {
        "zarr_format": 1,
        "shape": [10],
        "chunks": [5],
        "dtype": "<f8",
        "compression": "zlib",
        "fill_value": 0,
        "order": "C",
        "attributes": {},
    }
    scalar = v1.ArrayMetadata.from_json({**base, "compression_opts": 1})
    obj = v1.ArrayMetadata.from_json(
        {**base, "compression_opts": {"level": 1}}
    )

    v2_meta = scalar.to_version(2)
    assert v2_meta.compressor.to_json() == {"id": "zlib", "level": 1}
    validate(v2_meta.to_json(), "v2", "array")

    # the scalar form converts to exactly what the object form does, for v2
    # and v3 alike (the point of the widening -- a scalar is not a special
    # case downstream).
    assert scalar.to_version(2).to_json() == obj.to_version(2).to_json()
    assert scalar.to_version(3).to_json() == obj.to_version(3).to_json()

    # a scalar for a codec that IS a v3 core codec round-trips all the way to
    # a valid v3 document (gzip ``5`` -> ``level=5``).
    gzip = v1.ArrayMetadata.from_json(
        {**base, "compression": "gzip", "compression_opts": 5}
    )
    validate(gzip.to_version(3).to_json(), "v3", "array")

    # a string scalar (blosc's cname) is expanded the same way.
    blosc = v1.ArrayMetadata.from_json(
        {**base, "compression": "blosc", "compression_opts": "lz4"}
    )
    assert blosc.to_version(2).compressor.to_json()["cname"] == "lz4"
