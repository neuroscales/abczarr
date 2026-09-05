import json

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
