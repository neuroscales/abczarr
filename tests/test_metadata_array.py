import json

from abczarr.metadata import v2, v3


def test_zarray_v3():

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
        chunk_key_encoding=v3.DefaultChunkKeyEncoding(configuration={"separator": "/"}),
        codecs=(v3.BytesCodec(configuration={"endian": "little"}),),
        fill_value=float("nan"),
        attributes={"foo": 42, "bar": "apples", "baz": [1, 2, 3, 4]},
    )

    metadata = v3.ArrayMetadata.from_dict(EXAMPLE_JSON)

    assert metadata == EXAMPLE_META


def test_zarray_v3_extension():

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
        data_type=v3.DType(name="urn:example:datetime", configuration={"unit": "ns"}),
        chunk_grid=v3.RegularChunkGrid(configuration=(1000, 100)),
        chunk_key_encoding=v3.DefaultChunkKeyEncoding(configuration={"separator": "/"}),
        codecs=(v3.BytesCodec(configuration={"endian": "big"}),),
        fill_value=None,
    )

    metadata = v3.ArrayMetadata.from_dict(EXAMPLE_JSON)

    assert metadata == EXAMPLE_META


def test_zarray_v2():

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

    metadata = v2.ArrayMetadata.from_dict(EXAMPLE_JSON)

    assert metadata == EXAMPLE_META
