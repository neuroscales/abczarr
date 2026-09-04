"""Conversion over realistic, real-world metadata.

Round trips and cross-version conversion exercised on the kind of metadata a
microscopy dataset actually carries: multi-level pyramids, real dtypes and
codecs, sharding, and full OME-NGFF multiscales / omero / plate / well
structures.
"""

import pytest

from abczarr.metadata import v1, v2, v3
from abczarr.ome import v0_3, v0_4

# ==========================================================================
#   Zarr array metadata
# ==========================================================================

# A realistic v3 microscopy volume: 3D float32, sharded, blosc-compressed.
_V3_SHARDED_VOLUME = {
    "zarr_format": 3,
    "node_type": "array",
    "shape": [512, 2048, 2048],
    "data_type": "float32",
    "chunk_grid": {
        "name": "regular",
        "configuration": {"chunk_shape": [128, 512, 512]},
    },
    "chunk_key_encoding": {
        "name": "default",
        "configuration": {"separator": "/"},
    },
    "codecs": [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [64, 256, 256],
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {
                        "name": "blosc",
                        "configuration": {
                            "cname": "zstd",
                            "clevel": 5,
                            "shuffle": "shuffle",
                            "blocksize": 0,
                            "typesize": None,
                        },
                    },
                ],
            },
        }
    ],
    "fill_value": 0.0,
    "attributes": {"acquired": "2024-01-01", "instrument": "confocal"},
    "dimension_names": ["z", "y", "x"],
}

# A realistic v2 array: 2D uint16, gzip, delta filter, C order.
_V2_LABELS = {
    "zarr_format": 2,
    "shape": [4096, 4096],
    "chunks": [1024, 1024],
    "dtype": ">u2",
    "compressor": {"id": "gzip", "level": 6},
    "filters": [{"id": "delta", "dtype": ">u2"}],
    "fill_value": 0,
    "order": "C",
    "dimension_separator": "/",
    "attributes": {"purpose": "segmentation"},
}


@pytest.mark.parametrize(
    "dtype",
    [
        "|b1", "|i1", "<i2", ">i2", "<i4", ">i4", "<i8",
        "|u1", "<u2", ">u4", "<u8",
        "<f2", "<f4", ">f4", "<f8", ">f8",
        "<c8", "<c16",
    ],
)
def test_v2_v3_roundtrip_over_real_dtypes(dtype: str) -> None:
    meta = {
        "zarr_format": 2,
        "shape": [64, 64],
        "chunks": [16, 16],
        "dtype": dtype,
        "compressor": {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
        },
        "filters": [],
        "fill_value": 0,
        "order": "C",
        "dimension_separator": ".",
        "attributes": {},
    }
    m2 = v2.ArrayMetadata.from_json(meta)
    assert m2.to_version(3).to_version(2) == m2


def test_v2_labels_array_roundtrips_through_v3() -> None:
    # v3 has both filters and a chunk-key separator, so a filtered v2 array
    # survives the trip through it unchanged
    m2 = v2.ArrayMetadata.from_json(_V2_LABELS)
    assert m2.to_version(3).to_version(2) == m2


def test_v2_labels_array_to_v1_reports_filter_loss() -> None:
    # v1 predates filters, so the delta filter cannot survive the conversion
    m2 = v2.ArrayMetadata.from_json(_V2_LABELS)
    with pytest.warns(UserWarning, match="filters"):
        m1 = m2.to_version(1, policy="warn")
    # what v1 can hold does carry over
    assert m1.compression == "gzip"
    assert m1.dtype.numpy == m2.dtype.numpy


def test_v3_sharded_volume_to_v2_reports_sharding_loss() -> None:
    m3 = v3.ArrayMetadata.from_json(_V3_SHARDED_VOLUME)
    # the shard grid has no v2 equivalent
    with pytest.warns(UserWarning, match="sharding"):
        m2 = m3.to_version(2, policy="warn")
    # but the inner chunk shape, dtype, compressor and attributes survive
    assert m2.chunks == (64, 256, 256)
    assert m2.dtype.numpy == m3.data_type.numpy
    assert m2.compressor.id == "blosc"
    assert dict(m2.attributes) == dict(m3.attributes)


def test_v1_array_roundtrips_up_to_v3() -> None:
    m1 = v1.ArrayMetadata.from_json(
        {
            "zarr_format": 1,
            "shape": [1000, 1000],
            "chunks": [100, 100],
            "dtype": "<f8",
            "compression": "zlib",
            "compression_opts": {"level": 4},
            "fill_value": None,
            "order": "C",
            "attributes": {},
        }
    )
    assert m1.to_version(3).to_version(1) == m1


# ==========================================================================
#   OME-NGFF metadata
# ==========================================================================

# A realistic 5D pyramid: (t, c, z, y, x), three resolution levels.
_MULTISCALE_5D = {
    "version": "0.4",
    "name": "embryo",
    "type": "gaussian",
    "axes": [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ],
    "datasets": [
        {
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.36, 0.36]}
            ],
        },
        {
            "path": "1",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 0.5, 0.72, 0.72]}
            ],
        },
        {
            "path": "2",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 0.5, 1.44, 1.44]}
            ],
        },
    ],
    "coordinateTransformations": [
        {"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}
    ],
}

_OMERO = {
    "channels": [
        {
            "color": "00FF00",
            "window": {"start": 0, "end": 1500, "min": 0, "max": 65535},
            "active": True,
            "label": "GFP",
        },
        {
            "color": "FF0000",
            "window": {"start": 0, "end": 2000, "min": 0, "max": 65535},
            "active": True,
            "label": "mCherry",
        },
    ]
}

_PLATE = {
    "name": "screen-plate-1",
    "field_count": 4,
    "rows": [{"name": "A"}, {"name": "B"}],
    "columns": [{"name": "1"}, {"name": "2"}, {"name": "3"}],
    "wells": [
        {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
        {"path": "A/2", "rowIndex": 0, "columnIndex": 1},
        {"path": "B/3", "rowIndex": 1, "columnIndex": 2},
    ],
    "acquisitions": [
        {"id": 0, "name": "day0", "maximumfieldcount": 4},
        {"id": 1, "name": "day1", "maximumfieldcount": 4},
    ],
}

_WELL = {
    "images": [
        {"path": "0", "acquisition": 0},
        {"path": "1", "acquisition": 1},
    ]
}


@pytest.mark.parametrize(
    ("cls", "data"),
    [
        (v0_4.Multiscale, _MULTISCALE_5D),
        (v0_4.Omero, _OMERO),
        (v0_4.Plate, _PLATE),
        (v0_4.Well, _WELL),
    ],
)
def test_ome_structures_roundtrip_through_dict(cls: type, data: dict) -> None:
    m = cls.from_json(data)
    assert cls.from_json(m.to_json()) == m


@pytest.mark.parametrize(
    ("cls", "data"),
    [
        (v0_4.Multiscale, _MULTISCALE_5D),
        (v0_4.Omero, _OMERO),
        (v0_4.Plate, _PLATE),
        (v0_4.Well, _WELL),
    ],
)
def test_ome_structures_roundtrip_v04_v05(cls: type, data: dict) -> None:
    m = cls.from_json(data)
    assert m.to_version("0.5").to_version("0.4") == m


def test_5d_multiscale_roundtrips_v03_v04_v05() -> None:
    # start from v0.3 (bare axis names), climb to v0.5 and back
    m3 = v0_3.Multiscale.from_json(
        {
            "version": "0.3",
            "name": "embryo",
            "type": "gaussian",
            "axes": ["t", "c", "z", "y", "x"],
            "datasets": [{"path": "0"}, {"path": "1"}, {"path": "2"}],
        }
    )
    assert m3.to_version("0.5").to_version("0.3") == m3


def test_multiscale_scale_transforms_survive_conversion() -> None:
    m4 = v0_4.Multiscale.from_json(_MULTISCALE_5D)
    m5 = m4.to_version("0.5")
    scales = [
        d.coordinateTransformations[0].scale for d in m5.datasets
    ]
    assert scales == [
        [1.0, 1.0, 0.5, 0.36, 0.36],
        [1.0, 1.0, 0.5, 0.72, 0.72],
        [1.0, 1.0, 0.5, 1.44, 1.44],
    ]
