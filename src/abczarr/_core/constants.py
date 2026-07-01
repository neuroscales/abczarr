
Z3_JSON = "zarr.json"
Z2ARRAY_JSON = ".zarray"
Z2GROUP_JSON = ".zgroup"
Z2ATTRS_JSON = ".zattrs"
Z1META_JSON = "meta"
Z1ATTRS_JSON = "attrs"


FILE_MODES = ("r", "r+", "a", "w", "w-")
LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
COMPRESSORS_V2 = ("blosc", "zlib", "bz2", "zstd", "none")
COMPRESSORS_V3 = ("blosc", "gzip", "none")
ZARR_VERSIONS = (1, 2, 3)
OME_VERSIONS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6")
DRIVERS = ("zarr-python", "tensorstore", "zarrita")
