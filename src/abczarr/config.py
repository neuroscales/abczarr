"""Configuration related to output Zarr Archive."""

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# core
from ._core import typing as tz
from ._core.attrs import autodefine, evolve
from ._core.dtypes import to_zarr3 as dtype_to_zarr3
from ._core.sharding import ChunkSpec, auto_chunk, auto_shard
from .metadata.base import ArrayMetadata


@autodefine
class ZarrConfig:
    """
    Parameters
    ----------
    zarr_version
        Zarr version to use.
    overwrite
        Overwrite the existing zarr file if it exists.
    driver
        library used for Zarr IO Operation
    """
    zarr_version: tz.ZarrVersion = 3
    overwrite: bool = False
    driver: tz.AnyDriver = "zarr-python"


@autodefine
class ZarrArrayConfig(ZarrConfig):
    """
    Configuration related to Zarr arrays.

    Parameters
    ----------
    zarr_version
        Zarr version to use. If `shard` is used, 3 is required.
    chunks
        Output chunk size.
        * Will be copy-padded to match the number of dimensions of the data.
        * If a dictionary, maps dimension names to chunk sizes.
        * Zero means no chunking along that dimension.
        * "auto" means that the chunk size will be automatically determined
          to match a target (binary) chunk size.
    shards
        Output shard size.
        * If a dictionary, maps dimension names to shard sizes.
        * Zero means no sharding along that dimension.
        * "auto" means that the shard size will be automatically determined
          to match a target (binary) file size.
    max_chunk_bytes
        Target chunk size in bytes when `chunks` is set to "auto".
    max_shard_bytes
        Target shard size in bytes when `shards` is set to "auto".
    dimension_separator
        The separator placed between the dimensions of a chunk.
    order:
        Memory layout order for the data array.
    compressor
        Compression method
    compressor_opt
        Compression options
    overwrite
        Overwrite the existing zarr file if it exists.
    driver
        library used for Zarr IO Operation

    """
    shape: tz.Shape = ()
    dtype: np.dtype = np.float32
    names: tx.Tuple[tx.Optional[str], ...] = ()
    chunks: ChunkSpec = "auto"
    shards: tx.Optional[ChunkSpec] = None
    max_chunk_bytes: int = 8 * 1024**2  # 8 MB
    max_shard_bytes: int = 2 * 1024**3  # 2 GB
    dimension_separator: tz.DimensionSeparator = "/"
    order: tz.MemoryOrder = "C"
    fill_value: tx.Optional[tz.BuiltinNumber] = None
    compressor: tz.CompressorTypeV3 = "blosc"
    compressor_opt: tz.CompressorOptions

    def __attrs_post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Ensure that sharding options (shard, shard_channels, shard_time)
          are only used when zarr_version == 3;
          otherwise raise NotImplementedError.
        """
        if self.zarr_version < 3:
            if self.shards:
                raise ValueError("Shard is not supported for Zarr < 3.")

    def finalize(
        self,
        data: tx.Optional[npt.ArrayLike] = None,
        **kwargs
    ) -> tx.Self:
        """
        Finalize the configuration by computing any "auto" values.

        Other Parameters
        ----------------
        shape : sequence[int]
            Shape of the data array.
        dtype : int | str | numpy.dtype
            Data type of the data array, or its number of bytes.
        names : sequence[str]
            Names of the dimensions, if `chunks` or `shards` is a mapping.
        """
        shape = kwargs.get("shape", self.shape)
        dtype = kwargs.get("dtype", self.dtype)
        names = kwargs.get("names", self.names)
        shape = getattr(data, "shape", shape)
        dtype = getattr(data, "dtype", dtype)
        names = getattr(data, "names", names)

        shards, chunks = self.shards, self.chunks
        chunks = auto_chunk(
            shape,
            chunks,
            names=names,
            itemsize=dtype,
            maxsize=self.max_chunk_bytes,
        )
        if self.shards:
            shards, chunks = auto_shard(
                shape,
                shards,
                chunks,
                names=names,
                itemsize=dtype,
                maxsize=self.max_shard_bytes,
            )
        return evolve(
            self,
            shape=shape, dtype=dtype, names=names,
            shards=shards, chunks=chunks,
        )

    def to_metadata(self) -> ArrayMetadata:
        """
        Convert the configuration to Zarr metadata.

        Returns
        -------
        metadata : ArrayMetadata
            Metadata dictionary for the Zarr array.
        """
        config = self.finalize()
        version = config.zarr_version

        compressor = config.compressor.lower()
        if compressor not in ("raw", "none"):
            compressors = [{
                "name": compressor,
                "configuration": config.compressor_opt or {}
            }]
        else:
            compressors = []

        codec_little_endian =  {
            "name": "bytes",
            "configuration": {"endian": "little"},
        }

        if config.shards:

            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": config.shards},
            }

            codecs = [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": config.chunks,
                    "codecs": [
                        codec_little_endian,
                        *compressors
                    ],
                    "index_codecs": [
                        codec_little_endian,
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }]

        else:

            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": config.chunks},
            }

            codecs = [
                codec_little_endian,
                *compressors,
            ]

        metadata = {
            "zarr_format": 3,
            "shape": config.shape,
            "data_type": dtype_to_zarr3(config.dtype),
            "chunk_grid": chunk_grid,
            "codecs": codecs,
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": config.dimension_separator}
            },
            "fill_value": config.fill_value,
        }

        return ArrayMetadata.from_dict(metadata).to_version(version)


@autodefine
class OMEZarrConfig(ZarrConfig):
    """
    Configuration related to output OME Zarr archives.

    Parameters
    ----------
    zarr_version
        Zarr version to use. If `shard` is used, 3 is required.
    chunks
        Output chunk size.
        Behavior depends on the number of values provided:
        * one:   used for all spatial dimensions
        * three: used for spatial dimensions ([z, y, x])
        * four+:  used for channels and spatial dimensions ([c, z, y, x])
        If `"auto"`, find chunk size smaller than 1 MB (TODO: not implemented)
    chunk_channels
        Put channels in different chunk.
        If False, combine all channels in a single chunk.
    chunk_time
        Put time points in different chunk.
        If False, combine all time points in a single chunk.
    shards
        Output shard size.
        Behavior same as chunk.
        If `"auto"`, find shard size that ensures files smaller than 2TB,
        assuming a compression ratio or 2.
    shard_channels
        Put channels in different shards.
        If False, combine all channels in a single shard.
    shard_time
        Put time points in different shards.
        If False, combine all time points in a single shard.
    no_time
        If True, indicates that the dataset does not have a time dimension.
        In such cases, any fourth dimension is interpreted as the channel
        dimension.
    no_pyramid_axis
        Spatial axis that should not be downsampled when generating
        pyramid levels.
        If None, downsampling is applied across all spatial axes.
    levels : int, optional
        Number of pyramid levels to generate.
        If set to -1, all possible levels are generated until the
        smallest level fits into one chunk.
    ome_version
        Version of the OME-Zarr specification to use
    overwrite
        when no name is supplied and using default output name, if
        overwrite is set, it won't ask if overwrite
    driver : {"zarr-python", "tensorstore", "zarrita"}
        library used for Zarr IO Operation
    """

    chunk_channels: bool = False
    chunk_time: bool = True
    shard_channels: bool = False
    shard_time: bool = False
    no_time: bool = False
    no_pyramid_axis: tx.Optional[tz.SpatialAxisName] = None
    levels: int = -1
    ome_version: tz.OMEVersion = "0.4"

    def __attrs_post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Ensure that sharding options (shard, shard_channels, shard_time)
          are only used when zarr_version == 3;
          otherwise raise NotImplementedError.
        """
        if self.zarr_version < 3:
            if self.shards or self.shard_channels or self.shard_time:
                raise ValueError("Shard is not supported for Zarr < 3.")


@autodefine
class GeneralConfig:
    """
    General configuration for the conversion process.

    Parameters
    ----------
    out : str | None
        Output path for the converted data.
    max_load : int
        Maximum number of items to load into memory at once.
    log_level : {"debug", "info", "warning", "error", "critical"}
        Logging level for the conversion process.
    verbose : bool
        If True, set log_level to "debug".
    """

    out:tx.Optional[str] = None
    max_load: int = 1024
    log_level: tz.LogLevel = "info"
    verbose: bool = False

    def __attrs_post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Ensure that max_load is a positive integer;
          otherwise raise ValueError.
        - If verbose is True, set log_level to "debug".
        """
        if not isinstance(self.max_load, int) or self.max_load <= 0:
            raise ValueError("max_load must be a positive integer")
        if self.verbose:
            self.log_level = "debug"

    def set_default_name(self, name: str, variant: str = "") -> None:
        """
        Assign a default output name if none was specified.

        !!! tip
            - If `self.out` is already set, does nothing.
            - Otherwise sets `self.out` to `f"{name}{self.variant}.zarr"`
              if the variant is specified, or `f"{name}.zarr"` otherwise.
            - If the resulting path exists and `overwrite` is False, prompts
              the user for confirmation and raises FileExistsError if not
              confirmed.

        Parameters
        ----------
        name : str
            Base filename (without extension) to use for the output archive.
        variant : str
            The file extension prefix to use for the output archive.
            Example: `".ome"` for OME-Zarr files.
        """
        if self.out is not None:
            return
        self.out = name
        self.out += f"{self.variant}.zarr"
