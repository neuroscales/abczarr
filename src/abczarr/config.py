"""Configuration related to output Zarr Archive."""

# stdlib
from types import NoneType

# dependencies
import typing_extensions as tx

# locals
from ._core import typing as tz
from ._core.attrs import autodefine, field


@autodefine
class ZarrConfig:
    """
    Configuration related to output Zarr archives.

    Parameters
    ----------
    zarr_version
        Zarr version to use. If `shard` is used, 3 is required.
    chunk
        Output chunk size.
        Will be copy-padded to match the number of dimensions of the data.
    shard
        Output shard size.
        If `"auto"`, find shard size that ensures files smaller than 2TB,
        assuming a compression ratio or 2.
    dimension_separator
        The separator placed between the dimensions of a chunk.
    order:
        Memory layout order for the data array.
    compressor
        Compression method
    compressor_opt
        Compression options

    """
    zarr_version: tz.ZarrVersion = 3
    chunk: tz.Shape = (128,)
    shard: tx.Union[tz.Shape, tx.Literal["auto"], NoneType] = None
    dimension_separator: tz.DimensionSeparator = "/"
    order: tz.MemoryOrder = "C"
    compressor: tz.CompressorType = "blosc"
    compressor_opt: tz.CompressorOptions = field(default_factory=dict)
    overwrite: bool = False
    driver: tz.AnyDriver = "zarr-python"

    def __post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Ensure that sharding options (shard, shard_channels, shard_time)
          are only used when zarr_version == 3;
          otherwise raise NotImplementedError.
        """
        if self.zarr_version < 3:
            if self.shard:
                raise ValueError("Shard is not supported for Zarr < 3.")


@autodefine
class OMEZarrConfig(ZarrConfig):
    """
    Configuration related to output OME Zarr archives.

    Parameters
    ----------
    chunk
        Output chunk size.
        Behavior depends on the number of values provided:
        * one:   used for all spatial dimensions
        * three: used for spatial dimensions ([z, y, x])
        * four+:  used for channels and spatial dimensions ([c, z, y, x])
        If `"auto"`, find chunk size smaller than 1 MB (TODO: not implemented)
    zarr_version
        Zarr version to use. If `shard` is used, 3 is required.
    chunk_channels
        Put channels in different chunk.
        If False, combine all channels in a single chunk.
    chunk_time
        Put time points in different chunk.
        If False, combine all time points in a single chunk.
    shard
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

    def __post_init__(self) -> None:
        """
        Perform post-initialization checks and adjustments.

        - Ensure that sharding options (shard, shard_channels, shard_time)
          are only used when zarr_version == 3;
          otherwise raise NotImplementedError.
        """
        if self.zarr_version < 3:
            if self.shard or self.shard_channels or self.shard_time:
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

    def __post_init__(self) -> None:
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
