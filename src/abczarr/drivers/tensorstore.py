"""TensorStore driver for Zarr arrays and groups."""

# stdlib
import json
import os
from urllib.parse import urlparse

# dependencies
import numpy as np
import numpy.typing as npt
import typing_extensions as tx

# abczarr
from abczarr._core import typing as tz
from abczarr._core.path import Path
from abczarr.abc import ZarrArray, ZarrArrayConfig, ZarrGroup, ZarrNode
from abczarr._core.attributes import Attributes
from abczarr.config import ZarrConfig
from abczarr._core.sharding import auto_shard, fix_shard_chunk
from abczarr.metadata.base import GroupMetadata
from abczarr.registry import UnavailableDriverError

# optionals
try:
    import tensorstore as ts
except ImportError:
    raise UnavailableDriverError("tensorstore")


class ZarrTSNode(ZarrNode): ...


class ZarrTSArray(ZarrArray, ZarrTSNode):
    """Zarr array backed by TensorStore."""

    def __init__(self, ts_array: ts.TensorStore) -> None:
        """
        Initialize the ZarrTSArray with a TensorStore array.

        Parameters
        ----------
        ts_array : tensorstore.TensorStore
            Underlying TensorStore array.
        """
        super().__init__(str(ts_array.kvstore.path))
        self._ts = ts_array
        self._attrs: tx.Optional[Attributes] = None

    @property
    def ndim(self) -> int:
        """Number of dimensions of the array."""
        return self._ts.ndim

    @property
    def shape(self) -> tz.Shape:
        """Shape of the array."""
        return self._ts.shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of the array."""
        return self._ts.dtype.numpy_dtype

    @property
    def chunks(self) -> tz.Shape:
        """Chunk shape for the array."""
        return self._ts.chunk_layout.read_chunk.shape

    @property
    def shards(self) -> tx.Optional[tz.Shape]:
        """Shard shape, if supported; otherwise None."""
        read_shape = self._ts.chunk_layout.read_chunk.shape
        write_shape = self._ts.chunk_layout.write_chunk.shape
        return None if read_shape == write_shape else write_shape

    @property
    def attrs(self) -> Attributes:
        """Access metadata/attributes for this node."""
        if self._attrs is None:
            self._attrs = Attributes(self, write_through=True)
        return self._attrs

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        """Get the Zarr format version."""
        driver = self._ts.schema.codec.to_json().get("driver", "")
        return 3 if driver == "zarr3" else 2

    def __getitem__(self, key: str) -> npt.ArrayLike:
        """Read data from the array."""
        return self._ts[key].read().result()

    def __setitem__(self, key: str, value: npt.ArrayLike) -> None:
        """Write data to the array."""
        self._ts[key] = value

    @classmethod
    def open(
        cls,
        path: tz.PathLike,
        mode: tz.AccessMode = "a",
        *,
        zarr_version: tz.ZarrVersion = 3,
    ) -> tx.Self:
        """
        Open an existing Zarr array.

        Parameters
        ----------
        path : Union[str, PathLike]
            Path to the array's directory.
        zarr_version : {2, 3}
            Zarr format version to use.
        mode : {'r','r+','a','w','w-'}
            Persistence mode: 'r' means read only (must exist); 'r+' means
            read/write (must exist); 'a' means read/write (create if doesn't
            exist); 'w' means create (overwrite if exists); 'w-' means create
            (fail if exists).

        Returns
        -------
        ZarrTSArray
        """
        create = False
        delete_existing = False
        if mode == "w":
            delete_existing = True
            create = True
        elif mode == "w-" or mode == "a":
            create = True

        spec = {
            "kvstore": make_kvstore(path),
            "driver": "zarr3" if zarr_version == 3 else "zarr",
            "open": True,
            "create": create,
            "delete_existing": delete_existing,
        }
        ts_array = ts.open(spec).result()
        return cls(ts_array)


class ZarrTSGroup(ZarrGroup, ZarrTSNode):
    """Zarr Group implementation using TensorStore as backend."""

    def __init__(self, store_path: tz.PathLike) -> None:
        """
        Initialize the ZarrTSGroup.

        Parameters
        ----------
        store_path : Union[str, PathLike]
            Path to the group's directory.
        """
        super().__init__(store_path)
        self._path: os.PathLike = Path(store_path)
        meta = _detect_metadata(self._path)
        assert meta and meta[0] == "group"
        self._zarr_version: ts.ZarrVersion = meta[1]
        self._attrs: tx.Optional[Attributes] = None
        self._metadata: tx.Optional[GroupMetadata] = None

    @classmethod
    def from_config(cls, out: tz.PathLike, zarr_config: ZarrConfig) -> tx.Self:
        """
        Create a ZarrTSGroup from a configuration object.

        Parameters
        ----------
        zarr_config : ZarrConfig
            Configuration with .out and .zarr_version.

        Returns
        -------
        ZarrTSGroup
        """
        return cls.open(out, mode="a", zarr_version=zarr_config.zarr_version)

    @classmethod
    def open(
        cls,
        path: tz.PathLike,
        mode: tz.AccessMode = "a",
        *,
        zarr_version: tz.ZarrVersion = 3,
    ) -> tx.Self:
        """
        Open or create a Zarr group backed by TensorStore.

        Parameters
        ----------
        path : Union[str, PathLike]
            Path to the Zarr group.
        mode : {'r','r+','a','w','w-'}
            Persistence mode: 'r' means read only (must exist); 'r+' means
            read/write (must exist); 'a' means read/write (create if doesn't
            exist); 'w' means create (overwrite if exists); 'w-' means create
            (fail if exists).
        zarr_version : {2,3}
            Zarr format version.

        Returns
        -------
        ZarrTSGroup
        """
        p = Path(path)
        if mode in ("r", "r+"):
            if not p.exists() or not p.is_dir():
                raise FileNotFoundError(f"Group path '{p}' does not exist")
        elif mode == "w-":
            if p.exists():
                raise FileExistsError(f"Group path '{p}' already exists")
        elif mode == "a":
            if not p.exists():
                _init_group(p, zarr_version)
        elif mode == "w":
            if p.exists():
                p.rmdir(recursive=True)
            _init_group(p, zarr_version)
        else:
            raise ValueError(f"Invalid mode '{mode}'")
        return cls(p)

    @property
    def attrs(self) -> Attributes:
        """Access attributes for this node."""
        if self._attrs is None:
            self._attrs = Attributes(self, write_through=True)
        return self._attrs

    @property
    def metadata(self) -> GroupMetadata:
        """Access metadata for this node."""
        if self._metadata is None:
            self._metadata = GroupMetadata.from_files(self._path)
        return self._metadata

    @property
    def zarr_version(self) -> tz.ZarrVersion:
        """Get the Zarr format version."""
        return self._zarr_version

    def __getitem__(self, key: str) -> ZarrTSNode:
        """Get a subgroup or array by name within this group."""
        meta = _detect_metadata(self._path / key)
        if not meta:
            raise KeyError(f"Key '{key}' not found")
        if meta[0] == "group":
            return ZarrTSGroup(self._path / key)
        return ZarrTSArray.open(self._path / key, zarr_version=meta[1])

    def __setitem__(self, key: str, value: ZarrTSNode) -> None:
        """Set a subgroup or array by name within this group."""
        raise NotImplementedError(
            "Assign to zarr group is not supported with tensorstore."
        )

    def __delitem__(self, key: str) -> None:
        """Delete a subgroup or array by name within this group."""
        target = self._path / key
        if target.exists():
            target.rmdir(recursive=True)

    def keys(self) -> tx.Iterator[str]:
        """Get the names of all subgroups and arrays in this group."""
        return (
            p.name
            for p in self._path.iterdir()
            if p.is_dir() and _detect_metadata(p)
        )

    def __contains__(self, name: str) -> bool:
        """Check whether a subgroup or array exists in this group."""
        p = self._path / name
        return p.exists() and bool(_detect_metadata(p))

    def create_group(self, name: str, overwrite: bool = False) -> tx.Self:
        """
        Create or open a subgroup within this group.

        Parameters
        ----------
        name : str
        overwrite : bool
            If True, delete existing before creating.

        Returns
        -------
        ZarrTSGroup
        """
        mode = "w" if overwrite else "w-"
        return self.open(
            self._path / name, mode=mode, zarr_version=self._zarr_version
        )

    def create_array(
        self,
        name: str,
        shape: tz.ShapeLike,
        dtype: npt.DTypeLike = np.int32,
        *,
        overwrite: bool = True,
        data: tx.Optional[npt.ArrayLike] = None,
        zarr_config: tx.Optional[ZarrConfig] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrTSArray:
        """
        Create a new array within this group.

        Parameters
        ----------
        name : str
        shape : Sequence[int]
        dtype : DTypeLike
        overwrite: bool
        zarr_config : ZarrConfig | None
        data : ArrayLike | None

        Returns
        -------
        ZarrTSArray
        """

        def _normalize_keys(d: dict) -> dict:
            # map plural/common variants -> canonical keys
            mapping = {
                "chunks": "chunk",
                "shards": "shard",
                "compressors": "compressor",
                "compressor_opts": "compressor_opt",
            }
            out = {}
            for k, v in d.items():
                if k in ("chunk_key_encoding", "fill_value"):
                    # explicitly unsupported/ignored
                    continue
                out[mapping.get(k, k)] = v
            # drop Nones so we don't pass them through
            return {k: v for k, v in out.items() if v is not None}

        # Start with defaults from zarr_config (if provided)
        base: dict = {}
        if zarr_config is not None:
            base = _normalize_keys(
                {
                    "chunk": getattr(zarr_config, "chunk", None),
                    "shard": getattr(zarr_config, "shard", None),
                    "compressor": getattr(zarr_config, "compressor", None),
                    "compressor_opt": getattr(
                        zarr_config, "compressor_opt", None
                    ),
                }
            )

        # Normalize kwargs and make them override zarr_config-provided defaults
        kw = _normalize_keys(kwargs)
        merged = {**base, **kw}  # kwargs win

        # Build the write config
        conf = default_write_config(
            self._path / name,
            shape=shape,
            dtype=dtype,
            version=self.zarr_version,
            **merged,
        )

        if overwrite:
            conf.update(delete_existing=True)
        conf.update(create=True)
        arr = ts.open(conf).result()
        if data is not None:
            arr[:] = data
        return ZarrTSArray(arr)

    def create_array_from_base(
        self,
        name: str,
        shape: tz.ShapeLike,
        data: tx.Optional[npt.ArrayLike] = None,
        **kwargs: tx.Unpack[ZarrArrayConfig],
    ) -> ZarrTSArray:
        """
        Create a new array using metadata of an existing base-level array.

        Parameters
        ----------
        name : str
        shape : Sequence[int]
        data : ArrayLike | None

        Returns
        -------
        ZarrTSArray
        """
        base = self["0"]._ts.spec().to_json()
        base["metadata"]["shape"] = shape
        base["kvstore"] = make_kvstore(self._path / name)
        base.update(delete_existing=True, create=True)
        arr = ts.open(base).result()
        if data is not None:
            arr[:] = data
        return ZarrTSArray(arr)


def make_compressor_v2(name: tx.Optional[str], **prm: dict) -> dict:
    """Build compressor dictionary for Zarr v2."""
    name = name.lower()
    if name not in tz.COMPRESSORS_V2:
        raise ValueError("Unknown compressor", name)
    return {"id": name, **prm}


def make_compressor_v3(name: tx.Optional[str], **prm: dict) -> dict:
    """Build compressor dictionary for Zarr v3."""
    name = name.lower()
    if name not in tz.COMPRESSORS_V3:
        raise ValueError("Unknown compressor", name)
    return {"name": name, "configuration": prm}


def make_kvstore(path: tz.PathLike) -> dict:
    """Transform a URI into a kvstore JSON object."""
    path = Path(path)
    protocol = getattr(path, "protocol", "") or "file"

    if protocol in ("file", "memory"):
        return {"driver": protocol, "path": path.path}

    if protocol in ("http", "https"):
        url = urlparse(str(path))
        base_url = f"{url.scheme}://{url.netloc}"
        if url.params:
            base_url += ";" + url.params
        if url.query:
            base_url += "?" + url.query
        if url.fragment:
            base_url += "#" + url.fragment
        return {"driver": "http", "base_url": base_url, "path": url.path}

    if protocol == "gcs":
        url = urlparse(str(path))
        return {"driver": "gcs", "bucket": url.netloc, "path": url.path}

    if protocol == "s3":
        url = urlparse(str(path))
        path = {"path": url.path} if url.path else {}
        return {"driver": "s3", "bucket": url.netloc, **path}

    raise ValueError("Unsupported protocol:", path.protocol)


def default_read_config(path: tz.PathLike) -> dict:
    """
    Generate a TensorStore configuration to read an existing Zarr.

    Parameters
    ----------
    path : PathLike | str
        Path to zarr array.
    """
    path = Path(path)
    protocol = getattr(path, "protocol", "")
    if hasattr(path, "protocol") and not protocol:
        path = Path("file://" + str(path))
    if (path / "zarr.json").exists():
        zarr_version = 3
    elif (path / ".zarray").exists():
        zarr_version = 2
    else:
        raise ValueError("Cannot find zarr.json or .zarray file")
    return {
        "kvstore": make_kvstore(path),
        "driver": "zarr3" if zarr_version == 3 else "zarr",
        "open": True,
        "create": False,
        "delete_existing": False,
    }


def _detect_metadata(
    path: tz.PathLike,
) -> tx.Optional[tx.Tuple[tz.NodeType, tz.ZarrVersion]]:
    """
    Look for Zarr metadata files in `path` and return (node_type, version).

    Checks zarr.json (v3), then .zarray/.zgroup (v2).
    """
    # Zarr v3
    z3 = path / "zarr.json"
    if z3.is_file():
        try:
            meta = json.loads(z3.read_text())
            fmt = meta.get("zarr_format")
            if fmt == 3:
                node = meta.get("node_type", "array")
                if node in ("array", "group"):
                    return node, 3
        except json.JSONDecodeError:
            pass
    # Zarr v2
    for fname, ntype in ((".zarray", "array"), (".zgroup", "group")):
        f = path / fname
        if f.is_file():
            try:
                meta = json.loads(f.read_text())
                if meta.get("zarr_format") == 2:
                    return ntype, 2
            except json.JSONDecodeError:
                pass
    return None


def default_write_config(
    path: tz.PathLike,
    shape: tz.ShapeLike,
    dtype: npt.DTypeLike,
    chunk: tz.ShapeLike = (32,),
    shard: tx.Union[tz.ShapeLike, tx.Literal["auto"], None] = None,
    compressor: tz.CompressorType = "blosc",
    compressor_opt: tx.Optional[tz.AnyCompressorType] = None,
    fill_value: tx.Optional[tz.Value] = 0,
    version: tz.ZarrVersion = 3,
) -> dict:
    """
    Generate a default TensorStore configuration.

    Parameters
    ----------
    chunk : list[int]
        Chunk size.
    shard : list[int], optional
        Shard size. No sharding if `None`.
    compressor : str
        Compressor name
    fill_value:
    version : int
        Zarr version

    Returns
    -------
    config : dict
        Configuration
    """
    path = Path(path)
    protocol = getattr(path, "protocol", "")
    if hasattr(path, "protocol") and not protocol:
        path = Path("file://" + str(path))

    # Format compressor
    if version == 3 and compressor == "zlib":
        compressor = "gzip"
    if version == 2 and compressor == "gzip":
        compressor = "zlib"
    compressor_opt = compressor_opt or {}

    # Prepare chunk size
    if isinstance(chunk, int):
        chunk = [chunk]
    else:
        chunk = list(chunk)
    chunk = chunk[:1] * max(0, len(shape) - len(chunk)) + chunk

    # Prepare shard size
    if shard:
        if shard == "auto":
            shard = auto_shard(shape, dtype)
        if isinstance(shard, int):
            shard = [shard]
        shard = shard[:1] * max(0, len(shape) - len(shard)) + shard

        # Fix incompatibilities
        shard, chunk = fix_shard_chunk(shard, chunk, shape)
    else:
        for i, _ in enumerate(chunk):
            chunk[i] = min(chunk[i], shape[i])
    # ------------------------------------------------------------------
    #   Zarr 3
    # ------------------------------------------------------------------
    if version == 3:
        if compressor and compressor != "raw" and compressor != "none":
            compressor = [make_compressor_v3(compressor, **compressor_opt)]
        else:
            compressor = []

        codec_little_endian = {
            "name": "bytes",
            "configuration": {"endian": "little"},
        }

        if shard:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": shard},
            }

            sharding_codec = {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunk,
                    "codecs": [
                        codec_little_endian,
                        *compressor,
                    ],
                    "index_codecs": [
                        codec_little_endian,
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
            codecs = [sharding_codec]

        else:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": chunk},
            }
            codecs = [
                codec_little_endian,
                *compressor,
            ]

        metadata = {
            "chunk_grid": chunk_grid,
            "codecs": codecs,
            "data_type": np.dtype(dtype).name,
            "fill_value": fill_value,
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": r"/"},
            },
        }
        config = {
            "driver": "zarr3",
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    #   Zarr 2
    # ------------------------------------------------------------------
    else:
        if compressor and compressor != "raw" and compressor != "none":
            compressor = make_compressor_v2(compressor, **compressor_opt)
        else:
            compressor = None
        for i in range(len(shape)):
            if shape[i] < chunk[i]:
                chunk[i] = shape[i]
        metadata = {
            "chunks": chunk,
            "order": "F" if len(shape) >= 2 else "C",
            "dtype": np.dtype(dtype).str,
            "fill_value": fill_value,
            "compressor": compressor,
        }
        config = {
            "driver": "zarr",
            "metadata": metadata,
            "key_encoding": r"/",
        }

    # Prepare store
    config["metadata"]["shape"] = shape
    config["kvstore"] = make_kvstore(path)

    return config


def _init_group(group_path: tz.PathLike, version: tz.ZarrVersion) -> None:
    group_path.mkdir(parents=True, exist_ok=True)
    GroupMetadata(zarr_version=version).to_file(group_path)
