__all__ = [
    "Codec",
    "CodecImpl",
]

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr._core.metadata import Metadata


@autofrozen(extra_items=tz.FrozenJson)
class Codec(Metadata):
    """A `Codec` names a numcodecs codec through ``id`` and carries that
    codec's own parameters.

    A v2 codec's parameters sit directly alongside ``id``, not nested
    under a separate key.

    The three metadata versions spell the same codec differently. A v1
    codec carries its numcodecs id as a class attribute and its
    parameters as fields. A v2 codec carries the id as a field alongside
    its parameters, as shown here. A v3 codec names itself through
    ``name`` and nests its parameters under a separate ``configuration``
    object. The gzip codec, with its single ``level`` parameter, shows
    the three side by side:

    ```pycon
    >>> from abczarr.metadata import v1, v2, v3
    >>> v1.GzipCodecOptions(level=1)
    GzipCodecOptions(level=1)
    >>> v2.GzipCodec(id="gzip", level=1)
    GzipCodec(id='gzip', level=1)
    >>> codec = v3.GzipCodec(name="gzip", configuration={"level": 1})
    >>> codec.name, codec.configuration
    ('gzip', GzipConfig(level=1))

    ```
    """

    id: str
    """The numcodecs id of the codec, such as ``"zlib"`` or
    ``"blosc"``."""

    def to_version(self, version: tz.ZarrVersion) -> "Codec":
        """Convert this codec to another Zarr version.

        Parameters
        ----------
        version : ZarrVersion
            The target Zarr format version: 1, 2 or 3.

        Returns
        -------
        Codec
            The equivalent codec for `version`: this object
            unchanged for version 2, or the corresponding v1 or v3
            codec object otherwise.

        Raises
        ------
        ValueError
            If `version` is not 1, 2 or 3.
        """
        if version == 2:
            return self
        if version == 1:
            # v1 and v2 share the numcodecs model: a v2 codec is a valid v1
            # codec, carried as v1 codec options ({id, **options}).
            from abczarr.metadata.v1 import CodecOptions
            as_dict = self.to_json()
            if isinstance(as_dict, str):
                as_dict = {"id": as_dict}
            return CodecOptions.from_json(as_dict)
        if version == 3:
            from abczarr.metadata.v3 import Codec as CodecV3
            as_dict = self.to_json()
            if isinstance(as_dict, str):
                as_dict = {"id": as_dict}
            else:
                config = as_dict
                as_dict = {"name": config.pop("id"), "configuration": config}
            return CodecV3.from_json(as_dict)
        else:
            raise ValueError(f"Unsupported version: {version}")


@autofrozen(extra_items=False)
class CodecImpl(Codec):
    """This class is the base for a v2 codec whose options are all
    declared, not open-ended."""
