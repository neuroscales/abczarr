__all__ = [
    "Filter",
]

# dependencies
import typing_extensions as tx

# core
from abczarr._core import typing as tz
from abczarr._core.auto.attrs import autofrozen
from abczarr._core.metadata import Metadata


@autofrozen(extra_items=tz.FrozenJson)
class Filter(Metadata):
    """A `Filter` names a numcodecs codec through ``id`` and carries that
    codec's own parameters.

    A filter runs before the compressor, in the order the filters are
    listed. Each filter transforms an array's data on encode and
    reverses that transform on decode.
    """

    id: str

    def to_version(self, version: tz.ZarrVersion) -> tx.Self:
        if version == 2:
            return self
        if version == 3:
            from abczarr.metadata.v3 import Codec as CodecV3
            as_dict = self.to_json()
            if isinstance(as_dict, str):
                config = {}
                id = as_dict
            else:
                config = dict(as_dict)
                id = config.pop("id")
            codec = CodecV3.from_json(
                {"name": id, "configuration": config}
            )
            # A numcodecs filter with no dedicated v3 codec resolves to the
            # generic base codec rather than a modeled subclass. Zarr v3 has
            # no codec plainly named "delta"/"quantize"/"shuffle"; the valid
            # spelling is the numcodecs extension namespace, e.g.
            # "numcodecs.delta", which zarr-python also emits. Re-key it there
            # instead of leaving a bare id that names no v3 codec.
            if type(codec) is CodecV3 and not id.startswith("numcodecs."):
                codec = CodecV3.from_json(
                    {"name": "numcodecs." + id, "configuration": config}
                )
            return codec
        else:
            raise ValueError(f"Unsupported version: {version}")


@autofrozen(extra_items=False)
class FilterImpl(Filter):
    """This class is the base for a v2 filter whose options are declared,
    not open-ended."""
