__all__ = ["OMESchemaItem", "OME"]

import typing_extensions as tx

from abczarr._core import typing as tz

ome_schema_opt = dict(total=False, extra_items=tz.Json)


class OMESchemaItem(tx.TypedDict, **ome_schema_opt):
    ...


class OME(OMESchemaItem):
    ...


class OMEAttributes(OMESchemaItem):
    ...
