# Nodes

Every object abczarr hands back, an array or a group, is a *node*.
[ZarrNode][abczarr.abc.node.ZarrNode] is the common base: it carries
the store path, the Zarr format version, the metadata, and the user
attributes, and it is where the capability query lives.

[ZarrArray][abczarr.abc.array.ZarrArray] is the n-dimensional node.
Read and write it like a NumPy array, with NumPy-style selections:

```python
array[0, :10]
array[...] = data
```

[ZarrGroup][abczarr.abc.group.ZarrGroup] is the container node. It
holds other nodes by name, arrays and groups alike, and behaves like a
mapping: `group["images"]`, `group.keys()`, `"images" in group`.

## `abczarr.abc.node`

::: abczarr.abc.node

## `abczarr.abc.array`

::: abczarr.abc.array

## `abczarr.abc.group`

::: abczarr.abc.group
