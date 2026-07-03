import importlib

import typing_extensions as tx


def import_symbol(path: str) -> type:
    """Import a symbol given its full path as 'module:attr'."""
    mod_path, _, attr = path.partition(":")
    module = importlib.import_module(mod_path)
    if not attr:
        return module
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise ImportError(f"Cannot import '{attr}' from '{mod_path}'") from e


def import_all(
    modules: tx.List[str],
    namespace: tx.MutableMapping[str, tx.Any],
    package: tx.Optional[str] = None,
    add_to_all: tx.Iterable[tx.Literal["module", "attrs"]] = ("module", "attrs")
) -> None:
    """
    Import all symbols from the given submodules into the provided namespace.
    """
    if isinstance(add_to_all, str):
        add_to_all = (add_to_all,)
    add_to_all = tuple(add_to_all)

    namespace.setdefault("__all__", [])

    for module in modules:
        # Import module
        _module = f".{module}" if package else module
        imported = importlib.import_module(_module, package)
        namespace[module] = imported
        if "module" in add_to_all:
            namespace["__all__"].append(module)

        # Import attributes
        for attr in getattr(imported, "__all__", []):
            namespace[attr] = getattr(imported, attr)
            if "attrs" in add_to_all:
                namespace["__all__"].append(attr)
