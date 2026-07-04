"""Type registry for retriever_typing — a delegation shim.

There is one authoritative type registry in the ecosystem: the runtime's
(`retriever.registry.types`). This module re-exports it so applied Golden
types register alongside the standard types (one namespace, loud conflicts,
no parallel bookkeeping), while keeping the historical
`retriever_typing.registry` API surface working.

Lookups bootstrap the retriever_typing payload modules on first miss so
`get_type("WorldState")` works without callers importing `robotics_types`
first — same behavior the standalone registry had.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from retriever.registry.types import (
    TypeInfo,
    TypeRegistry,
    get_global_registry,
    register_type,
)
from retriever.registry.types import find_types as _core_find_types
from retriever.registry.types import get_registered_types as _core_get_registered_types
from retriever.registry.types import get_type as _core_get_type
from retriever.registry.types import get_type_info as _core_get_type_info
from retriever.registry.types import get_type_name as _core_get_type_name
from retriever.registry.types import is_registered_type as _core_is_registered_type
from retriever.registry.types import list_types as _core_list_types

_bootstrapped = False


def _bootstrap() -> None:
    """Import the payload modules once so their @register_type calls run."""
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    from . import core_types, robotics_types, v1, vision_types  # noqa: F401


def get_type(name: str) -> Type:
    try:
        return _core_get_type(name)
    except ValueError:
        _bootstrap()
        return _core_get_type(name)


def get_type_info(name_or_type: str | Type[Any]) -> TypeInfo:
    if isinstance(name_or_type, str):
        try:
            return _core_get_type_info(name_or_type)
        except ValueError:
            _bootstrap()
            return _core_get_type_info(name_or_type)
    _bootstrap()
    return _core_get_type_info(name_or_type)


def list_types(category: Optional[str] = None) -> Dict[str, TypeInfo]:
    _bootstrap()
    return _core_list_types(category=category)


def get_registered_types(category: Optional[str] = None) -> Dict[str, TypeInfo]:
    _bootstrap()
    return _core_get_registered_types(category)


def is_registered_type(type_class: Type[Any]) -> bool:
    _bootstrap()
    return _core_is_registered_type(type_class)


def get_type_name(type_class: Type[Any]) -> Optional[str]:
    _bootstrap()
    return _core_get_type_name(type_class)


def get_arrow_converter(type_class: Type[Any]) -> Any:
    _bootstrap()
    return get_global_registry().get_arrow_converter(type_class)


def find_types(
    base_class: Optional[Type] = None,
    category: Optional[str] = None,
    tags: Optional[list] = None,
) -> Dict[str, TypeInfo]:
    """Historical Golden signature: optional base_class filter on top of
    the core registry's category/tags filtering."""
    _bootstrap()
    infos = _core_find_types(category=category, tags=tags)
    if base_class is not None:
        infos = {
            name: info
            for name, info in infos.items()
            if isinstance(info.type_class, type) and issubclass(info.type_class, base_class)
        }
    return infos


__all__ = [
    "TypeInfo",
    "TypeRegistry",
    "find_types",
    "get_arrow_converter",
    "get_global_registry",
    "get_registered_types",
    "get_type",
    "get_type_info",
    "get_type_name",
    "is_registered_type",
    "list_types",
    "register_type",
]
