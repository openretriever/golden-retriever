"""
Type Registry System for External Type Registration

Allows external packages to register custom types that work seamlessly
with Retriever's Flow system and Dora code generation.
"""

from typing import Dict, Type, Any, Optional, Callable
from dataclasses import dataclass
import inspect
from importlib import import_module

@dataclass
class TypeInfo:
    """Information about a registered type."""
    type_class: Type
    name: str
    module: str
    serializable: bool = True
    arrow_converter: Optional[Callable] = None
    description: str = ""
    category: str = "general"
    tags: list = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class TypeRegistry:
    """Global registry for Retriever types."""
    
    def __init__(self):
        self._types: Dict[str, TypeInfo] = {}
        self._type_to_name: Dict[Type, str] = {}
    
    def register(self, 
                 type_class: Type, 
                 name: Optional[str] = None,
                 arrow_converter: Optional[Callable] = None,
                 description: str = "",
                 category: str = "general",
                 tags: Optional[list] = None) -> Type:
        """
        Register a type for use in Retriever flows.
        
        Args:
            type_class: The class to register
            name: Optional name (defaults to class.__name__)
            arrow_converter: Optional function to convert to PyArrow format
            description: Human-readable description
            
        Returns:
            The registered type class (for use as decorator)
            
        Example:
            @register_type
            class MyType:
                pass
                
            # Or with options
            @register_type(name="CustomName", description="My custom type")  
            class MyType:
                pass
        """
        if name is None:
            name = type_class.__name__
            
        # Check for conflicts
        if name in self._types:
            existing = self._types[name]
            if existing.type_class != type_class:
                raise ValueError(f"Type name '{name}' already registered for {existing.type_class}")
            return type_class  # Already registered
            
        # Register the type
        type_info = TypeInfo(
            type_class=type_class,
            name=name,
            module=type_class.__module__,
            arrow_converter=arrow_converter,
            description=description,
            category=category,
            tags=tags or []
        )
        
        self._types[name] = type_info
        self._type_to_name[type_class] = name
        
        # Add metadata to the class
        type_class._retriever_type_name = name
        type_class._retriever_registered = True
        
        return type_class
    
    def get_type_info(self, name_or_type) -> Optional[TypeInfo]:
        """Get type information by name or type class."""
        if isinstance(name_or_type, str):
            return self._types.get(name_or_type)
        elif isinstance(name_or_type, type):
            name = self._type_to_name.get(name_or_type)
            return self._types.get(name) if name else None
        else:
            return None
    
    def is_registered(self, type_class: Type) -> bool:
        """Check if a type is registered."""
        return type_class in self._type_to_name
    
    def get_registered_types(self) -> Dict[str, TypeInfo]:
        """Get all registered types."""
        return self._types.copy()
    
    def get_arrow_converter(self, type_class: Type) -> Optional[Callable]:
        """Get PyArrow converter for a type."""
        type_info = self.get_type_info(type_class)
        return type_info.arrow_converter if type_info else None


# Global registry instance
_global_registry = TypeRegistry()
_did_bootstrap_builtin_types = False


def _bootstrap_builtin_types() -> None:
    """Import built-in type modules once so registry lookups are stable."""
    global _did_bootstrap_builtin_types
    if _did_bootstrap_builtin_types:
        return

    root_package = __name__.split(".", 1)[0]
    modules = (
        f"{root_package}.types.core_types",
        f"{root_package}.types.vision_types",
        f"{root_package}.types.robotics_types",
        f"{root_package}.robotics_typing.v1",
    )
    for module_name in modules:
        try:
            import_module(module_name)
        except Exception:
            # Keep lazy bootstrap best-effort; missing optional modules should not hard fail.
            pass
    _did_bootstrap_builtin_types = True

def register_type(name_or_class=None, **kwargs):
    """
    Register a type with the global registry.
    
    Can be used as a decorator with or without arguments:
    
    @register_type
    class MyType: pass
    
    @register_type("CustomName", description="My type")
    class MyType: pass
    """
    def decorator(cls):
        return _global_registry.register(cls, name_or_class, **kwargs)
    
    # Handle both @register_type and @register_type(...) syntax
    if name_or_class is not None and inspect.isclass(name_or_class):
        # Used as @register_type (without parentheses)
        return _global_registry.register(name_or_class)
    else:
        # Used as @register_type(...) (with parentheses)
        return decorator

def get_registered_types() -> Dict[str, TypeInfo]:
    """Get all registered types from global registry."""
    return _global_registry.get_registered_types()

def is_registered_type(type_class: Type) -> bool:
    """Check if a type is registered."""
    return _global_registry.is_registered(type_class)

def get_type_name(type_class: Type) -> Optional[str]:
    """Get the registered name for a type."""
    info = _global_registry.get_type_info(type_class)
    return info.name if info else None

def get_type(name: str) -> Type:
    """Get a registered type by name - PyTorch-style access.
    
    Args:
        name: The registered type name
        
    Returns:
        The type class
        
    Raises:
        ValueError: If type is not found
        
    Example:
        # Register a type
        @register_type("pose_3d", category="geometry")
        class Pose3D: pass
        
        # Get it later
        pose_type = get_type("pose_3d")
        instance = pose_type(x=1, y=2, z=3)
    """
    info = _global_registry.get_type_info(name)
    if info is None:
        _bootstrap_builtin_types()
        info = _global_registry.get_type_info(name)
    if info is None:
        available = list(_global_registry._types.keys())
        raise ValueError(f"Type '{name}' not found. Available types: {available}")
    return info.type_class

def list_types(category: Optional[str] = None) -> Dict[str, TypeInfo]:
    """List all registered types, optionally filtered by category.
    
    Args:
        category: Optional category filter
        
    Returns:
        Dictionary mapping type names to TypeInfo objects
    """
    all_types = _global_registry.get_registered_types()
    if category is None:
        return all_types
    return {name: info for name, info in all_types.items() 
            if getattr(info, 'category', None) == category}

def find_types(base_class: Optional[Type] = None, 
               category: Optional[str] = None,
               tags: Optional[list] = None) -> Dict[str, TypeInfo]:
    """Find types matching specific criteria.
    
    Args:
        base_class: Filter by inheritance from this base class
        category: Filter by category
        tags: Filter by tags (all must match)
        
    Returns:
        Dictionary of matching types
    """
    all_types = _global_registry.get_registered_types()
    results = {}
    
    for name, info in all_types.items():
        # Check base class filter
        if base_class and not issubclass(info.type_class, base_class):
            continue
            
        # Check category filter  
        if category and getattr(info, 'category', None) != category:
            continue
            
        # Check tags filter
        if tags:
            type_tags = getattr(info, 'tags', [])
            if not all(tag in type_tags for tag in tags):
                continue
                
        results[name] = info
    
    return results

def get_global_registry() -> TypeRegistry:
    """Get the global type registry instance."""
    return _global_registry

def convert_to_arrow(obj: Any) -> Any:
    """Convert object to PyArrow format using registered converter."""
    converter = _global_registry.get_arrow_converter(type(obj))
    return converter(obj) if converter else obj

def convert_from_arrow(arrow_obj: Any, target_type: Type) -> Any:
    """Convert from PyArrow format to target type (placeholder implementation)."""
    # TODO: Implement reverse conversion when needed
    return arrow_obj
