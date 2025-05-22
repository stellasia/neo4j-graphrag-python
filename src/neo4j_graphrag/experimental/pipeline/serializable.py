"""
Serializable base class for pipeline components, templates, and configurations.

This module provides a consistent serialization interface that enables saving
and loading of pipeline configurations to/from JSON.
"""

from __future__ import annotations

import json
import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, ClassVar, Type, Optional, TypeVar
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Serializable')


class Serializable(ABC):
    """Base class for objects that can be serialized to/from JSON.
    
    This provides a consistent interface for serialization across
    all components, pipelines, and templates in the system.
    """
    
    # Registry of types that can be serialized/deserialized
    # This allows for automatic registration of subclasses
    _registry: ClassVar[Dict[str, Type[Serializable]]] = {}
    
    def __init_subclass__(cls, **kwargs):
        """Register subclasses in the registry."""
        super().__init_subclass__(**kwargs)
        cls._registry[f"{cls.__module__}.{cls.__qualname__}"] = cls
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to a serializable dictionary.
        
        Returns:
            Dictionary representation of the object
        """
        # Base implementation captures class type for reconstruction
        result = {
            "__type__": f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        }
        
        # Add serialized data from subclass implementation
        result.update(self._serialize())
        return result
    
    @abstractmethod
    def _serialize(self) -> Dict[str, Any]:
        """Implement the actual serialization in subclasses.
        
        Returns:
            Dictionary with serialized data
        """
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Serializable:
        """Create an object from a dictionary representation.
        
        Args:
            data: Dictionary with serialized object data
            
        Returns:
            Reconstructed object
        """
        # Get the specific class type from the serialized data
        if "__type__" in data:
            class_path = data["__type__"]
            target_cls = cls._get_class(class_path)
        else:
            # If no type specified, use the current class
            target_cls = cls
        
        # Create the instance using the class's _deserialize method
        return target_cls._deserialize(data)
    
    @classmethod
    @abstractmethod
    def _deserialize(cls, data: Dict[str, Any]) -> Serializable:
        """Implement deserialization in subclasses.
        
        Args:
            data: Dictionary with serialized data
            
        Returns:
            Reconstructed instance
        """
        pass
    
    @classmethod
    def _get_class(cls, class_path: str) -> Type[Serializable]:
        """Get a class by its fully qualified path.
        
        Args:
            class_path: Fully qualified class path (module.Class)
            
        Returns:
            Class type
        """
        # First check the registry
        if class_path in cls._registry:
            return cls._registry[class_path]
        
        # If not in registry, dynamically import
        try:
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load class {class_path}: {e}")
            raise ValueError(f"Cannot deserialize {class_path}: {e}") from e
    
    def to_json(self, filepath: Optional[str] = None, indent: int = 2) -> Optional[str]:
        """Serialize object to JSON.
        
        Args:
            filepath: Optional path to save the JSON to
            indent: Indentation level for the JSON output
            
        Returns:
            JSON string if filepath is None, otherwise None
        """
        serialized = self.to_dict()
        
        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(serialized, f, indent=indent)
            return None
        else:
            return json.dumps(serialized, indent=indent)
    
    @classmethod
    def from_json(cls: Type[T], json_data: str, filepath: bool = False) -> T:
        """Deserialize object from JSON.
        
        Args:
            json_data: JSON string or filepath
            filepath: If True, json_data is treated as a filepath
            
        Returns:
            Reconstructed object
        """
        if filepath:
            # Load from file
            file_path = json_data
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            # Check if json_data is actually a file path
            if Path(json_data).exists():
                with open(json_data, "r") as f:
                    data = json.load(f)
            else:
                # Parse JSON string
                data = json.loads(json_data)
        
        return cls.from_dict(data)


def serialize_object(obj: Any) -> Any:
    """Helper function to serialize any object for JSON storage.
    
    Returns serializable representations of:
    - Serializable objects (calls to_dict)
    - Dictionaries (recursively processes values)
    - Lists/tuples (recursively processes items)
    - Basic JSON types (returns as-is)
    - Other objects (converts to string)
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, Serializable):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: serialize_object(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_object(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # For other objects (like custom classes) try to get dict representation
        if hasattr(obj, '__dict__'):
            return {
                "__type__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "data": serialize_object(obj.__dict__)
            }
        # As a fallback, convert to string
        return str(obj)


def deserialize_object(data: Any) -> Any:
    """Helper function to deserialize objects from JSON data.
    
    Handles:
    - Dictionaries with __type__ keys (reconstructs objects)
    - Regular dictionaries (recursively processes values)
    - Lists (recursively processes items)
    - Basic JSON types (returns as-is)
    
    Args:
        data: Serialized data to deserialize
        
    Returns:
        Deserialized object
    """
    if isinstance(data, dict):
        if "__type__" in data:
            try:
                # This might be a Serializable object
                cls_path = data["__type__"]
                
                # Check if this is a simple object with just a __dict__
                if "data" in data and len(data) == 2:
                    # Try to recreate an object from its __dict__
                    try:
                        module_name, class_name = cls_path.rsplit(".", 1)
                        module = importlib.import_module(module_name)
                        cls = getattr(module, class_name)
                        if issubclass(cls, Serializable):
                            # If it's a Serializable, use proper deserialization
                            return cls.from_dict(data)
                        else:
                            # Otherwise, create an instance and set attributes
                            obj = cls.__new__(cls)
                            for key, value in deserialize_object(data["data"]).items():
                                setattr(obj, key, value)
                            return obj
                    except (ImportError, AttributeError) as e:
                        logger.warning(f"Failed to deserialize {cls_path}: {e}")
                        # Return the dictionary as is
                        return {k: deserialize_object(v) for k, v in data.items()}
                else:
                    # This is a Serializable object
                    try:
                        return Serializable.from_dict(data)
                    except (ImportError, ValueError) as e:
                        logger.warning(f"Failed to deserialize object: {e}")
                        # Fall back to returning the dictionary
                        return {k: deserialize_object(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Error during deserialization: {e}")
                # Return as dictionary if anything goes wrong
                return {k: deserialize_object(v) for k, v in data.items()}
        else:
            # Regular dictionary, deserialize values
            return {k: deserialize_object(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deserialize_object(i) for i in data]
    else:
        # Basic types (str, int, float, bool, None) are returned as-is
        return data 