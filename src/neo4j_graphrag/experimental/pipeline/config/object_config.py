#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Config for all parameters that can be both provided as object instance or
config dict with 'class_' and 'params_' keys.

Nomenclature in this file:

- `*Config` models are used to represent "things" as dict to be used in a config file.
    e.g.:
    - neo4j.Driver => {"uri": "", "user": "", "password": ""}
    - LLMInterface => {"class_": "OpenAI", "params_": {"model_name": "gpt-4o"}}
- `*Type` models are wrappers around an object and a 'Config' the object can be created
    from. They are used to allow the instantiation of "PipelineConfig" either from
    instantiated objects (when used in code) and from a config dict (when used to
    load config from file).
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import (
    Any,
    ClassVar,
    Generic,
    Optional,
    TypeVar,
    Union,
    cast,
)

import neo4j
from pydantic import (
    ConfigDict,
    Field,
    RootModel,
    field_validator,
)

from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.pipeline import Component
from neo4j_graphrag.experimental.pipeline.config.base import AbstractConfig
from neo4j_graphrag.experimental.pipeline.config.param_resolver import (
    ParamConfig,
)
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.utils.validation import issubclass_safe


logger = logging.getLogger(__name__)


T = TypeVar("T")
"""Generic type to help mypy with the parse method when we know the exact
expected return type (e.g. for the Neo4jDriverConfig below).
"""


class LazyNeo4jDriver:
    """A Neo4j driver that's created lazily in the worker process.
    
    This class stores the configuration needed to create a driver but doesn't
    actually create it until it's used. This ensures the driver is created
    in the worker process, not the main process, avoiding serialization issues.
    
    This integrates with the existing pipeline configuration system.
    """
    
    def __init__(self, uri: str, user: str, password: str, **driver_kwargs):
        self._uri = uri
        self._user = user  
        self._password = password
        self._driver_kwargs = driver_kwargs
        self._actual_driver: Optional[neo4j.Driver] = None
        self._created_in_process_id = None
    
    def _ensure_driver(self) -> neo4j.Driver:
        """Ensure we have an actual driver, creating it if necessary."""
        current_process_id = os.getpid()
        
        # Create driver if it doesn't exist or we're in a different process (worker)
        if self._actual_driver is None or self._created_in_process_id != current_process_id:
            from neo4j_graphrag.utils import driver_config
            
            if self._actual_driver is not None:
                # Close old driver if switching processes
                try:
                    self._actual_driver.close()
                except:
                    pass
            
            self._actual_driver = neo4j.GraphDatabase.driver(
                self._uri, 
                auth=(self._user, self._password), 
                **self._driver_kwargs
            )
            
            # Apply user agent override if it was stored in driver_kwargs
            if 'user_agent' in self._driver_kwargs:
                user_agent = self._driver_kwargs.pop('user_agent')  # Remove to avoid conflicts
                self._actual_driver._pool.pool_config.user_agent = user_agent
            else:
                # Apply default user agent override
                self._actual_driver = driver_config.override_user_agent(self._actual_driver)
            
            self._created_in_process_id = current_process_id
            
            logger.debug(f"Created Neo4j driver in process {current_process_id}")
        
        return self._actual_driver
    
    def close(self) -> None:
        """Close the actual driver if it exists."""
        if self._actual_driver is not None:
            self._actual_driver.close()
            self._actual_driver = None
    
    # Delegate all neo4j.Driver methods to the actual driver
    def __getattr__(self, name: str) -> Any:
        """Delegate to the actual Neo4j driver, creating it if necessary."""
        return getattr(self._ensure_driver(), name)
    
    # Support serialization for distributed execution
    def __getstate__(self) -> dict:
        """Custom serialization - only serialize config, not the actual driver."""
        return {
            '_uri': self._uri,
            '_user': self._user,
            '_password': self._password,
            '_driver_kwargs': self._driver_kwargs,
            # Don't serialize the actual driver or process ID
        }
    
    def __setstate__(self, state: dict) -> None:
        """Custom deserialization - restore config, driver will be created on demand."""
        self._uri = state['_uri']
        self._user = state['_user']
        self._password = state['_password']
        self._driver_kwargs = state['_driver_kwargs']
        self._actual_driver = None
        self._created_in_process_id = None


class ObjectConfig(AbstractConfig, Generic[T]):
    """A config class to represent an object from a class name
    and its constructor parameters.
    """

    class_: str | None = Field(default=None, validate_default=True)
    """Path to class to be instantiated."""
    params_: dict[str, ParamConfig] = {}
    """Initialization parameters."""

    DEFAULT_MODULE: ClassVar[str] = "."
    """Default module to import the class from."""
    INTERFACE: ClassVar[type] = object
    """Constraint on the class (must be a subclass of)."""
    REQUIRED_PARAMS: ClassVar[list[str]] = []
    """List of required parameters for this object constructor."""

    @field_validator("params_")
    @classmethod
    def validate_params(cls, params_: dict[str, Any]) -> dict[str, Any]:
        """Make sure all required parameters are provided."""
        for p in cls.REQUIRED_PARAMS:
            if p not in params_:
                raise ValueError(f"Missing parameter {p}")
        return params_

    def get_module(self) -> str:
        return self.DEFAULT_MODULE

    def get_interface(self) -> type:
        return self.INTERFACE

    @classmethod
    def _get_class(cls, class_path: str, optional_module: Optional[str] = None) -> type:
        """Get class from string and an optional module

        Will first try to import the class from `class_path` alone. If it results in an ImportError,
        will try to import from `f'{optional_module}.{class_path}'`

        Args:
            class_path (str): Class path with format 'my_module.MyClass'.
            optional_module (Optional[str]): Optional module path. Used to provide a default path for some known objects and simplify the notation.

        Raises:
            ValueError: if the class can't be imported, even using the optional module.
        """
        *modules, class_name = class_path.rsplit(".", 1)
        module_name = modules[0] if modules else optional_module
        if module_name is None:
            raise ValueError("Must specify a module to import class from")
        try:
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)
        except (ImportError, AttributeError):
            if optional_module and module_name != optional_module:
                full_klass_path = optional_module + "." + class_path
                return cls._get_class(full_klass_path)
            raise ValueError(f"Could not find {class_name} in {module_name}")
        return cast(type, klass)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> T:
        """Import `class_`, resolve `params_` and instantiate object."""
        self._global_data = resolved_data or {}
        logger.debug(f"OBJECT_CONFIG: parsing {self} using {resolved_data}")
        if self.class_ is None:
            raise ValueError(f"`class_` is required to parse object {self}")
        klass = self._get_class(self.class_, self.get_module())
        if not issubclass_safe(klass, self.get_interface()):
            raise ValueError(
                f"Invalid class '{klass}'. Expected a subclass of '{self.get_interface()}'"
            )
        params = self.resolve_params(self.params_)
        try:
            obj = klass(**params)
        except TypeError as e:
            logger.error(
                "OBJECT_CONFIG: failed to instantiate object due to improperly configured parameters"
            )
            raise e
        return cast(T, obj)


class Neo4jDriverConfig(ObjectConfig[neo4j.Driver]):
    """Configuration for Neo4j drivers with support for distributed execution.
    
    This configuration can create either regular drivers (for local execution)
    or lazy drivers (for distributed execution) based on the lazy parameter.
    """
    
    REQUIRED_PARAMS = ["uri", "user", "password"]
    
    # Add lazy parameter to control driver creation behavior
    lazy: bool = Field(default=True, description="Create lazy driver for distributed execution")

    @field_validator("class_", mode="before")
    @classmethod
    def validate_class(cls, class_: Any) -> str:
        """`class_` parameter is not used because we're always using the sync driver."""
        if class_:
            logger.info("Parameter class_ is not used for Neo4jDriverConfig")
        # not used
        return "not used"

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Union[neo4j.Driver, LazyNeo4jDriver]:
        """Parse the configuration into either a regular or lazy Neo4j driver."""
        params = self.resolve_params(self.params_)
        # we know these params are there because of the required params validator
        uri = params.pop("uri")
        user = params.pop("user")
        password = params.pop("password")
        
        if self.lazy:
            # Create lazy driver for distributed execution
            logger.debug("Creating lazy Neo4j driver for distributed execution")
            return LazyNeo4jDriver(uri, user, password, **params)
        else:
            # Create regular driver for local execution
            from neo4j_graphrag.utils import driver_config
            driver = neo4j.GraphDatabase.driver(uri, auth=(user, password), **params)
            return driver_config.override_user_agent(driver)


# note: using the notation with RootModel + root: <type> field
# instead of RootModel[<type>] for clarity
# but this requires the type: ignore comment below
class Neo4jDriverType(RootModel):  # type: ignore[type-arg]
    """A model to wrap neo4j.Driver, LazyNeo4jDriver and Neo4jDriverConfig objects.

    The `parse` method always returns a neo4j.Driver or LazyNeo4jDriver.
    """

    root: Union[neo4j.Driver, LazyNeo4jDriver, Neo4jDriverConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Union[neo4j.Driver, LazyNeo4jDriver]:
        if isinstance(self.root, (neo4j.Driver, LazyNeo4jDriver)):
            return self.root
        # self.root is a Neo4jDriverConfig object
        return self.root.parse(resolved_data)


class LLMConfig(ObjectConfig[LLMInterface]):
    """Configuration for any LLMInterface object with support for distributed execution.

    By default, will try to import from `neo4j_graphrag.llm`.
    """

    DEFAULT_MODULE = "neo4j_graphrag.llm"
    INTERFACE = LLMInterface
    
    # Add lazy parameter to control LLM creation behavior
    lazy: bool = Field(default=True, description="Create lazy LLM for distributed execution")
    
    def parse(self, resolved_data: dict[str, Any] | None = None) -> Union[LLMInterface, Any]:
        """Parse the configuration into either a regular or lazy LLM."""
        self._global_data = resolved_data or {}
        logger.debug(f"OBJECT_CONFIG: parsing {self} using {resolved_data}")
        if self.class_ is None:
            raise ValueError(f"`class_` is required to parse object {self}")
        
        params = self.resolve_params(self.params_)
        
        # Check if this is an OpenAI LLM and we want lazy loading
        if self.lazy and self.class_ in ["OpenAILLM", "AzureOpenAILLM"]:
            from neo4j_graphrag.llm.openai_llm import LazyOpenAILLM
            logger.debug("Creating lazy OpenAI LLM for distributed execution")
            azure = self.class_ == "AzureOpenAILLM"
            return LazyOpenAILLM(azure=azure, **params)
        else:
            # Use regular creation for other LLMs or when lazy=False
            klass = self._get_class(self.class_, self.get_module())
            if not issubclass_safe(klass, self.get_interface()):
                raise ValueError(
                    f"Invalid class '{klass}'. Expected a subclass of '{self.get_interface()}'"
                )
            try:
                obj = klass(**params)
            except TypeError as e:
                logger.error(
                    "OBJECT_CONFIG: failed to instantiate object due to improperly configured parameters"
                )
                raise e
            return cast(LLMInterface, obj)


class EmbedderConfig(ObjectConfig[Embedder]):
    """Configuration for any Embedder object with support for distributed execution.

    By default, will try to import from `neo4j_graphrag.embeddings`.
    """

    DEFAULT_MODULE = "neo4j_graphrag.embeddings"
    INTERFACE = Embedder
    
    # Add lazy parameter to control embedder creation behavior
    lazy: bool = Field(default=True, description="Create lazy embedder for distributed execution")
    
    def parse(self, resolved_data: dict[str, Any] | None = None) -> Union[Embedder, Any]:
        """Parse the configuration into either a regular or lazy embedder."""
        self._global_data = resolved_data or {}
        logger.debug(f"OBJECT_CONFIG: parsing {self} using {resolved_data}")
        if self.class_ is None:
            raise ValueError(f"`class_` is required to parse object {self}")
        
        params = self.resolve_params(self.params_)
        
        # Check if this is an OpenAI embedder and we want lazy loading
        if self.lazy and self.class_ in ["OpenAIEmbeddings", "AzureOpenAIEmbeddings"]:
            from neo4j_graphrag.embeddings.openai import LazyOpenAIEmbeddings
            logger.debug("Creating lazy OpenAI embeddings for distributed execution")
            azure = self.class_ == "AzureOpenAIEmbeddings"
            return LazyOpenAIEmbeddings(azure=azure, **params)
        else:
            # Use regular creation for other embedders or when lazy=False
            klass = self._get_class(self.class_, self.get_module())
            if not issubclass_safe(klass, self.get_interface()):
                raise ValueError(
                    f"Invalid class '{klass}'. Expected a subclass of '{self.get_interface()}'"
                )
            try:
                obj = klass(**params)
            except TypeError as e:
                logger.error(
                    "OBJECT_CONFIG: failed to instantiate object due to improperly configured parameters"
                )
                raise e
            return cast(Embedder, obj)


class LLMType(RootModel):  # type: ignore[type-arg]
    """A model to wrap LLMInterface, LazyOpenAILLM and LLMConfig objects.

    The `parse` method always returns a LLMInterface or lazy equivalent.
    """

    root: Union[LLMInterface, LLMConfig, Any]  # Any for lazy types

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Union[LLMInterface, Any]:
        if isinstance(self.root, LLMInterface) or hasattr(self.root, '_ensure_llm'):
            return self.root
        # self.root is a LLMConfig object
        return self.root.parse(resolved_data)


class EmbedderType(RootModel):  # type: ignore[type-arg]
    """A model to wrap Embedder, LazyOpenAIEmbeddings and EmbedderConfig objects.

    The `parse` method always returns a Embedder or lazy equivalent.
    """

    root: Union[Embedder, EmbedderConfig, Any]  # Any for lazy types

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Union[Embedder, Any]:
        if isinstance(self.root, Embedder) or hasattr(self.root, '_ensure_embedder'):
            return self.root
        # self.root is a EmbedderConfig object
        return self.root.parse(resolved_data)


class ComponentConfig(ObjectConfig[Component]):
    """A config model for all components.

    In addition to the object config, components can have pre-defined parameters
    that will be passed to the `run` method, ie `run_params_`.
    """

    run_params_: dict[str, ParamConfig] = {}

    DEFAULT_MODULE = "neo4j_graphrag.experimental.components"
    INTERFACE = Component

    def get_run_params(self, resolved_data: dict[str, Any]) -> dict[str, Any]:
        self._global_data = resolved_data
        return self.resolve_params(self.run_params_)


class ComponentType(RootModel):  # type: ignore[type-arg]
    root: Union[Component, ComponentConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> Component:
        if isinstance(self.root, Component):
            return self.root
        return self.root.parse(resolved_data)

    def get_run_params(self, resolved_data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.root, Component):
            return {}
        return self.root.get_run_params(resolved_data)
