"""
Template Builder for Pipeline Construction

This module provides a flexible and reusable template-based approach
for building various pipeline types with a consistent interface.

It includes:
- A generic TemplateBuilder class that works with any pipeline type
- Base classes for pipeline configurations and implementations
- An implementation example for SimpleKGPipeline
- Usage examples showing how to construct pipelines
"""

from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union, Callable, cast, get_args, get_origin
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect

import neo4j

from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor, OnError
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, Neo4jWriter
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader, PdfLoader
from neo4j_graphrag.experimental.components.resolver import (
    EntityResolver, SinglePropertyExactMatchResolver
)
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema, SchemaBuilder, SchemaFromTextExtractor, NodeType, RelationshipType
)
from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig, SchemaEnforcementMode
)
from neo4j_graphrag.experimental.pipeline.component import Component, Serializable
from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline, PipelineResult
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.experimental.pipeline.types.schema import (
    EntityInputType, RelationInputType
)
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.experimental.pipeline.types.definitions import ConnectionDefinition
from neo4j_graphrag.experimental.pipeline.serializable import serialize_object, deserialize_object

logger = logging.getLogger(__name__)

# Generic type for pipeline implementations
P = TypeVar('P', bound='TemplatePipeline')



@dataclass
class ComponentSpec():
    """
    Specification for a pipeline component.
    
    This class encapsulates all aspects of component configuration including
    default class, required interface, parameters, and documentation. It also
    includes the logic to instantiate and validate components.
    
    To customize class resolution, subclass ComponentSpec and override the get_class method.
    
    Args:
        default_class: The default class to use for this component
        required_interface: The interface/base class this component must implement
        params: Parameters to pass to the component constructor
        description: Human-readable description of the component's purpose
    """
    default_class: Type
    required_interface: Type
    params: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def validate_class(self, cls: Type) -> None:
        """
        Validate that a class implements the required interface.
        
        Args:
            cls: The class to validate
            
        Raises:
            TypeError: If the class does not implement the required interface
        """
        if not issubclass(cls, self.required_interface):
            raise TypeError(
                f"Component class must be a subclass of {self.required_interface.__name__}, "
                f"but got {cls.__name__}."
            )
    
    def validate_instance(self, instance: Any) -> None:
        """
        Validate that an instance implements the required interface.
        
        Args:
            instance: The instance to validate
            
        Raises:
            TypeError: If the instance does not implement the required interface
        """
        if not isinstance(instance, self.required_interface):
            raise TypeError(
                f"Component instance must be an instance of {self.required_interface.__name__}, "
                f"but got {type(instance).__name__}."
            )
    
    def get_class(self, pipeline: 'TemplatePipeline', component_name: str) -> Type:
        """
        Get the component class to use.
        
        This method can be overridden in subclasses to provide dynamic class resolution
        based on configuration or other factors.
        
        Args:
            pipeline: The pipeline instance
            component_name: The name of the component
            
        Returns:
            The component class to instantiate
        """
        return self.default_class
    
    def get_component_class(self, pipeline: 'TemplatePipeline', component_name: str) -> Type:
        """
        Get and validate the component class to use.
        
        This method calls get_class() to determine the class, then validates it.
        
        Args:
            pipeline: The pipeline instance
            component_name: The name of the component
            
        Returns:
            The validated component class to instantiate
        """
        # Get the class from the customizable get_class method
        component_class = self.get_class(pipeline, component_name)
            
        # Validate that the component class matches the required interface
        self.validate_class(component_class)
        
        return component_class
    
    def resolve_parameters(self, 
                          pipeline: 'TemplatePipeline', 
                          user_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Resolve all parameters for the component, including config references.
        
        Args:
            pipeline: The pipeline instance
            user_params: User-provided parameter overrides
            
        Returns:
            Dict with all resolved parameters
            
        Raises:
            ValueError: If a required config parameter is None
        """
        params = {}
        missing_config_params = []
        
        # First, resolve parameters defined in the component specification
        for param_name, param_value in self.params.items():
            if isinstance(param_value, str) and param_value.startswith('$'):
                # This is a config reference - remove the '$' prefix and get from config
                config_attr = param_value[1:]  # Remove the '$' prefix
                config_value = getattr(pipeline.config, config_attr)
                
                # Check if the parameter is provided (not None)
                if config_value is None:
                    missing_config_params.append(config_attr)
                else:
                    params[param_name] = config_value
            else:
                # This is a regular parameter
                params[param_name] = param_value
        
        # If we found missing parameters, raise an error
        if missing_config_params:
            raise ValueError(
                f"Component requires the following config parameters that were not provided: "
                f"{missing_config_params}. You must either provide these parameters "
                f"using with_params() or replace the component."
            )
        
        # Then, override with user-provided parameters
        if user_params:
            params.update(user_params)
            
        return params
    
    def instantiate(self, 
                   pipeline: 'TemplatePipeline', 
                   component_name: str,
                   user_params: Dict[str, Any] = None) -> Component:
        """
        Instantiate a component according to this specification.
        
        Args:
            pipeline: The pipeline instance
            component_name: The name of the component
            user_params: User-provided parameter overrides
            
        Returns:
            An instance of the component
            
        Raises:
            ValueError: If a required config parameter is None
            TypeError: If the component doesn't match the required interface
        """
        # Get the component class
        component_class = self.get_component_class(pipeline, component_name)
        
        # Resolve parameters
        params = self.resolve_parameters(pipeline, user_params)
        
        # Create and return the component instance
        return component_class(**params)
    

# Create a specialized ComponentSpec for the schema component
class SchemaComponentSpec(ComponentSpec):
    """
    A specialized ComponentSpec that dynamically selects a schema class
    based on whether a schema is provided in the configuration.
    """
    
    def get_class(self, pipeline: 'TemplatePipeline', component_name: str) -> Type:
        """
        Determine which schema class to use based on configuration.
        
        Returns SchemaBuilder if a schema is provided in the config,
        otherwise returns SchemaFromTextExtractor.
        
        Args:
            pipeline: The pipeline instance
            component_name: The name of the component
            
        Returns:
            The schema component class to use
        """
        # If a schema is provided in the config, use SchemaBuilder
        if pipeline.config.schema is not None:
            return SchemaBuilder
        # Otherwise, extract schema from text
        return SchemaFromTextExtractor


class PipelineConfig(Serializable, ABC):
    """Base class for all pipeline configurations."""
    
    def _serialize(self) -> Dict[str, Any]:
        """Serialize the config to a dictionary.
        
        Returns:
            Dictionary with serialized configuration
        """
        # Serialize all public attributes
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = serialize_object(value)
        return result
    
    @classmethod
    def _deserialize(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Deserialize a configuration from a dictionary.
        
        Args:
            data: Dictionary with serialized configuration
            
        Returns:
            Reconstructed configuration instance
        """
        # Create new instance
        instance = cls.__new__(cls)
        
        # Set attributes from data, skipping the type key
        for key, value in data.items():
            if key != "__type__":
                setattr(instance, key, deserialize_object(value))
                
        return instance



class TemplatePipeline(Serializable, ABC):
    """Abstract base class for all template-based pipelines.
    
    This class defines the interface that all pipeline implementations must follow.
    
    Subclasses must define COMPONENTS to specify what components the pipeline uses
    and how they're configured. This includes:
    
    - The default class to use for each component
    - The required interface each component must implement
    - Default parameters (values prefixed with '$' indicate config references)
    - Documentation for each component
    
    All config parameters are optional at initialization time, but will be required
    if they're referenced by an active component and that component is not replaced.
    
    For dynamic component class resolution, implement methods with the naming pattern
    `get_<component_name>_class()`. These methods will be called automatically when
    determining which class to use for a component, which allows for dynamic class
    selection based on configuration parameters or other runtime conditions.
    
    Example:
    ```python
    class MyPipeline(TemplatePipeline):
        COMPONENTS = {
            "component1": ComponentSpec(
                default_class=Component1Class,
                required_interface=BaseComponent1Interface,
                params={"param1": "$required_param"},  # From config
                description="Processes input data and produces output"
            )
        }
        
        # This method will be called instead of using the default class
        def get_component1_class(self):
            # Dynamically determine which class to use
            if some_condition:
                return AlternativeComponent1Class
            return Component1Class
    ```
    """
    
    # Class variables to be defined by subclasses
    COMPONENTS: Dict[str, ComponentSpec] = {}
    VERSION: str = "1.0.0"  # Initial version
    
    @classmethod
    @abstractmethod
    def get_config_class(cls) -> Type[PipelineConfig]:
        """Get the configuration class for this pipeline.
        
        Returns:
            The configuration class type
        """
        pass
    
    def __init__(
        self,
        config: Any,
        component_configs: Dict[str, Dict[str, Any]],
        component_replacements: Dict[str, Component],
    ):
        """Initialize the pipeline with configuration and component customizations."""
        self.config = config
        self.component_configs = component_configs
        self.component_replacements = component_replacements
        self.pipeline = Pipeline()
        self._setup_pipeline()
    
    def get_component(self, component_name: str) -> Component:
        """
        Get a component instance for the given name.
        
        This method first checks if a replacement exists. If not, it creates 
        the default component with any provided custom configuration.
        
        Args:
            component_name: Name of the component to retrieve
            
        Returns:
            An instance of the component
            
        Raises:
            ValueError: If a required config parameter is None or component not found
            TypeError: If a component doesn't match the required interface
        """
        # If component was explicitly replaced, return the replacement
        if component_name in self.component_replacements:
            component = self.component_replacements[component_name]
            
            # Validate that the replacement component matches the required interface
            if component_name in self.COMPONENTS:
                self.COMPONENTS[component_name].validate_instance(component)
                
            return component
        
        # Get component specification
        if component_name not in self.COMPONENTS:
            raise ValueError(f"Unknown component name: {component_name}")
        
        component_spec = self.COMPONENTS[component_name]
        
        # Get user-provided configuration
        user_params = self.component_configs.get(component_name, {})
        
        # Use the ComponentSpec to instantiate the component
        return component_spec.instantiate(self, component_name, user_params)
    
    @abstractmethod
    def _setup_pipeline(self) -> None:
        """Set up the pipeline with components and connections."""
        pass
    
    def _serialize(self) -> Dict[str, Any]:
        """Serialize the template pipeline to a dictionary.
        
        Returns:
            Dictionary with serialized pipeline data
        """
        return {
            "version": self.VERSION,
            "config": serialize_object(self.config),
            "component_configs": serialize_object(self.component_configs),
            "component_replacements": serialize_object(self.component_replacements)
        }
    
    @classmethod
    def _deserialize(cls, data: Dict[str, Any]) -> TemplatePipeline:
        """Deserialize a template pipeline from a dictionary.
        
        Args:
            data: Dictionary with serialized pipeline data
            
        Returns:
            Reconstructed pipeline instance
        """
        # Deserialize config, component configs, and component replacements
        config = deserialize_object(data.get("config", {}))
        component_configs = deserialize_object(data.get("component_configs", {}))
        component_replacements = deserialize_object(data.get("component_replacements", {}))
        
        # Create and return the pipeline instance
        return cls(
            config=config,
            component_configs=component_configs,
            component_replacements=component_replacements
        )


class TemplateBuilder(Generic[P]):
    """Universal builder for template pipelines.
    
    This class provides a fluent interface for building any template pipeline.
    It intelligently manages parameter requirements based on which components
    are actually used:
    
    - Required parameters are only enforced if they're needed by components 
      that haven't been replaced
    - When you replace a component that requires certain config parameters,
      those parameters become optional if no other component needs them
    
    Example:
        ```python
        # Create a SimpleKGPipeline
        pipeline = (
            TemplateBuilder(SimpleKGPipeline)
            .with_params(llm=my_llm, driver=driver, embedder=embedder)
            .configure_component("writer", batch_size=500)
            .build()
        )
        
        # Create another pipeline type
        another_pipeline = (
            TemplateBuilder(AnotherPipeline)
            .with_params(required_param="value")
            .configure_component("component1", some_param="value")
            .build()
        )
        
        # Replace a component that requires a parameter
        # Note: 'some_required_param' becomes optional!
        another_pipeline_with_replacement = (
            TemplateBuilder(AnotherPipeline)
            .with_params(some_optional_param=False)  # some_required_param not needed!
            .replace_component("component2", custom_component)  
            .build()
        )
        ```
    """
    
    def __init__(self, pipeline_class: Type[P]):
        """Initialize the builder.
        
        Args:
            pipeline_class: The pipeline class to instantiate
        """
        self._pipeline_class = pipeline_class
        self._config_params: Dict[str, Any] = {}
        self._component_configs: Dict[str, Dict[str, Any]] = {}
        self._component_replacements: Dict[str, Component] = {}
        
        # Get configuration class
        self._config_class = pipeline_class.get_config_class()
   
    def with_params(self, **kwargs: Any) -> TemplateBuilder[P]:
        """Set configuration parameters.
        
        Args:
            **kwargs: Named parameters for configuration
            
        Returns:
            The builder instance for method chaining
        """
        self._config_params.update(kwargs)
        return self

    def configure_component(
        self, 
        component_name: str, 
        **kwargs: Any
    ) -> TemplateBuilder[P]:
        """Configure a component with custom parameters.
        
        Args:
            component_name: Name of the component to configure
            **kwargs: Configuration parameters for the component
            
        Returns:
            The builder instance for method chaining
        """
        if component_name not in self._pipeline_class.COMPONENTS:
            raise ValueError(
                f"Unknown component name: {component_name}. "
                f"Valid components are: {list(self._pipeline_class.COMPONENTS.keys())}"
            )
        
        self._component_configs[component_name] = kwargs
        return self
    
    def replace_component(
        self, 
        component_name: str, 
        component: Component
    ) -> TemplateBuilder[P]:
        """Replace a component with a custom implementation.
        
        Args:
            component_name: Name of the component to replace
            component: The custom component instance
            
        Returns:
            The builder instance for method chaining
        """
        if component_name not in self._pipeline_class.COMPONENTS:
            raise ValueError(
                f"Unknown component name: {component_name}. "
                f"Valid components are: {list(self._pipeline_class.COMPONENTS.keys())}"
            )
        
        self._component_replacements[component_name] = component
        return self
        
    def build(self) -> P:
        """Build the pipeline instance.
        
        This method creates a configuration instance and a pipeline instance.
        The pipeline will validate that all required parameters for active
        components are provided when the components are instantiated.
        
        Returns:
            A configured pipeline instance
        """
        # Create configuration instance with all provided parameters
        config = self._config_class(**self._config_params)
        
        # Create pipeline instance
        return cast(P, self._pipeline_class(
            config=config,
            component_configs=self._component_configs,
            component_replacements=self._component_replacements
        ))
        
    def save_to_json(self, filepath: str) -> None:
        """Save the pipeline configuration as a template to a JSON file.
        
        This allows saving the configuration before building the pipeline,
        which can be useful for storing templates that can be loaded later.
        
        Args:
            filepath: Path to save the JSON file
        """
        # Build a temporary pipeline to serialize
        pipeline = self.build()
        
        # Serialize and save
        pipeline.to_json(filepath=filepath)
    
    @classmethod
    def load_from_json(cls, pipeline_class: Type[P], filepath: str) -> P:
        """Load a pipeline from a JSON template file.
        
        Args:
            pipeline_class: The pipeline class to instantiate
            filepath: Path to the JSON template file
            
        Returns:
            An instantiated pipeline
        """
        return pipeline_class.from_json(filepath, filepath=True)


# SimpleKGPipeline implementation example

@dataclass
class SimpleKGPipelineConfig(PipelineConfig):
    """Configuration for SimpleKGPipeline.
    
    This class contains all standard parameters needed to create a SimpleKGPipeline.
    All parameters are optional at initialization time, but will be required
    if used by an active component in the pipeline.
    
    Args:
        llm: An instance of an LLM to use for entity and relation extraction.
        driver: A Neo4j driver instance for database connection.
        embedder: An instance of an embedder used to generate chunk embeddings from text chunks.
        schema: A schema configuration defining entities, relations, and potential schema relationships.
               This is the recommended way to provide schema information.
        enforce_schema: Validation mode for extracted entities/relations. Defaults to "NONE".
        from_pdf: Determines whether to include the PdfLoader in the pipeline.
        perform_entity_resolution: Merge entities with same label and name. Default: True
        lexical_graph_config: Configuration to customize node labels in the lexical graph.
        neo4j_database: The Neo4j database name to use.
    """
    # All parameters are optional at initialization time
    # They will be required later if used by active components
    llm: Optional[LLMInterface] = None
    driver: Optional[neo4j.Driver] = None
    embedder: Optional[Embedder] = None
    
    schema: Optional[Union[GraphSchema, dict[str, list[Any]]]] = None
    enforce_schema: str = "NONE"
    from_pdf: bool = True
    perform_entity_resolution: bool = True
    lexical_graph_config: Optional[LexicalGraphConfig] = None
    neo4j_database: Optional[str] = None


class SimpleKGPipeline(TemplatePipeline):
    """
    A class to simplify the process of building a knowledge graph from text documents.
    It abstracts away the complexity of setting up the pipeline and its components.
    
    Use the TemplateBuilder to create instances of this class:
    
    ```python
    pipeline = (
        TemplateBuilder(SimpleKGPipeline)
        .with_params(llm=my_llm, driver=driver, embedder=embedder)
        .configure_component("writer", batch_size=500)
        .build()
    )
    ```
    """
    
    # Pipeline version - increment when making breaking changes
    VERSION = "1.0.0"
    
    # Define components with their specifications
    COMPONENTS = {
        "pdf_loader": ComponentSpec(
            default_class=PdfLoader,
            required_interface=DataLoader,
            params={},
            description="Loads data from PDF files"
        ),
        "splitter": ComponentSpec(
            default_class=FixedSizeSplitter,
            required_interface=TextSplitter,
            params={},
            description="Splits text into chunks"
        ),
        "chunk_embedder": ComponentSpec(
            default_class=TextChunkEmbedder,
            required_interface=TextChunkEmbedder,
            params={
                "embedder": "$embedder",  # Parameter from config
            },
            description="Embeds text chunks"
        ),
        "schema": SchemaComponentSpec(  # Use the specialized ComponentSpec for schema
            default_class=SchemaBuilder,  # This is the default, but may be overridden by get_class
            required_interface=GraphSchema,
            params={},
            description="Defines the graph schema"
        ),
        "extractor": ComponentSpec(
            default_class=LLMEntityRelationExtractor,
            required_interface=LLMEntityRelationExtractor,
            params={
                "llm": "$llm",
                "prompt_template": "$prompt_template",
                "enforce_schema": "$enforce_schema",
                "on_error": "$on_error",
            },
            description="Extracts entities and relations from text"
        ),
        "writer": ComponentSpec(
            default_class=Neo4jWriter,
            required_interface=KGWriter,
            params={
                "driver": "$driver",
                "neo4j_database": "$neo4j_database",
            },
            description="Writes graph data to Neo4j"
        ),
        "resolver": ComponentSpec(
            default_class=SinglePropertyExactMatchResolver,
            required_interface=EntityResolver,
            params={
                "driver": "$driver",
                "neo4j_database": "$neo4j_database",
            },
            description="Resolves entities with exact matches"
        ),
    }
    
    @classmethod
    def get_config_class(cls) -> Type[PipelineConfig]:
        """Get the configuration class for this pipeline."""
        return SimpleKGPipelineConfig
    
    def _setup_pipeline(self) -> None:
        """Set up the pipeline with components and connections."""
        # Define the components to include
        components_to_add = ["splitter", "chunk_embedder", "schema", "extractor", "writer"]
        
        # Add PDF loader if needed
        if self.config.from_pdf:
            components_to_add.insert(0, "pdf_loader")
        
        # Add resolver if needed
        if self.config.perform_entity_resolution:
            components_to_add.append("resolver")
        
        # Add components to pipeline
        for component_name in components_to_add:
            component = self.get_component(component_name)
            self.pipeline.add_component(component, component_name)
        
        # Set up connections
        connections = []
        
        # PDF loader connections (if used)
        if self.config.from_pdf:
            connections.append(
                ConnectionDefinition(
                    start="pdf_loader",
                    end="splitter",
                    input_config={"text": "pdf_loader.text"},
                )
            )
            # Connect PDF loader to schema for automatic extraction
            connections.append(
                ConnectionDefinition(
                    start="pdf_loader",
                    end="schema",
                    input_config={"text": "pdf_loader.text"},
                )
            )
            # Document info from PDF to extractor
            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={
                        "schema": "schema",
                        "document_info": "pdf_loader.document_info",
                    },
                )
            )
        else:
            # Direct schema to extractor connection when not using PDF
            connections.append(
                ConnectionDefinition(
                    start="schema",
                    end="extractor",
                    input_config={"schema": "schema"},
                )
            )
        
        # Common connections
        connections.append(
            ConnectionDefinition(
                start="splitter",
                end="chunk_embedder",
                input_config={"text_chunks": "splitter"},
            )
        )
        connections.append(
            ConnectionDefinition(
                start="chunk_embedder",
                end="extractor",
                input_config={"chunks": "chunk_embedder"},
            )
        )
        connections.append(
            ConnectionDefinition(
                start="extractor",
                end="writer",
                input_config={"graph": "extractor"},
            )
        )
        
        # Entity resolution (if enabled)
        if self.config.perform_entity_resolution:
            connections.append(
                ConnectionDefinition(
                    start="writer",
                    end="resolver",
                    input_config={},
                )
            )
        
        # Add connections to pipeline
        for connection in connections:
            self.pipeline.connect(
                connection.start,
                connection.end,
                connection.input_config,
            )
    
    def _get_run_params(self, file_path: Optional[str] = None, text: Optional[str] = None) -> dict[str, Any]:
        """Get the run parameters for the pipeline."""
        run_params = {}
        
        # Add lexical graph config if provided
        if self.config.lexical_graph_config:
            run_params["extractor"] = {
                "lexical_graph_config": self.config.lexical_graph_config
            }
        
        # Validate input parameters
        if not ((text is None) ^ (file_path is None)):
            # Exactly one of text or file_path must be set
            raise PipelineDefinitionError(
                "Use either 'text' (when from_pdf=False) or 'file_path' (when from_pdf=True) argument."
            )
        
        # Set up input based on from_pdf flag
        if self.config.from_pdf:
            if not file_path:
                raise PipelineDefinitionError(
                    "Expected 'file_path' argument when 'from_pdf' is True."
                )
            run_params["pdf_loader"] = {"filepath": file_path}
        else:
            if not text:
                raise PipelineDefinitionError(
                    "Expected 'text' argument when 'from_pdf' is False."
                )
            run_params["splitter"] = {"text": text}  
            # Add full text to schema component for automatic schema extraction
            run_params["schema"] = {"text": text}
            
        return run_params
    
    async def run_async(
        self, file_path: Optional[str] = None, text: Optional[str] = None
    ) -> PipelineResult:
        """
        Asynchronously runs the knowledge graph building process.

        Args:
            file_path (Optional[str]): The path to the PDF file to process. Required if `from_pdf` is True.
            text (Optional[str]): The text content to process. Required if `from_pdf` is False.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        run_params = self._get_run_params(file_path, text)
        return await self.pipeline.run(run_params)


# Example of another hypothetical pipeline type to demonstrate extensibility

# This would be imported from another module in a real implementation
class SomeComponent1Class(Component):
    pass

class SomeComponent2Class(Component):
    pass

@dataclass
class AnotherPipelineConfig(PipelineConfig):
    """Configuration for another hypothetical pipeline type.
    
    All parameters are optional at initialization time, but will be required
    if used by an active component in the pipeline.
    """
    # All parameters are optional at initialization
    some_required_param: Optional[str] = None
    some_optional_param: bool = True


# Create a specialized ComponentSpec for component1 in AnotherPipeline
class CustomComponent1Spec(ComponentSpec):
    """
    A specialized ComponentSpec for component1 that demonstrates another way
    to customize component class selection.
    """
    
    def get_class(self, pipeline: 'TemplatePipeline', component_name: str) -> Type:
        """
        Select the component class based on a configuration parameter.
        
        This is just a demonstration of how to customize component class resolution.
        
        Args:
            pipeline: The pipeline instance
            component_name: The name of the component
            
        Returns:
            The component class to use
        """
        # In a real implementation, this could use any pipeline configuration
        # to determine which class to use
        if pipeline.config.some_optional_param:
            return SomeComponent1Class
        # Use a different class when some_optional_param is False
        return Component  # This is just for demonstration


class AnotherPipeline(TemplatePipeline):
    """Another hypothetical pipeline implementation.
    
    This is included as an example to demonstrate how the TemplateBuilder
    can be used with different pipeline types.
    """
    
    # Pipeline version - increment when making breaking changes
    VERSION = "1.0.0"
    
    COMPONENTS = {
        "component1": CustomComponent1Spec(  # Use a specialized ComponentSpec
            default_class=SomeComponent1Class,
            required_interface=Component,
            params={},
            description="First component of the pipeline - class depends on some_optional_param"
        ),
        "component2": ComponentSpec(
            default_class=SomeComponent2Class,
            required_interface=Component,
            params={
                "param1": "$some_required_param",  # Parameter from config
            },
            description="Second component of the pipeline"
        ),
    }
    
    @classmethod
    def get_config_class(cls) -> Type[PipelineConfig]:
        """Get the configuration class for this pipeline."""
        return AnotherPipelineConfig
    
    def _setup_pipeline(self) -> None:
        """Set up the pipeline with components and connections."""
        # Implementation would create and connect components
        pass


# Usage examples

def usage_examples():
    """
    Examples showing how to use the TemplateBuilder with different pipeline types.
    
    These examples are not meant to be run as-is, but rather to demonstrate
    the usage patterns.
    """
    # Import necessary modules
    from neo4j import GraphDatabase
    from some_module import llm, embedder, custom_splitter
    
    # Initialize required objects
    driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password"))
    
    # Example 1: Create a SimpleKGPipeline with default components
    pipeline1 = (
        TemplateBuilder(SimpleKGPipeline)
        .with_params(
            llm=llm,
            driver=driver,
            embedder=embedder
        )
        .build()
    )
    
    # Example 2: Create a SimpleKGPipeline with custom component configurations
    pipeline2 = (
        TemplateBuilder(SimpleKGPipeline)
        .with_params(
            llm=llm,
            driver=driver,
            embedder=embedder,
            from_pdf=False,  # Process text instead of PDFs
            enforce_schema="STRICT"
        )
        .configure_component("writer", batch_size=500)
        .configure_component("splitter", chunk_size=1000)
        .build()
    )
    
    # Example 3: Create a SimpleKGPipeline with component replacement
    # The custom_splitter must be an instance of TextSplitter to match
    # the required interface defined in SimpleKGPipeline.COMPONENTS["splitter"].required_interface
    pipeline3 = (
        TemplateBuilder(SimpleKGPipeline)
        .with_params(
            llm=llm,
            driver=driver,
            embedder=embedder
        )
        .replace_component("splitter", custom_splitter)  # custom_splitter must be a TextSplitter instance
        .build()
    )
    
    # Example 4: Create another pipeline type
    # This demonstrates how different pipeline types can be built with the same builder
    another_pipeline = (
        TemplateBuilder(AnotherPipeline)
        .with_params(some_required_param="value", some_optional_param=True)
        .configure_component("component1", some_param="value")
        .build()
    )
    # Since some_optional_param=True, the CustomComponent1Spec will select SomeComponent1Class
    
    # Example 5: Replace a component that requires a config parameter
    # The 'some_required_param' is normally required for AnotherPipeline
    # since component2 needs it (as defined in AnotherPipeline.COMPONENTS["component2"].params).
    # However, when we replace component2, we no longer need to provide this parameter because:
    # 1. The parameter is optional at config initialization time (default=None)
    # 2. The check for required parameters happens when each component is created
    # 3. Since component2 is replaced, its parameters are never checked
    another_pipeline_with_replacement = (
        TemplateBuilder(AnotherPipeline)
        .with_params(some_optional_param=False)  # Note: some_required_param is not provided!
        .replace_component("component2", custom_splitter)  # This component doesn't need some_required_param
        .build()
    )
    # Since some_optional_param=False, the CustomComponent1Spec will select Component class
    # This demonstrates dynamic component class selection based on configuration
    