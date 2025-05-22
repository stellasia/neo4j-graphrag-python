"""
Example demonstrating parameter references with the TemplateBuilder.

This example shows how to:
1. Define global parameters that apply to multiple components
2. Use parameter references in component parameter values
3. Save and load templates with parameter references
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional, Type

from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline
from neo4j_graphrag.experimental.pipeline.serializable import Serializable
from neo4j_graphrag.experimental.pipeline.template_builder import (
    PipelineConfig, 
    TemplatePipeline, 
    TemplateBuilder,
    ComponentSpec
)
from pydantic import BaseModel


# Define some simple components for the example
class ConfigOutput(DataModel):
    config: Dict[str, Any]

class DatabaseComponent(Component):
    """A component that simulates database access."""
    
    def __init__(self, connection_string: str, timeout: int = 30, debug: bool = False):
        self.connection_string = connection_string
        self.timeout = timeout
        self.debug = debug
        
    async def run(self) -> ConfigOutput:
        """Return component configuration."""
        return ConfigOutput(config={
            "type": "database",
            "connection_string": self.connection_string,
            "timeout": self.timeout,
            "debug": self.debug
        })

class ApiComponent(Component):
    """A component that simulates API access."""
    
    def __init__(self, api_key: str, base_url: str, timeout: int = 30, debug: bool = False):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.debug = debug
        
    async def run(self) -> ConfigOutput:
        """Return component configuration."""
        return ConfigOutput(config={
            "type": "api",
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "debug": self.debug
        })

class LoggerComponent(Component):
    """A component that simulates logging."""
    
    def __init__(self, log_level: str = "INFO", debug: bool = False):
        self.log_level = log_level
        self.debug = debug
        
    async def run(self) -> ConfigOutput:
        """Return component configuration."""
        return ConfigOutput(config={
            "type": "logger",
            "log_level": self.log_level,
            "debug": self.debug
        })


# Define a pipeline configuration class with global parameters
class ParamReferencesConfig(PipelineConfig):
    """Configuration with global parameters."""
    
    # Global connection parameters
    connection_string: str = "postgres://localhost:5432/mydb"
    api_key: str = "default-api-key"
    base_url: str = "https://api.example.com"
    
    # Global settings that apply to multiple components
    timeout: int = 30
    debug: bool = False
    log_level: str = "INFO"


# Define a template pipeline class that uses parameter references
class ParamReferencesPipeline(TemplatePipeline):
    """Template pipeline demonstrating parameter references."""
    
    VERSION = "1.0.0"
    
    # Define components with parameter references
    COMPONENTS = {
        "database": ComponentSpec(
            default_class=DatabaseComponent,
            required_interface=Component,
            params={
                "connection_string": "$connection_string",  # Reference to config parameter
                "timeout": "$timeout",                      # Shared parameter
                "debug": "$debug"                          # Shared parameter
            },
            description="Database access component"
        ),
        "api": ComponentSpec(
            default_class=ApiComponent,
            required_interface=Component,
            params={
                "api_key": "$api_key",                     # Reference to config parameter
                "base_url": "$base_url",                   # Reference to config parameter
                "timeout": "$timeout",                     # Shared parameter
                "debug": "$debug"                         # Shared parameter
            },
            description="API access component"
        ),
        "logger": ComponentSpec(
            default_class=LoggerComponent,
            required_interface=Component,
            params={
                "log_level": "$log_level",                 # Reference to config parameter
                "debug": "$debug"                         # Shared parameter
            },
            description="Logging component"
        )
    }
    
    @classmethod
    def get_config_class(cls) -> Type[PipelineConfig]:
        """Get the configuration class for this pipeline."""
        return ParamReferencesConfig
    
    def _setup_pipeline(self) -> None:
        """Set up the pipeline with components and connections."""
        # Add all components to the pipeline
        for name in self.COMPONENTS:
            component = self.get_component(name)
            self.pipeline.add_component(component, name)
        
        # No connections needed for this example


async def main():
    """Demonstrate parameter references with TemplateBuilder."""
    
    print("\n=== Parameter References Example ===\n")
    
    # Create a pipeline with custom global parameters
    print("Creating pipeline with custom global parameters...")
    pipeline = (
        TemplateBuilder(ParamReferencesPipeline)
        .with_params(
            # Override global connection parameters
            connection_string="mysql://localhost:3306/customdb",
            api_key="custom-api-key",
            
            # Override shared parameters
            timeout=60,
            debug=True
        )
        .build()
    )
    
    # Let's check the component parameters
    print("\nComponent parameter values:")
    for name in ["database", "api", "logger"]:
        component = pipeline.get_component(name)
        print(f"\n{name.upper()} COMPONENT:")
        for key, value in component.__dict__.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
    
    # Save the pipeline to a JSON file
    print("\nSaving pipeline to JSON...")
    pipeline_file = "param_references_pipeline.json"
    pipeline.to_json(filepath=pipeline_file)
    print(f"Pipeline saved to {pipeline_file}")
    
    # Load the pipeline back from JSON
    print("\nLoading pipeline from JSON...")
    loaded_pipeline = ParamReferencesPipeline.from_json(pipeline_file, filepath=True)
    
    # Modify specific component parameters after loading
    print("\nCustomizing specific component parameters...")
    customized_pipeline = (
        TemplateBuilder(ParamReferencesPipeline)
        .with_params(
            # Use the same global parameters as the original
            connection_string="mysql://localhost:3306/customdb",
            api_key="custom-api-key",
            timeout=60,
            debug=True
        )
        # Override specific component parameters
        .configure_component("database", timeout=120)
        .configure_component("logger", log_level="DEBUG")
        .build()
    )
    
    # Let's check the component parameters again
    print("\nCustomized component parameter values:")
    for name in ["database", "api", "logger"]:
        component = customized_pipeline.get_component(name)
        print(f"\n{name.upper()} COMPONENT:")
        for key, value in component.__dict__.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
    
    # Clean up example files
    os.remove(pipeline_file)
    print(f"\nExample file {pipeline_file} cleaned up.")
    print("\nParameter references demonstration complete.")


if __name__ == "__main__":
    asyncio.run(main()) 