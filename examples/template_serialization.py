"""
Example demonstrating the use of serialization with TemplateBuilder.

This file shows how templates can be:
1. Created using TemplateBuilder
2. Saved to JSON files for later use
3. Loaded back from JSON files
4. Customized after loading
"""

import asyncio
import json
import os
from typing import Any, Dict, Optional, Type

import neo4j
from neo4j import GraphDatabase

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
class TextOutput(DataModel):
    text: str

class TextProcessor(Component):
    """A simple text processing component."""
    
    def __init__(self, uppercase: bool = False):
        self.uppercase = uppercase
        
    async def run(self, text: str) -> TextOutput:
        """Process text by converting to uppercase if configured."""
        if self.uppercase:
            result = text.upper()
        else:
            result = text
        return TextOutput(text=result)

class TextFormatter(Component):
    """A component that formats text."""
    
    def __init__(self, prefix: str = "", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix
        
    async def run(self, text: str) -> TextOutput:
        """Add prefix and suffix to text."""
        formatted = f"{self.prefix}{text}{self.suffix}"
        return TextOutput(text=formatted)

class CounterComponent(Component):
    """A component that counts characters in text."""
    
    async def run(self, text: str) -> TextOutput:
        """Count characters in text."""
        count = len(text)
        return TextOutput(text=f"Character count: {count}")


# Define a pipeline configuration class
class TextProcessingConfig(PipelineConfig):
    """Configuration for text processing pipeline."""
    
    debug_mode: bool = False
    uppercase: bool = False
    prefix: str = ""
    suffix: str = ""


# Define a template pipeline class
class TextProcessingPipeline(TemplatePipeline):
    """Template pipeline for text processing."""
    
    VERSION = "1.0.0"
    
    # Define components used by this pipeline
    COMPONENTS = {
        "processor": ComponentSpec(
            default_class=TextProcessor,
            required_interface=Component,
            params={"uppercase": "$uppercase"},
            description="Processes input text"
        ),
        "formatter": ComponentSpec(
            default_class=TextFormatter,
            required_interface=Component,
            params={"prefix": "$prefix", "suffix": "$suffix"},
            description="Formats text with prefix and suffix"
        ),
        "counter": ComponentSpec(
            default_class=CounterComponent,
            required_interface=Component,
            params={},
            description="Counts characters in text"
        )
    }
    
    @classmethod
    def get_config_class(cls) -> Type[PipelineConfig]:
        """Get the configuration class for this pipeline."""
        return TextProcessingConfig
    
    def _setup_pipeline(self) -> None:
        """Set up the pipeline with components and connections."""
        # Add all components to the pipeline
        for name in self.COMPONENTS:
            component = self.get_component(name)
            self.pipeline.add_component(component, name)
        
        # Set up connections
        self.pipeline.connect(
            "processor", 
            "formatter", 
            {"text": "processor.text"}
        )
        
        self.pipeline.connect(
            "formatter", 
            "counter", 
            {"text": "formatter.text"}
        )


async def main():
    """Demonstrate the use of serialization with TemplateBuilder."""
    
    # Create a template pipeline using the builder
    print("Creating a template pipeline...")
    template = (
        TemplateBuilder(TextProcessingPipeline)
        .with_params(
            uppercase=True,
            prefix="<<< ",
            suffix=" >>>"
        )
        .build()
    )
    
    # Save the template to a JSON file
    print("Saving template to JSON...")
    template_file = "text_processing_template.json"
    template.to_json(filepath=template_file)
    print(f"Template saved to {template_file}")
    
    # Print the template JSON for inspection
    with open(template_file, "r") as f:
        template_json = json.load(f)
    print("\nTemplate JSON (partial):")
    print(json.dumps(template_json, indent=2)[:500] + "...\n")
    
    # Load the template back from JSON
    print("Loading template from JSON...")
    loaded_template = TextProcessingPipeline.from_json(template_file, filepath=True)
    
    # Modify some components after loading
    print("Customizing loaded template...")
    builder = (
        TemplateBuilder(TextProcessingPipeline)
        .with_params(
            uppercase=False,  # Override uppercase setting
            prefix="*** ",   # Different prefix
            suffix=" ***"    # Different suffix
        )
    )
    
    # Build the customized pipeline
    customized_template = builder.build()
    
    # Save the customized template
    customized_file = "customized_template.json"
    customized_template.to_json(filepath=customized_file)
    print(f"Customized template saved to {customized_file}")
    
    # You can also save directly from the builder without building first
    print("Saving template directly from builder...")
    builder_file = "builder_template.json"
    builder.save_to_json(builder_file)
    print(f"Builder template saved to {builder_file}")
    
    # Load a template directly into the builder
    print("Loading template directly into builder...")
    loaded_from_builder = TemplateBuilder.load_from_json(
        TextProcessingPipeline, 
        template_file
    )
    
    # Clean up example files
    for file in [template_file, customized_file, builder_file]:
        if os.path.exists(file):
            os.remove(file)
    
    print("\nExample files cleaned up. Serialization demonstration complete.")


if __name__ == "__main__":
    asyncio.run(main()) 