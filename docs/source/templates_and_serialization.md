# Template Pipelines and Serialization

This guide explains the template pipeline system and its serialization capabilities.

## Overview

Template pipelines provide a way to define reusable pipeline structures with configurable
components. The serialization framework allows these templates to be saved to and loaded
from JSON files, making it easy to share and reuse pipeline configurations.

## Template Pipelines

Template pipelines define a standard structure for a specific task, with configurable
components and parameters. They simplify the creation of complex pipelines by providing:

- Default components with sensible configurations
- Connection patterns between components
- Parameter resolution with automatic application of global settings
- Clear documentation of available customization options

## Serialization Framework

The serialization framework provides a consistent way to convert pipeline objects to and
from JSON. This enables:

- Saving pipeline templates for later use
- Sharing pipeline configurations between projects
- Versioning pipeline templates
- Creating libraries of reusable pipeline patterns

## Using the Template Builder

The `TemplateBuilder` class provides a fluent interface for creating and customizing template
pipelines.

### Creating a Template Pipeline

```python
from neo4j_graphrag.experimental.pipeline.template_builder import TemplateBuilder
from my_pipelines import TextProcessingPipeline

# Create a template pipeline using the builder
template = (
    TemplateBuilder(TextProcessingPipeline)
    .with_params(
        uppercase=True,
        prefix="<<< ",
        suffix=" >>>"
    )
    .configure_component("processor", debug=True)
    .build()
)

# Run the pipeline
result = await template.run_async(text="Hello, world!")
```

### Saving Templates to JSON

You can save a template to a JSON file for later use:

```python
# Save the template to a JSON file
template.to_json(filepath="text_processing_template.json")

# Or directly from the builder without building first
builder.save_to_json("template.json")
```

### Loading Templates from JSON

Templates can be loaded back from JSON files:

```python
# Load a template directly
loaded_template = TextProcessingPipeline.from_json(
    "text_processing_template.json", 
    filepath=True
)

# Or load into a builder for further customization
loaded_builder = TemplateBuilder.load_from_json(
    TextProcessingPipeline, 
    "text_processing_template.json"
)

# Then customize and build
customized = (
    loaded_builder
    .with_params(uppercase=False)
    .configure_component("formatter", prefix="*** ")
    .build()
)
```

## Creating Custom Templates

To create a custom template pipeline, you need to define:

1. A configuration class that inherits from `PipelineConfig`
2. A template pipeline class that inherits from `TemplatePipeline`
3. Component specifications for each component used in the pipeline

### Example

```python
from neo4j_graphrag.experimental.pipeline.component import Component, DataModel
from neo4j_graphrag.experimental.pipeline.template_builder import (
    PipelineConfig, 
    TemplatePipeline, 
    TemplateBuilder,
    ComponentSpec
)
from typing import Type

# Define a pipeline configuration class
class CustomPipelineConfig(PipelineConfig):
    param1: str = "default"
    param2: bool = False
    
# Define a template pipeline class
class CustomPipeline(TemplatePipeline):
    VERSION = "1.0.0"
    
    # Define components used by this pipeline
    COMPONENTS = {
        "component1": ComponentSpec(
            default_class=Component1Class,
            required_interface=Component,
            params={"some_param": "$param1"},
            description="First component in the pipeline"
        ),
        "component2": ComponentSpec(
            default_class=Component2Class,
            required_interface=Component,
            params={"flag": "$param2"},
            description="Second component in the pipeline"
        )
    }
    
    @classmethod
    def get_config_class(cls) -> Type[PipelineConfig]:
        return CustomPipelineConfig
    
    def _setup_pipeline(self) -> None:
        # Add all components to the pipeline
        for name in self.COMPONENTS:
            component = self.get_component(name)
            self.pipeline.add_component(component, name)
        
        # Set up connections
        self.pipeline.connect(
            "component1", 
            "component2", 
            {"input": "component1.output"}
        )
```

## Complete Example

For a complete example of using template serialization, see the example file:

```python
examples/template_serialization.py
```

This example demonstrates:
1. Creating template pipelines
2. Saving templates to JSON
3. Loading templates from JSON
4. Customizing templates after loading 