"""This example demonstrates how to use Structured Output with strict schema guidance"""

from __future__ import annotations
import asyncio
from typing import Any, List, Optional, Type, Union, Literal
from pydantic import BaseModel, Field, create_model, ConfigDict
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    RelationshipType,
    PropertyType,
    Pattern,
)
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.experimental.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
)

# Mapping from Neo4j types to Python types
TYPE_MAPPING = {
    "BOOLEAN": bool,
    "DATE": str,  # Simplified for LLM extraction
    "DURATION": str,
    "FLOAT": float,
    "INTEGER": int,
    "LIST": List[Any],
    "LOCAL_DATETIME": str,
    "LOCAL_TIME": str,
    "POINT": str,
    "STRING": str,
    "ZONED_DATETIME": str,
    "ZONED_TIME": str,
}


def get_pydantic_type_from_neo4j_property(
    prop: PropertyType,
) -> tuple[Type[Any], Field]:
    """Helper to convert a PropertyType to a Pydantic field definition."""
    if prop.values:
        python_type = Literal[tuple(prop.values)]
    else:
        python_type = TYPE_MAPPING.get(prop.type, Any)

    if not prop.required:
        python_type = Optional[python_type]

    # OpenAI structured output requires all properties to be in the 'required' list
    # even if they are optional in our domain.
    # In Pydantic, using Ellipsis (...) as the default value makes it required.
    return (python_type, Field(default=..., description=prop.description))


# For OpenAI structured output, all models must have additionalProperties: False
STRICT_CONFIG = ConfigDict(extra="forbid")


def graph_schema_to_pydantic_model(schema: GraphSchema) -> Type[BaseModel]:
    """
    Converts a GraphSchema into a Pydantic model suitable for structured output extraction.

    The resulting model will have:
    - nodes: List[Union[NodeModel1, NodeModel2, ...]]
    - relationships: List[Union[RelModel1, RelModel2, ...]]

    Note: Large schemas with many node types or relationship patterns can result in
    very large JSON schemas, which may exceed LLM provider limits for structured outputs
    (e.g., OpenAI has limits on schema size and complexity).
    For very large schemas, consider:
    1. Breaking the schema into smaller, domain-specific sub-schemas.
    2. Using a two-stage extraction (nodes first, then relationships).
    3. Reducing the number of explicit patterns if they are not strictly necessary.
    """
    node_models = []
    for node_type in schema.node_types:
        # Create a Pydantic model for each node type
        # We include the label as a literal to help the LLM identify the type
        model_name = node_type.label.replace(" ", "_").capitalize()

        fields = {
            "label": (
                Literal[node_type.label],
                Field(default=..., description=f"Node label: {node_type.label}"),
            ),
            "id": (
                str,
                Field(
                    default=...,
                    description="Unique identifier for the node (e.g., name or ID)",
                ),
            ),
        }

        # Add properties from the schema
        for prop in node_type.properties:
            if prop.name == "id":  # Skip if already defined
                continue
            fields[prop.name] = get_pydantic_type_from_neo4j_property(prop)

        node_model = create_model(
            model_name, __base__=BaseModel, __config__=STRICT_CONFIG, **fields
        )
        node_models.append(node_model)

    rel_models = []
    # If patterns are defined, we create models for each pattern
    if schema.patterns:
        for pattern in schema.patterns:
            # Try to find the relationship type in schema.relationship_types
            # to get its properties
            rel_type = schema.relationship_type_from_label(pattern.relationship)

            # Create a specific model for this pattern
            # Model name: SourceLabel_RelLabel_TargetLabel
            model_name = (
                f"{pattern.source}_{pattern.relationship}_{pattern.target}".replace(
                    " ", "_"
                )
                .replace("-", "_")
                .capitalize()
            )

            fields = {
                "label": (
                    Literal[pattern.relationship],
                    Field(
                        default=...,
                        description=f"Relationship label: {pattern.relationship}",
                    ),
                ),
                "source_id": (
                    str,
                    Field(
                        default=...,
                        description=f"Unique identifier of the {pattern.source} node",
                    ),
                ),
                "target_id": (
                    str,
                    Field(
                        default=...,
                        description=f"Unique identifier of the {pattern.target} node",
                    ),
                ),
                "source_label": (
                    Literal[pattern.source],
                    Field(
                        default=..., description=f"Source node label: {pattern.source}"
                    ),
                ),
                "target_label": (
                    Literal[pattern.target],
                    Field(
                        default=..., description=f"Target node label: {pattern.target}"
                    ),
                ),
            }

            # Add properties from the relationship type if it exists
            if rel_type:
                for prop in rel_type.properties:
                    if prop.name in fields:
                        continue
                    fields[prop.name] = get_pydantic_type_from_neo4j_property(prop)

            rel_model = create_model(
                model_name, __base__=BaseModel, __config__=STRICT_CONFIG, **fields
            )
            rel_models.append(rel_model)
    else:
        # Fallback to current behavior if no patterns are defined
        for rel_type in schema.relationship_types:
            # Create a Pydantic model for each relationship type
            model_name = rel_type.label.replace(" ", "_").capitalize()

            fields = {
                "label": (
                    Literal[rel_type.label],
                    Field(
                        default=..., description=f"Relationship label: {rel_type.label}"
                    ),
                ),
                "source_id": (
                    str,
                    Field(
                        default=...,
                        description="Unique identifier or name of the source node",
                    ),
                ),
                "target_id": (
                    str,
                    Field(
                        default=...,
                        description="Unique identifier or name of the target node",
                    ),
                ),
            }

            # Add properties from the schema
            for prop in rel_type.properties:
                fields[prop.name] = get_pydantic_type_from_neo4j_property(prop)

            rel_model = create_model(
                model_name, __base__=BaseModel, __config__=STRICT_CONFIG, **fields
            )
            rel_models.append(rel_model)

    # Define the final extraction model
    extraction_fields = {
        "nodes": (
            List[Union[tuple(node_models)]] if node_models else List[Any],
            Field(default=...),
        ),
        "relationships": (
            List[Union[tuple(rel_models)]] if rel_models else List[Any],
            Field(default=...),
        ),
    }

    Model = create_model(
        "GraphExtraction",
        __base__=BaseModel,
        __config__=STRICT_CONFIG,
        **extraction_fields,
    )

    # Check schema size and warn if large
    import json
    import logging

    schema_json = json.dumps(Model.model_json_schema())
    if len(schema_json) > 50000:  # 50KB as a threshold for warning
        logging.warning(
            f"Generated extraction schema is large ({len(schema_json)} bytes). "
            "This may exceed LLM provider limits for structured outputs."
        )

    return Model


# Example Usage
async def main():
    # Define a more complex schema
    person_node = NodeType(
        label="Person",
        properties=[
            PropertyType(
                name="name",
                type="STRING",
                required=True,
                description="Full name of the person",
            ),
            PropertyType(
                name="role",
                type="STRING",
                required=False,
                description="Job title or role",
            ),
        ],
    )

    org_node = NodeType(
        label="Organization",
        properties=[
            PropertyType(
                name="name",
                type="STRING",
                required=True,
                description="Name of the organization",
            ),
            PropertyType(
                name="headquarters",
                type="STRING",
                required=False,
                description="City where the HQ is located",
            ),
        ],
    )

    project_node = NodeType(
        label="Project",
        properties=[
            PropertyType(
                name="name",
                type="STRING",
                required=True,
                description="Name of the project",
            ),
            PropertyType(
                name="budget",
                type="FLOAT",
                required=False,
                description="Project budget in USD",
            ),
            PropertyType(
                name="status",
                type="STRING",
                required=True,
                values=["Active", "Completed", "On Hold"],
                description="Current status of the project",
            ),
        ],
    )

    works_at_rel = RelationshipType(
        label="WORKS_AT",
        properties=[
            PropertyType(
                name="since",
                type="INTEGER",
                required=False,
                description="The year the person started working there",
            )
        ],
    )

    member_of_rel = RelationshipType(
        label="MEMBER_OF",
        properties=[
            PropertyType(
                name="role",
                type="STRING",
                required=False,
                description="Role within the project",
            )
        ],
    )

    schema = GraphSchema(
        node_types=(person_node, org_node, project_node),
        relationship_types=(works_at_rel, member_of_rel),
        patterns=(
            Pattern(source="Person", relationship="WORKS_AT", target="Organization"),
            Pattern(source="Person", relationship="MEMBER_OF", target="Project"),
        ),
    )

    # Convert to Pydantic model
    ExtractionModel = graph_schema_to_pydantic_model(schema)
    # Print the JSON schema to see what would be sent to the LLM
    # print("Generated JSON Schema for Extraction:")
    # print(json.dumps(ExtractionModel.model_json_schema(), indent=2))

    # Setup the LLM
    # We use OpenAILLM if an API key is provided, otherwise we use MockLLM
    print("Using OpenAILLM for extraction...")
    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    # Mock text chunk
    chunk = TextChunk(
        text=(
            "Alice Johnson has been a Senior Engineer at TechCorp since 2015. "
            "TechCorp is headquartered in San Francisco. "
            "Alice is also a key member of 'Project Alpha', where she serves as the Lead Architect. "
            "The project has an estimated budget of 1.5 million dollars."
        ),
        index=0,
    )

    # Run extraction using the dynamic model as response_format
    print(
        "\nRunning extraction with LLMEntityRelationExtractor and dynamic response_format..."
    )

    # Attach the dynamic model to the schema so the extractor can find it
    schema._pydantic_model = ExtractionModel

    # Instantiate the extractor with structured output enabled
    extractor = LLMEntityRelationExtractor(llm=llm, use_structured_output=True)

    # Run extraction
    # The extractor will now use our dynamic model because it's attached to the schema
    # and map the results back to Neo4jGraph
    extracted_graph = await extractor.run(TextChunks(chunks=[chunk]), schema=schema)

    print("\nExtracted Graph (Neo4jGraph):")
    print(extracted_graph.model_dump_json(indent=2))


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
