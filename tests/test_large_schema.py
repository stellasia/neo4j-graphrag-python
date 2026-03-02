from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
    Pattern,
)
from examples.extraction_so import graph_schema_to_pydantic_model


def test_very_large_schema_model_generation():
    # Create 100 node types
    node_types = []
    for i in range(100):
        node_types.append(
            NodeType(
                label=f"NodeType_{i}",
                properties=[
                    PropertyType(name="prop_id", type="STRING", required=True),
                    PropertyType(name=f"prop_{i}", type="STRING", required=False),
                ],
            )
        )

    # Create 500 relationship patterns
    patterns = []
    for i in range(100):
        for j in range(1, 6):
            target_idx = (i + j) % 100
            patterns.append(
                Pattern(
                    source=f"NodeType_{i}",
                    relationship="CONNECTED_TO",
                    target=f"NodeType_{target_idx}",
                )
            )

    schema = GraphSchema(
        node_types=tuple(node_types),
        relationship_types=(RelationshipType(label="CONNECTED_TO"),),
        patterns=tuple(patterns),
    )

    # This might be slow or fail if there are internal recursion limits or memory issues
    Model = graph_schema_to_pydantic_model(schema)

    assert "nodes" in Model.model_fields
    assert "relationships" in Model.model_fields

    # In Pydantic V2, NodeUnion is List[Union[...]]
    NodeListAnnotation = Model.model_fields["nodes"].annotation
    # Get the Union from List[Union[...]]
    NodeUnion = NodeListAnnotation.__args__[0]
    node_models = NodeUnion.__args__
    assert len(node_models) == 100

    RelListAnnotation = Model.model_fields["relationships"].annotation
    RelUnion = RelListAnnotation.__args__[0]
    rel_models = RelUnion.__args__
    assert len(rel_models) == 500

    # Print the schema size for information
    import json

    schema_json = json.dumps(Model.model_json_schema())
    print(f"\nSchema size: {len(schema_json)} characters")
    print(f"Number of node models: {len(node_models)}")
    print(f"Number of relationship models: {len(rel_models)}")
