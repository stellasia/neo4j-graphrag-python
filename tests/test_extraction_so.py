from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel
from neo4j_graphrag.experimental.components.schema import (
    GraphSchema,
    NodeType,
    RelationshipType,
    PropertyType,
    Pattern,
)
from examples.extraction_so import graph_schema_to_pydantic_model


def test_node_type_generation():
    person_node = NodeType(
        label="Person",
        properties=[
            PropertyType(name="name", type="STRING", required=True),
            PropertyType(name="age", type="INTEGER", required=False),
        ],
    )
    schema = GraphSchema(node_types=(person_node,))

    Model = graph_schema_to_pydantic_model(schema)

    # Check general structure
    assert issubclass(Model, BaseModel)
    assert "nodes" in Model.model_fields
    assert "relationships" in Model.model_fields

    # Check Node model
    NodeUnion = Model.model_fields["nodes"].annotation
    # Extract the Union members (List[Union[...]])
    # For a single node, it might be just List[Person] or List[Union[Person]]
    # Pydantic's internal representation varies, but we can check if Person is in there
    node_models = NodeUnion.__args__ if hasattr(NodeUnion, "__args__") else [NodeUnion]
    PersonModel = node_models[0]

    assert PersonModel.__name__ == "Person"
    assert "label" in PersonModel.model_fields
    assert "id" in PersonModel.model_fields
    assert "name" in PersonModel.model_fields
    assert "age" in PersonModel.model_fields

    # Check constraints
    assert PersonModel.model_fields["label"].annotation == Literal["Person"]
    assert PersonModel.model_config.get("extra") == "forbid"


def test_property_values_enum():
    project_node = NodeType(
        label="Project",
        properties=[
            PropertyType(
                name="status",
                type="STRING",
                required=True,
                values=["Active", "Completed"],
            ),
        ],
    )
    schema = GraphSchema(node_types=(project_node,))
    Model = graph_schema_to_pydantic_model(schema)

    ProjectModel = Model.model_fields["nodes"].annotation.__args__[0]

    # The type should be Literal["Active", "Completed"]
    # (Note: pydantic might wrap it or flatten it depending on version)
    status_field = ProjectModel.model_fields["status"].annotation
    assert status_field == Literal["Active", "Completed"]


def test_relationship_patterns():
    person_node = NodeType(
        label="Person", properties=[PropertyType(name="name", type="STRING")]
    )
    org_node = NodeType(
        label="Organization", properties=[PropertyType(name="name", type="STRING")]
    )
    works_at_rel = RelationshipType(
        label="WORKS_AT", properties=[PropertyType(name="since", type="INTEGER")]
    )
    pattern = Pattern(source="Person", relationship="WORKS_AT", target="Organization")

    schema = GraphSchema(
        node_types=(person_node, org_node),
        relationship_types=(works_at_rel,),
        patterns=(pattern,),
    )

    Model = graph_schema_to_pydantic_model(schema)

    RelUnion = Model.model_fields["relationships"].annotation
    rel_models = RelUnion.__args__ if hasattr(RelUnion, "__args__") else [RelUnion]
    RelModel = rel_models[0]

    # Pattern name logic: Person_WORKS_AT_Organization
    assert RelModel.__name__ == "Person_works_at_organization"
    assert "source_id" in RelModel.model_fields
    assert "target_id" in RelModel.model_fields
    assert "source_label" in RelModel.model_fields
    assert "target_label" in RelModel.model_fields
    assert "since" in RelModel.model_fields

    assert RelModel.model_fields["source_label"].annotation == Literal["Person"]
    assert RelModel.model_fields["target_label"].annotation == Literal["Organization"]


def test_openai_strict_requirements():
    schema = GraphSchema(
        node_types=(
            NodeType(label="X", properties=[PropertyType(name="p", type="STRING")]),
        )
    )
    Model = graph_schema_to_pydantic_model(schema)

    json_schema = Model.model_json_schema()

    # Check top level
    assert json_schema.get("additionalProperties") is False
    assert "nodes" in json_schema.get("required", [])
    assert "relationships" in json_schema.get("required", [])

    # Check node level (usually in $defs)
    node_schema = list(json_schema.get("$defs", {}).values())[0]
    assert node_schema.get("additionalProperties") is False
    # All properties must be in 'required' for OpenAI structured output
    props = node_schema.get("properties", {}).keys()
    required = node_schema.get("required", [])
    for p in props:
        assert p in required


def test_diverse_property_types():
    # Test mapping of various Neo4j types to Python types
    node = NodeType(
        label="Diverse",
        properties=[
            PropertyType(name="b", type="BOOLEAN", required=True),
            PropertyType(name="i", type="INTEGER", required=True),
            PropertyType(name="f", type="FLOAT", required=True),
            PropertyType(name="s", type="STRING", required=True),
            PropertyType(name="l", type="LIST", required=True),
            PropertyType(name="d", type="DATE", required=False),
        ],
    )
    schema = GraphSchema(node_types=(node,))
    Model = graph_schema_to_pydantic_model(schema)
    DiverseModel = Model.model_fields["nodes"].annotation.__args__[0]

    assert DiverseModel.model_fields["b"].annotation == bool
    assert DiverseModel.model_fields["i"].annotation == int
    assert DiverseModel.model_fields["f"].annotation == float
    assert DiverseModel.model_fields["s"].annotation == str
    assert DiverseModel.model_fields["l"].annotation == List[Any]
    # Optional[str] for DATE (simplified mapping in example)
    assert DiverseModel.model_fields["d"].annotation == Optional[str]


def test_relationship_without_patterns():
    # Fallback behavior when no patterns are defined
    rel_type = RelationshipType(
        label="KNOWS", properties=[PropertyType(name="strength", type="INTEGER")]
    )
    schema = GraphSchema(
        node_types=(
            NodeType(
                label="Person", properties=[PropertyType(name="name", type="STRING")]
            ),
        ),
        relationship_types=(rel_type,),
    )

    Model = graph_schema_to_pydantic_model(schema)
    RelUnion = Model.model_fields["relationships"].annotation
    rel_models = RelUnion.__args__ if hasattr(RelUnion, "__args__") else [RelUnion]
    KnowsModel = rel_models[0]

    assert KnowsModel.__name__ == "Knows"
    assert "source_id" in KnowsModel.model_fields
    assert "target_id" in KnowsModel.model_fields
    assert "strength" in KnowsModel.model_fields


def test_multiple_node_types():
    schema = GraphSchema(
        node_types=(
            NodeType(label="A", properties=[PropertyType(name="a", type="STRING")]),
            NodeType(label="B", properties=[PropertyType(name="b", type="STRING")]),
        )
    )
    Model = graph_schema_to_pydantic_model(schema)
    NodeUnion = Model.model_fields["nodes"].annotation

    # NodeUnion is List[Union[A, B]]
    # Get the Union[A, B] part
    UnionPart = NodeUnion.__args__[0]
    # Get the members of the Union
    node_models = UnionPart.__args__

    assert len(node_models) == 2
    names = {m.__name__ for m in node_models}
    assert names == {"A", "B"}


def test_empty_schema():
    schema = GraphSchema(node_types=())
    Model = graph_schema_to_pydantic_model(schema)

    assert "nodes" in Model.model_fields
    assert "relationships" in Model.model_fields
    # Should default to List[Any] when empty
    assert Model.model_fields["nodes"].annotation == List[Any]
    assert Model.model_fields["relationships"].annotation == List[Any]
