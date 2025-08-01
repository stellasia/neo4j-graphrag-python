import asyncio

from neo4j_graphrag.experimental.components.kg_writer import (
    KGWriterModel,
    CsVWriter,
)
from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType
from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, \
    Neo4jRelationship


async def main(graph: Neo4jGraph, schema: GraphSchema) -> KGWriterModel:
    writer = CsVWriter(
        output_folder="output",
    )
    result = await writer.run(
        graph=graph,
        schema=schema,
        overwrite_output=True,
    )
    return result


if __name__ == "__main__":
    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(
                id="p-0",
                label="Person",
                properties={
                    "name": "Alice",
                    "eyeColor": "blue",
                }
            ),
            Neo4jNode(
                id="p-1",
                label="Person",
                properties={
                    "name": "Robert",
                    "eyeColor": "brown",
                    "nickName": "Bob",
                }
            ),
            Neo4jNode(
                id="l-0",
                label="Location",
                properties={
                    "name": "Wonderland",
                }
            )
        ],
        relationships=[
            Neo4jRelationship(
                type="KNOWS",
                start_node_id="p-0",
                end_node_id="p-1",
                properties={
                    "reason": "Cryptography"
                }
            )
        ]
    )

    schema = GraphSchema.model_validate({
        "node_types": [
            "Location",
            {
                "label": "Person",
                "properties": [
                    {
                        "name": "name",
                        "type": "STRING",
                        "required": True,
                    }
                ],
                "additional_properties": True,
            }
        ],
        "relationship_types": [
            "KNOWS",
        ]
    })

    res = asyncio.run(main(graph=graph, schema=schema))
    print(res)
