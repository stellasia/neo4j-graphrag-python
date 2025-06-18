import asyncio

import neo4j
from neo4j_graphrag.experimental.components.kg_writer import (
    KGWriterModel,
    Neo4jWriter,
)
from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode


async def run_writer(driver: neo4j.Driver, graph: Neo4jGraph) -> KGWriterModel:
    writer = Neo4jWriter(
        driver,
        # optionally, configure the neo4j database
        # neo4j_database="neo4j",
        # you can tune batch_size to improve speed
        # batch_size=1000,
    )
    result = await writer.run(graph=graph)
    return result


async def main():
    graph = Neo4jGraph(
        nodes=[Neo4jNode(id="1", label="Label", properties={"name": "test"})]
    )
    with neo4j.GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password"),
    ) as driver:
        await run_writer(driver=driver, graph=graph)


if __name__ == "__main__":
    asyncio.run(main())
