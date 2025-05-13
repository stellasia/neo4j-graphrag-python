import asyncio

import neo4j

from neo4j_graphrag.experimental.components.schema import SchemaFromGraphBuilder


# Define database credentials
URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX = "moviePlotsEmbedding"


async def main() -> None:
    driver = neo4j.GraphDatabase.driver(
        URI,
        auth=AUTH,
    )
    schema_builder = SchemaFromGraphBuilder(driver)
    res = await schema_builder.run()
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
