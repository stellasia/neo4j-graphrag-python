import asyncio
import json

from neo4j_graphrag.experimental.components.schema import (
    SchemaFromDataImporterModelBulder,
)


async def main():
    with open("./neo4j_importer_model_2025-05-13.json", "r") as f:
        data = json.load(f)

    schema_builder = SchemaFromDataImporterModelBulder()
    schema = await schema_builder.run(data)

    print(schema)


if __name__ == "__main__":
    asyncio.run(main())
