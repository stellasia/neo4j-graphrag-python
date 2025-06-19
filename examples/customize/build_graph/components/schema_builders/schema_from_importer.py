import sys
import asyncio
from neo4j_graphrag.experimental.components.schema import SchemaFromImporterExtractor


async def main(file_path):
    extractor = SchemaFromImporterExtractor(
        file_path=file_path,
    )
    schema = await extractor.run()
    print(schema)


if __name__ == "__main__":
    file_path = sys.argv[1]
    asyncio.run(main(file_path))
