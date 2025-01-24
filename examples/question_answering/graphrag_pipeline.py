import asyncio
import neo4j

from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.components.rag.retrievers import RetrieverWrapper
from neo4j_graphrag.experimental.components.rag.prompt_builder import PromptBuilder
from neo4j_graphrag.experimental.components.rag.generate import Generator
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.llm import OpenAILLM

from neo4j_graphrag.retrievers import VectorRetriever


URI = "neo4j+s://demo.neo4jlabs.com"
AUTH = ("recommendations", "recommendations")
DATABASE = "recommendations"
INDEX_NAME = "moviePlotsEmbedding"


async def main():
    pipeline = Pipeline()
    driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)
    llm = OpenAILLM(model_name="gpt-4o")
    embedder = OpenAIEmbeddings()
    retriever = VectorRetriever(
        driver,
        index_name=INDEX_NAME,
        neo4j_database=DATABASE,
        embedder=embedder,
    )
    pipeline.add_component(RetrieverWrapper(retriever), "retriever")
    pipeline.add_component(PromptBuilder(RagTemplate()), "prompt")
    pipeline.add_component(Generator(llm), "generate")

    pipeline.connect("retriever", "prompt", {
        "context": "retriever.result",
    })
    pipeline.connect("prompt", "generate", {
        "prompt": "prompt.prompt",
    })

    query = "show me a movie with cats"
    res = await pipeline.run({
        "retriever": {"query_text": query},
        "prompt": {"query_text": query, "examples": ""},
    })

    driver.close()
    await llm.async_client.close()
    return res


if __name__ == "__main__":
    print(
        asyncio.run(main())
    )
