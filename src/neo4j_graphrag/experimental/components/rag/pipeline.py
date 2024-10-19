from typing import Any, Optional

from neo4j_graphrag.experimental.components.rag.generate import Generate
from neo4j_graphrag.experimental.components.rag.prompt_builder import PromptBuilder
from neo4j_graphrag.experimental.components.rag.retrievers import RetrieverWrapper
from neo4j_graphrag.experimental.pipeline import InMemoryStore, Pipeline
from neo4j_graphrag.generation import PromptTemplate, RagTemplate
from neo4j_graphrag.generation.types import RagResultModel
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RetrieverResult


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: RagTemplate = RagTemplate(),
    ):
        self.store = InMemoryStore()
        self.pipeline = self._build_pipeline(retriever, llm, prompt_template)

    def _build_pipeline(
        self, retriever: Retriever, llm: LLMInterface, prompt_template: PromptTemplate
    ) -> Pipeline:
        pipeline = Pipeline(self.store)
        pipeline.add_component(RetrieverWrapper(retriever), "retriever")
        pipeline.add_component(PromptBuilder(prompt_template), "augmentation")
        pipeline.add_component(Generate(llm), "generate")
        pipeline.connect("retriever", "augmentation", {"context": "retriever.result"})
        pipeline.connect("augmentation", "generate", {"prompt": "augmentation.prompt"})
        return pipeline

    def _prepare_inputs(
        self, query_text: str, examples: str, retriever_config: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "retriever": {"query_text": query_text, **retriever_config},
            "augmentation": {
                "query_text": query_text,
                "examples": examples,
            },
        }

    async def run(
        self,
        query_text: str = "",
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: bool = False,
    ) -> RagResultModel:
        retriever_config = retriever_config or {}
        pipe_inputs = self._prepare_inputs(query_text, examples, retriever_config)
        pipeline_result = await self.pipeline.run(pipe_inputs)
        result: dict[str, Any] = {
            "answer": pipeline_result.result["generate"]["content"]
        }
        if return_context:
            context = await self.store.get_result_for_component(
                pipeline_result.run_id, "retriever"
            )
            retriever_result = context["result"]
            result["retriever_result"] = RetrieverResult(**retriever_result)
        return RagResultModel(**result)


if __name__ == "__main__":
    import asyncio

    import neo4j

    from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
    from neo4j_graphrag.llm import OpenAILLM
    from neo4j_graphrag.retrievers import VectorRetriever

    URI = "neo4j+s://demo.neo4jlabs.com"
    AUTH = ("recommendations", "recommendations")
    DATABASE = "recommendations"
    INDEX = "moviePlotsEmbedding"

    driver = neo4j.GraphDatabase.driver(
        URI,
        auth=AUTH,
        database=DATABASE,
    )

    embedder = OpenAIEmbeddings()

    retriever = VectorRetriever(
        driver,
        index_name=INDEX,
        embedder=embedder,
    )

    llm = OpenAILLM(model_name="gpt-4o", model_params={"temperature": 0})

    rag = RagPipeline(
        retriever=retriever,
        llm=llm,
    )

    result = asyncio.run(
        rag.run(
            query_text="Tell me more about Avatar movies",
            return_context=True,
        )
    )
    print(result)

    driver.close()
