from typing import Any

from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RetrieverResult


class RetrieverWrapperResult(DataModel):
    result: RetrieverResult


class RetrieverWrapper(Component):
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    async def run(self, **kwargs: Any) -> RetrieverWrapperResult:
        return RetrieverWrapperResult(
            result=self.retriever.search(**kwargs),
        )
