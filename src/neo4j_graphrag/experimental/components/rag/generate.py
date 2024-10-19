from typing import Any, Optional

from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.llm import LLMInterface


class GenerationResult(DataModel):
    content: str


class Generate(Component):
    def __init__(self, llm: LLMInterface, return_context: bool = True) -> None:
        self.llm = llm
        self.return_context = return_context

    async def run(self, prompt: str) -> GenerationResult:
        llm_response = await self.llm.ainvoke(prompt)
        return GenerationResult(
            content=llm_response.content,
        )
