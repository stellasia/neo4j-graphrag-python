from typing import Any

from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.generation import PromptTemplate


class PromptData(DataModel):
    inputs: dict[str, Any]


class PromptResult(DataModel):
    prompt: str


class PromptBuilder(Component):
    def __init__(self, template: PromptTemplate):
        self.template = template

    async def run(self, **kwargs: Any) -> PromptResult:
        return PromptResult(prompt=self.template.format(**kwargs))
