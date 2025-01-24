#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.llm import LLMInterface


class GenerationResult(DataModel):
    content: str


class Generator(Component):
    def __init__(self, llm: LLMInterface) -> None:
        self.llm = llm

    async def run(self, prompt: str) -> GenerationResult:
        llm_response = await self.llm.ainvoke(prompt)
        return GenerationResult(
            content=llm_response.content,
        )
