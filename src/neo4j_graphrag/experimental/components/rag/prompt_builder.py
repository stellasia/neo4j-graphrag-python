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
from typing import Any

from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.generation import PromptTemplate


# class PromptData(DataModel):
#     inputs: dict[str, Any]


class PromptResult(DataModel):
    prompt: str


class PromptBuilder(Component):
    def __init__(self, template: PromptTemplate):
        self.template = template

    async def run(self, **kwargs: Any) -> PromptResult:
        return PromptResult(prompt=self.template.format(**kwargs))
