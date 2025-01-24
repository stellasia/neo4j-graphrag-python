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
from typing import Any, ClassVar, Literal, Optional, Union, cast

from pydantic import ConfigDict, RootModel

from neo4j_graphrag.experimental.components.rag.generate import Generator
from neo4j_graphrag.experimental.components.rag.prompt_builder import PromptBuilder
from neo4j_graphrag.experimental.components.rag.retrievers import RetrieverWrapper
from neo4j_graphrag.experimental.pipeline.config.object_config import ObjectConfig
from neo4j_graphrag.experimental.pipeline.config.template_pipeline.base import (
    TemplatePipelineConfig,
)
from neo4j_graphrag.experimental.pipeline.config.types import PipelineType
from neo4j_graphrag.experimental.pipeline.types import ConnectionDefinition
from neo4j_graphrag.generation import RagTemplate
from neo4j_graphrag.retrievers.base import Retriever


class RetrieverConfig(ObjectConfig[RetrieverWrapper]):
    INTERFACE = Retriever
    # the result of _get_class is a Retriever
    # it is translated into a RetrieverWrapper (which is a Component)
    # in the 'parse' method below
    DEFAULT_MODULE = "neo4j_graphrag.retrievers"
    REQUIRED_PARAMS = ["driver"]

    def parse(self, resolved_data: Optional[dict[str, Any]] = None) -> RetrieverWrapper:
        retriever = cast(Retriever, super().parse(resolved_data))
        return RetrieverWrapper(retriever)


class RetrieverType(RootModel):  # type: ignore
    root: Union[RetrieverWrapper, RetrieverConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def parse(self, resolved_data: dict[str, Any] | None = None) -> RetrieverWrapper:
        if isinstance(self.root, RetrieverWrapper):
            return self.root
        return self.root.parse(resolved_data)


class SimpleRAGPipelineConfig(TemplatePipelineConfig):
    COMPONENTS: ClassVar[list[str]] = [
        "retriever",
        "prompt_builder",
        "generator",
    ]
    retriever: RetrieverType
    prompt_template: Union[RagTemplate, str] = RagTemplate()
    template_: Literal[PipelineType.SIMPLE_RAG_PIPELINE] = (
        PipelineType.SIMPLE_RAG_PIPELINE
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_retriever(self) -> RetrieverWrapper:
        retriever = self.retriever.parse(self._global_data)
        return retriever

    def _get_prompt_builder(self) -> PromptBuilder:
        if isinstance(self.prompt_template, str):
            return PromptBuilder(RagTemplate(template=self.prompt_template))
        return PromptBuilder(self.prompt_template)

    def _get_generator(self) -> Generator:
        llm = self.get_default_llm()
        return Generator(llm)

    def _get_connections(self) -> list[ConnectionDefinition]:
        connections = [
            ConnectionDefinition(
                start="retriever",
                end="prompt_builder",
                input_config={"context": "retriever.result"},
            ),
            ConnectionDefinition(
                start="prompt_builder",
                end="generator",
                input_config={
                    "prompt": "prompt_builder.prompt",
                },
            ),
        ]
        return connections

    def get_run_params(self, user_input: dict[str, Any]) -> dict[str, Any]:
        # query_text: str = "",
        # examples: str = "",
        # retriever_config: Optional[dict[str, Any]] = None,
        # return_context: bool | None = None,
        run_params = {
            "retriever": {
                "query_text": user_input["query_text"],
                **user_input.get("retriever_config", {}),
            },
            "prompt_builder": {
                "query_text": user_input["query_text"],
                "examples": user_input.get("examples", ""),
            },
            "generator": {},
        }
        return run_params
