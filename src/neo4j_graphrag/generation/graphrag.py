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
from __future__ import annotations

import logging
import warnings
from typing import Any, List, Optional, Union

from pydantic import ValidationError

from neo4j_graphrag.exceptions import (
    RagInitializationError,
    SearchValidationError,
)
from neo4j_graphrag.generation.prompts import (
    RagTemplate,
    QuestionRewriteWithHistoryTemplate,
)
from neo4j_graphrag.generation.types import RagInitModel, RagResultModel, RagSearchModel
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.types import LLMMessage
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.retrievers.base import Retriever
from neo4j_graphrag.types import RetrieverResult

logger = logging.getLogger(__name__)


class GraphRAG:
    """Performs a GraphRAG search using a specific retriever
    and LLM.

    Example:

    .. code-block:: python

      import neo4j
      from neo4j_graphrag.retrievers import VectorRetriever
      from neo4j_graphrag.llm.openai_llm import OpenAILLM
      from neo4j_graphrag.generation import GraphRAG

      driver = neo4j.GraphDatabase.driver(URI, auth=AUTH)

      retriever = VectorRetriever(driver, "vector-index-name", custom_embedder)
      llm = OpenAILLM()
      graph_rag = GraphRAG(retriever, llm)
      graph_rag.search(query_text="Find me a book about Fremen")

    Args:
        retriever (Retriever): The retriever used to find relevant context to pass to the LLM.
        llm (LLMInterface): The LLM used to generate the answer.
        prompt_template (RagTemplate): The prompt template that will be formatted with context and user question and passed to the LLM.

    Raises:
        RagInitializationError: If validation of the input arguments fail.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMInterface,
        prompt_template: RagTemplate = RagTemplate(),
    ):
        try:
            validated_data = RagInitModel(
                retriever=retriever,
                llm=llm,
                prompt_template=prompt_template,
            )
        except ValidationError as e:
            raise RagInitializationError(e.errors())
        self.retriever = validated_data.retriever
        self.llm = validated_data.llm
        self.prompt_template = validated_data.prompt_template

    def search(
        self,
        query_text: str = "",
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        examples: str = "",
        retriever_config: Optional[dict[str, Any]] = None,
        return_context: bool | None = None,
    ) -> RagResultModel:
        """
        .. warning::
            The default value of 'return_context' will change from 'False' to 'True' in a future version.


        This method performs a full RAG search:
            1. Retrieval: context retrieval
            2. Augmentation: prompt formatting
            3. Generation: answer generation with LLM


        Args:
            query_text (str): The user question.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            examples (str): Examples added to the LLM prompt.
            retriever_config (Optional[dict]): Parameters passed to the retriever.
                search method; e.g.: top_k
            return_context (bool): Whether to append the retriever result to the final result (default: False).

        Returns:
            RagResultModel: The LLM-generated answer.

        """
        if return_context is None:
            warnings.warn(
                "The default value of 'return_context' will change from 'False' to 'True' in a future version.",
                DeprecationWarning,
            )
            return_context = False
        try:
            validated_data = RagSearchModel(
                query_text=query_text,
                examples=examples,
                retriever_config=retriever_config or {},
                return_context=return_context,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors())
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        query_for_retrieval = self._build_query_for_retrieval(
            validated_data.query_text, message_history
        )
        logger.debug(f"RAG: {query_for_retrieval=}")
        retriever_result: RetrieverResult = self.retriever.search(
            query_text=query_for_retrieval, **validated_data.retriever_config
        )
        context = "\n".join(item.content for item in retriever_result.items)
        answer_generation_prompt = self.prompt_template.format(
            query_text=query_text, context=context, examples=validated_data.examples
        )
        logger.debug(f"RAG: retriever_result={retriever_result}")
        logger.debug(f"RAG: prompt={answer_generation_prompt}")
        logger.debug(f"RAG: message_history={message_history}")
        answer = self.llm.invoke(
            answer_generation_prompt,
            message_history,
            system_instruction=self.prompt_template.system_instructions,
        )
        result: dict[str, Any] = {"answer": answer.content}
        if return_context:
            result["retriever_result"] = retriever_result
        return RagResultModel(**result)

    def _build_query_for_retrieval(
        self,
        query_text: str,
        message_history: Optional[List[LLMMessage]] = None,
    ) -> str:
        if not message_history:
            return query_text
        question_rewrite_template = QuestionRewriteWithHistoryTemplate()
        question_rewrite_prompt = question_rewrite_template.format(
            context=str(message_history),
            question=query_text,
        )
        new_question = self.llm.invoke(
            input=question_rewrite_prompt,
            system_instruction=question_rewrite_template.system_instructions,
        ).content
        return new_question

    def _format_message_history(
        self, query_text: str, message_history: List[LLMMessage]
    ) -> str:
        history = list(message_history)  # make a copy of the list
        history.append({"role": "user", "content": query_text})
        message_list = [
            f"{message['role']}: {message['content']}" for message in history
        ]
        history_str = "\n".join(message_list)
        return history_str
