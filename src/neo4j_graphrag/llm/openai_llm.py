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

import abc
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Iterable,
    Sequence,
    Union,
    cast,
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from pydantic import ValidationError

from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

from ..exceptions import LLMGenerationError
from .base import LLMInterface
from .types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    ToolCall,
    ToolCallResponse,
    SystemMessage,
    UserMessage,
)

from neo4j_graphrag.tool import Tool
from .rate_limiter import BaseRateLimiter, RetryConfig

if TYPE_CHECKING:
    import openai

# Try to import tiktoken for precise token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class BaseOpenAILLM(LLMInterface, abc.ABC):
    client: openai.OpenAI
    async_client: openai.AsyncOpenAI

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limiter: Optional[BaseRateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Base class for OpenAI LLM.

        Makes sure the openai Python client is installed during init.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            rate_limiter (Optional[BaseRateLimiter]): Rate limiter to control request frequency. Defaults to None.
            retry_config (Optional[RetryConfig]): Configuration for retry behavior on rate limit errors. Defaults to None.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                """Could not import openai Python client.
                Please install it with `pip install "neo4j-graphrag[openai]"`."""
            )
        self.openai = openai
        super().__init__(model_name, model_params, rate_limiter, retry_config)
        
        # Initialize tokenizer for precise token counting
        self._tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models (most OpenAI models use this)
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def get_messages(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> Iterable[ChatCompletionMessageParam]:
        messages = []
        if system_instruction:
            messages.append(SystemMessage(content=system_instruction).model_dump())
        if message_history:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError as e:
                raise LLMGenerationError(e.errors()) from e
            messages.extend(cast(Iterable[dict[str, Any]], message_history))
        messages.append(UserMessage(content=input).model_dump())
        return messages  # type: ignore

    def _convert_tool_to_openai_format(self, tool: Tool) -> Dict[str, Any]:
        """Convert a Tool object to OpenAI's expected format.

        Args:
            tool: A Tool object to convert to OpenAI's format.

        Returns:
            A dictionary in OpenAI's tool format.
        """
        try:
            return {
                "type": "function",
                "function": {
                    "name": tool.get_name(),
                    "description": tool.get_description(),
                    "parameters": tool.get_parameters(),
                },
            }
        except AttributeError:
            raise LLMGenerationError(f"Tool {tool} is not a valid Tool object")

    def _invoke_openai(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Internal method to call OpenAI API."""
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        
        response = self.client.chat.completions.create(
            messages=self.get_messages(input, message_history, system_instruction),
            model=self.model_name,
            **self.model_params,
        )
        
        # Update rate limiter with API feedback from headers
        if hasattr(response, '_response') and hasattr(response._response, 'headers'):
            headers = dict(response._response.headers)
            self._update_from_api_response_if_supported(headers)
        
        # Update token usage if rate limiter supports it
        if response.usage and response.usage.total_tokens:
            self._update_token_usage_if_supported(response.usage.total_tokens)
        
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content)

    async def _ainvoke_openai(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Internal async method to call OpenAI API."""
        if isinstance(message_history, MessageHistory):
            message_history = message_history.messages
        
        response = await self.async_client.chat.completions.create(
            messages=self.get_messages(input, message_history, system_instruction),
            model=self.model_name,
            **self.model_params,
        )
        
        # Update rate limiter with API feedback from headers
        if hasattr(response, '_response') and hasattr(response._response, 'headers'):
            headers = dict(response._response.headers)
            await self._aupdate_from_api_response_if_supported(headers)
        
        # Update token usage if rate limiter supports it
        if response.usage and response.usage.total_tokens:
            await self._aupdate_token_usage_if_supported(response.usage.total_tokens)
        
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content)

    def _invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Internal method that implements the LLM invocation for OpenAI.
        
        Rate limiting is handled automatically by the base class.
        """
        try:
            return self._invoke_openai(input, message_history, system_instruction)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    async def _ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Internal async method that implements the LLM invocation for OpenAI.
        
        Rate limiting is handled automatically by the base class.
        """
        try:
            return await self._ainvoke_openai(input, message_history, system_instruction)
        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    def _invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Internal method that implements tool calling for OpenAI.
        
        Rate limiting is handled automatically by the base class.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages

            params = self.model_params.copy() if self.model_params else {}
            if "temperature" not in params:
                params["temperature"] = 0.0

            # Convert tools to OpenAI's expected type
            openai_tools: List[ChatCompletionToolParam] = []
            for tool in tools:
                openai_format_tool = self._convert_tool_to_openai_format(tool)
                openai_tools.append(cast(ChatCompletionToolParam, openai_format_tool))

            response = self.client.chat.completions.create(
                messages=self.get_messages(input, message_history, system_instruction),
                model=self.model_name,
                tools=openai_tools,
                tool_choice="auto",
                **params,
            )
            
            # Update rate limiter with API feedback from headers
            if hasattr(response, '_response') and hasattr(response._response, 'headers'):
                headers = dict(response._response.headers)
                self._update_from_api_response_if_supported(headers)
            
            # Update token usage if rate limiter supports it
            if response.usage and response.usage.total_tokens:
                self._update_token_usage_if_supported(response.usage.total_tokens)

            message = response.choices[0].message

            # If there's no tool call, return the content as a regular response
            if not message.tool_calls or len(message.tool_calls) == 0:
                return ToolCallResponse(
                    tool_calls=[],
                    content=message.content,
                )

            # Process all tool calls
            tool_calls = []

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, AttributeError) as e:
                    raise LLMGenerationError(
                        f"Failed to parse tool call arguments: {e}"
                    )

                tool_calls.append(
                    ToolCall(name=tool_call.function.name, arguments=args)
                )

            return ToolCallResponse(tool_calls=tool_calls, content=message.content)

        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    async def _ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],  # Tools definition as a sequence of Tool objects
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Internal async method that implements tool calling for OpenAI.
        
        Rate limiting is handled automatically by the base class.
        """
        try:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages

            params = self.model_params.copy()
            if "temperature" not in params:
                params["temperature"] = 0.0

            # Convert tools to OpenAI's expected type
            openai_tools: List[ChatCompletionToolParam] = []
            for tool in tools:
                openai_format_tool = self._convert_tool_to_openai_format(tool)
                openai_tools.append(cast(ChatCompletionToolParam, openai_format_tool))

            response = await self.async_client.chat.completions.create(
                messages=self.get_messages(input, message_history, system_instruction),
                model=self.model_name,
                tools=openai_tools,
                tool_choice="auto",
                **params,
            )
            
            # Update rate limiter with API feedback from headers
            if hasattr(response, '_response') and hasattr(response._response, 'headers'):
                headers = dict(response._response.headers)
                await self._aupdate_from_api_response_if_supported(headers)
            
            # Update token usage if rate limiter supports it
            if response.usage and response.usage.total_tokens:
                await self._aupdate_token_usage_if_supported(response.usage.total_tokens)

            message = response.choices[0].message

            # If there's no tool call, return the content as a regular response
            if not message.tool_calls or len(message.tool_calls) == 0:
                return ToolCallResponse(
                    tool_calls=[ToolCall(name="", arguments={})],
                    content=message.content or "",
                )

            # Process all tool calls
            tool_calls = []
            import json

            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, AttributeError) as e:
                    raise LLMGenerationError(
                        f"Failed to parse tool call arguments: {e}"
                    )

                tool_calls.append(
                    ToolCall(name=tool_call.function.name, arguments=args)
                )

            return ToolCallResponse(tool_calls=tool_calls, content=message.content)

        except self.openai.OpenAIError as e:
            raise LLMGenerationError(e)

    def _count_message_tokens(self, messages: Iterable[ChatCompletionMessageParam]) -> int:
        """Count tokens in messages using tiktoken for precise counting."""
        if not self._tokenizer:
            # Fallback estimation if tiktoken not available
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            return max(1, total_chars // 4)  # Rough estimate: 4 chars per token
        
        total_tokens = 0
        
        # Count tokens for each message
        for message in messages:
            # OpenAI chat format overhead: each message has ~4 tokens of overhead
            total_tokens += 4
            
            # Count content tokens
            content = message.get('content', '')
            if content:
                total_tokens += len(self._tokenizer.encode(str(content)))
            
            # Count role tokens
            role = message.get('role', '')
            if role:
                total_tokens += len(self._tokenizer.encode(role))
        
        # Add 2 tokens for the assistant's reply priming
        total_tokens += 2
        
        return total_tokens

    def _count_tools_tokens(self, tools: Optional[Sequence[Tool]]) -> int:
        """Count tokens used by tool definitions."""
        if not tools or not self._tokenizer:
            return 0
        
        total_tokens = 0
        for tool in tools:
            # Convert tool to OpenAI format and count tokens
            tool_dict = self._convert_tool_to_openai_format(tool)
            tool_json = json.dumps(tool_dict)
            total_tokens += len(self._tokenizer.encode(tool_json))
        
        return total_tokens

    def _estimate_input_tokens(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[Sequence[Tool]] = None,
    ) -> int:
        """Estimate total input tokens for a request."""
        # Get messages that will be sent
        messages = list(self.get_messages(input, message_history, system_instruction))
        
        # Count message tokens
        message_tokens = self._count_message_tokens(messages)
        
        # Count tool tokens
        tool_tokens = self._count_tools_tokens(tools)
        
        return message_tokens + tool_tokens


class OpenAILLM(BaseOpenAILLM):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """OpenAI LLM

        Wrapper for the openai Python client LLM.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        # Extract rate limiting parameters from kwargs if present
        rate_limiter = kwargs.pop('rate_limiter', None)
        retry_config = kwargs.pop('retry_config', None)
        
        super().__init__(model_name, model_params, rate_limiter, retry_config)
        self.client = self.openai.OpenAI(**kwargs)
        self.async_client = self.openai.AsyncOpenAI(**kwargs)


class AzureOpenAILLM(BaseOpenAILLM):
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        """Azure OpenAI LLM. Use this class when using an OpenAI model
        hosted on Microsoft Azure.

        Args:
            model_name (str):
            model_params (str): Parameters like temperature that will be passed to the model when text is sent to it. Defaults to None.
            kwargs: All other parameters will be passed to the openai.OpenAI init.
        """
        # Extract rate limiting parameters from kwargs if present
        rate_limiter = kwargs.pop('rate_limiter', None)
        retry_config = kwargs.pop('retry_config', None)
        
        super().__init__(model_name, model_params, rate_limiter, retry_config)
        self.client = self.openai.AzureOpenAI(**kwargs)
        self.async_client = self.openai.AsyncAzureOpenAI(**kwargs)
