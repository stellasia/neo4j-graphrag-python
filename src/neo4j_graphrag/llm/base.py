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

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable, List, Optional, Sequence, TypeVar, Union

from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage

from .types import LLMResponse, ToolCallResponse
from .rate_limiter import BaseRateLimiter, RetryConfig, is_rate_limit_error
from ..exceptions import LLMGenerationError

from neo4j_graphrag.tool import Tool

logger = logging.getLogger(__name__)

# Type variable for function return types
T = TypeVar("T")


class LLMInterface(ABC):
    """Interface for large language models.

    Args:
        model_name (str): The name of the language model.
        model_params (Optional[dict]): Additional parameters passed to the model when text is sent to it. Defaults to None.
        rate_limiter (Optional[BaseRateLimiter]): Rate limiter to control request frequency. Defaults to None.
        retry_config (Optional[RetryConfig]): Configuration for retry behavior on rate limit errors. Defaults to None.
        **kwargs (Any): Arguments passed to the model when for the class is initialised. Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        rate_limiter: Optional[BaseRateLimiter] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.model_params = model_params or {}
        
        # Rate limiting setup
        self.rate_limiter = rate_limiter
        self.retry_config = retry_config or RetryConfig()
        self._llm_name = f"{self.__class__.__name__}({model_name})"

    def _apply_rate_limiting(self, func: Callable[..., T]) -> Callable[..., T]:
        """Apply rate limiting to a function if rate limiter is configured."""
        if not self.rate_limiter:
            return func
            
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(self.retry_config.max_retries + 1):
                # Get precise token estimate if implementation supports it
                estimated_tokens = None
                if hasattr(self, '_estimate_input_tokens'):
                    try:
                        # Extract parameters based on function signature
                        if len(args) >= 1:  # input is first positional arg
                            input_text = args[0]
                            message_history = args[1] if len(args) >= 2 else kwargs.get('message_history')
                            system_instruction = args[2] if len(args) >= 3 else kwargs.get('system_instruction')
                            tools = args[3] if len(args) >= 4 else kwargs.get('tools')
                            
                            estimated_tokens = self._estimate_input_tokens(
                                input_text, message_history, system_instruction, tools
                            )
                    except Exception:
                        # If token estimation fails, continue without it
                        estimated_tokens = None
                
                # Wait for rate limiter with precise token count if available
                if estimated_tokens is not None:
                    while not self.rate_limiter.acquire(estimated_tokens):
                        time.sleep(0.1)
                else:
                    while not self.rate_limiter.acquire():
                        time.sleep(0.1)
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if is_rate_limit_error(e) and attempt < self.retry_config.max_retries:
                        delay = self.retry_config.get_delay(attempt)
                        logger.warning(
                            f"{self._llm_name}: Rate limit hit, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        # Re-raise if not a rate limit error or max retries reached
                        raise
            
            # This shouldn't be reached, but just in case
            raise last_error or LLMGenerationError("Max retries exceeded")
        
        return wrapper

    def _apply_async_rate_limiting(self, func: Callable[..., T]) -> Callable[..., T]:
        """Apply rate limiting to an async function if rate limiter is configured."""
        if not self.rate_limiter:
            return func
            
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(self.retry_config.max_retries + 1):
                # Get precise token estimate if implementation supports it
                estimated_tokens = None
                if hasattr(self, '_estimate_input_tokens'):
                    try:
                        # Extract parameters based on function signature
                        if len(args) >= 1:  # input is first positional arg
                            input_text = args[0]
                            message_history = args[1] if len(args) >= 2 else kwargs.get('message_history')
                            system_instruction = args[2] if len(args) >= 3 else kwargs.get('system_instruction')
                            tools = args[3] if len(args) >= 4 else kwargs.get('tools')
                            
                            estimated_tokens = self._estimate_input_tokens(
                                input_text, message_history, system_instruction, tools
                            )
                    except Exception:
                        # If token estimation fails, continue without it
                        estimated_tokens = None
                
                # Wait for rate limiter with precise token count if available
                if estimated_tokens is not None:
                    while not await self.rate_limiter.aacquire(estimated_tokens):
                        await asyncio.sleep(0.1)
                else:
                    while not await self.rate_limiter.aacquire():
                        await asyncio.sleep(0.1)
                
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    if is_rate_limit_error(e) and attempt < self.retry_config.max_retries:
                        delay = self.retry_config.get_delay(attempt)
                        logger.warning(
                            f"{self._llm_name}: Rate limit hit, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{self.retry_config.max_retries + 1})"
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Re-raise if not a rate limit error or max retries reached
                        raise
            
            # This shouldn't be reached, but just in case
            raise last_error or LLMGenerationError("Max retries exceeded")
        
        return wrapper

    # Abstract methods that implementations must override
    @abstractmethod
    def _invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Internal method to send a text input to the LLM and retrieve a response.
        
        This method should be overridden by concrete implementations.
        Rate limiting will be applied automatically by the public invoke method.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """

    @abstractmethod
    async def _ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Internal async method to send a text input to the LLM and retrieve a response.
        
        This method should be overridden by concrete implementations.
        Rate limiting will be applied automatically by the public ainvoke method.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """

    def _invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Internal method to send a text input to the LLM with tool definitions.
        
        This method can be overridden by concrete implementations that support tool calling.
        Rate limiting will be applied automatically by the public invoke_with_tools method.

        Args:
            input (str): Text sent to the LLM.
            tools (Sequence[Tool]): Sequence of Tools for the LLM to choose from. Each LLM implementation should handle the conversion to its specific format.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support tool calling.
        """
        raise NotImplementedError("This LLM provider does not support tool calling.")

    async def _ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Internal async method to send a text input to the LLM with tool definitions.
        
        This method can be overridden by concrete implementations that support tool calling.
        Rate limiting will be applied automatically by the public ainvoke_with_tools method.

        Args:
            input (str): Text sent to the LLM.
            tools (Sequence[Tool]): Sequence of Tools for the LLM to choose from. Each LLM implementation should handle the conversion to its specific format.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support tool calling.
        """
        raise NotImplementedError("This LLM provider does not support tool calling.")

    # Public methods that automatically apply rate limiting
    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Sends a text input to the LLM and retrieves a response.
        
        Rate limiting is applied automatically if configured.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        rate_limited_invoke = self._apply_rate_limiting(self._invoke)
        return rate_limited_invoke(input, message_history, system_instruction)

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        """Asynchronously sends a text input to the LLM and retrieves a response.
        
        Rate limiting is applied automatically if configured.

        Args:
            input (str): Text sent to the LLM.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            LLMResponse: The response from the LLM.

        Raises:
            LLMGenerationError: If anything goes wrong.
        """
        rate_limited_ainvoke = self._apply_async_rate_limiting(self._ainvoke)
        return await rate_limited_ainvoke(input, message_history, system_instruction)

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Sends a text input to the LLM with tool definitions and retrieves a tool call response.
        
        Rate limiting is applied automatically if configured.

        Args:
            input (str): Text sent to the LLM.
            tools (Sequence[Tool]): Sequence of Tools for the LLM to choose from. Each LLM implementation should handle the conversion to its specific format.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support tool calling.
        """
        rate_limited_invoke_with_tools = self._apply_rate_limiting(self._invoke_with_tools)
        return rate_limited_invoke_with_tools(input, tools, message_history, system_instruction)

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        """Asynchronously sends a text input to the LLM with tool definitions and retrieves a tool call response.
        
        Rate limiting is applied automatically if configured.

        Args:
            input (str): Text sent to the LLM.
            tools (Sequence[Tool]): Sequence of Tools for the LLM to choose from. Each LLM implementation should handle the conversion to its specific format.
            message_history (Optional[Union[List[LLMMessage], MessageHistory]]): A collection previous messages,
                with each message having a specific role assigned.
            system_instruction (Optional[str]): An option to override the llm system message for this invocation.

        Returns:
            ToolCallResponse: The response from the LLM containing a tool call.

        Raises:
            LLMGenerationError: If anything goes wrong.
            NotImplementedError: If the LLM provider does not support tool calling.
        """
        rate_limited_ainvoke_with_tools = self._apply_async_rate_limiting(self._ainvoke_with_tools)
        return await rate_limited_ainvoke_with_tools(input, tools, message_history, system_instruction)
