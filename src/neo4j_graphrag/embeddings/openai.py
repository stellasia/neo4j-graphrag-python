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
import os
import logging
from typing import TYPE_CHECKING, Any, Optional

from neo4j_graphrag.embeddings.base import Embedder

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)


class BaseOpenAIEmbeddings(Embedder, abc.ABC):
    """
    Abstract base class for OpenAI embeddings.
    """

    client: openai.OpenAI

    def __init__(self, model: str = "text-embedding-ada-002", **kwargs: Any) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                """Could not import openai python client.
                Please install it with `pip install "neo4j-graphrag[openai]"`."""
            )
        self.openai = openai
        self.model = model
        self.client = self._initialize_client(**kwargs)

    @abc.abstractmethod
    def _initialize_client(self, **kwargs: Any) -> Any:
        """
        Initialize the OpenAI client.
        Must be implemented by subclasses.
        """
        pass

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """
        Generate embeddings for a given query using an OpenAI text embedding model.

        Args:
            text (str): The text to generate an embedding for.
            **kwargs (Any): Additional arguments to pass to the OpenAI embedding generation function.
        """
        response = self.client.embeddings.create(input=text, model=self.model, **kwargs)
        embedding: list[float] = response.data[0].embedding
        return embedding


class OpenAIEmbeddings(BaseOpenAIEmbeddings):
    """
    OpenAI embeddings class.
    This class uses the OpenAI python client to generate embeddings for text data.

    Args:
        model (str): The name of the OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        kwargs: All other parameters will be passed to the openai.OpenAI init.
    """

    def _initialize_client(self, **kwargs: Any) -> Any:
        return self.openai.OpenAI(**kwargs)


class AzureOpenAIEmbeddings(BaseOpenAIEmbeddings):
    """
    Azure OpenAI embeddings class.
    This class uses the Azure OpenAI python client to generate embeddings for text data.

    Args:
        model (str): The name of the Azure OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        kwargs: All other parameters will be passed to the openai.AzureOpenAI init.
    """

    def _initialize_client(self, **kwargs: Any) -> Any:
        return self.openai.AzureOpenAI(**kwargs)


class LazyOpenAIEmbeddings:
    """A lazy-loading wrapper for OpenAI embeddings that defers client creation.
    
    This class stores the configuration needed to create an OpenAI embedder but doesn't
    actually create the client until it's used. This ensures the client is created 
    in the worker process, not the main process, avoiding serialization issues.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        azure: bool = False,
        **kwargs: Any,
    ):
        self._model = model
        self._azure = azure
        self._kwargs = kwargs
        self._actual_embedder: Optional[BaseOpenAIEmbeddings] = None
        self._created_in_process_id = None
    
    def _ensure_embedder(self) -> BaseOpenAIEmbeddings:
        """Ensure we have an actual embedder, creating it if necessary."""
        current_process_id = os.getpid()
        
        # Create embedder if it doesn't exist or we're in a different process (worker)
        if self._actual_embedder is None or self._created_in_process_id != current_process_id:
            if self._azure:
                self._actual_embedder = AzureOpenAIEmbeddings(
                    self._model,
                    **self._kwargs
                )
            else:
                self._actual_embedder = OpenAIEmbeddings(
                    self._model,
                    **self._kwargs
                )
            
            self._created_in_process_id = current_process_id
            logger.debug(f"Created OpenAI embeddings client in process {current_process_id}")
        
        return self._actual_embedder
    
    # Delegate all Embedder methods to the actual embedder
    def __getattr__(self, name: str) -> Any:
        """Delegate to the actual OpenAI embedder, creating it if necessary."""
        return getattr(self._ensure_embedder(), name)
    
    # Support serialization for distributed execution
    def __getstate__(self) -> dict:
        """Custom serialization - only serialize config, not the actual client."""
        return {
            '_model': self._model,
            '_azure': self._azure,
            '_kwargs': self._kwargs,
            # Don't serialize the actual embedder or process ID
        }
    
    def __setstate__(self, state: dict) -> None:
        """Custom deserialization - restore config, embedder will be created on demand."""
        self._model = state['_model']
        self._azure = state['_azure']
        self._kwargs = state['_kwargs']
        self._actual_embedder = None
        self._created_in_process_id = None
