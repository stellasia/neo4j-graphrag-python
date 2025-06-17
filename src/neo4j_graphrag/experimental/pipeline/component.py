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

import inspect
from typing import Any, AsyncGenerator, get_type_hints
from collections.abc import AsyncGenerator as AbcAsyncGenerator

from pydantic import BaseModel

from neo4j_graphrag.experimental.pipeline.types.context import RunContext
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.utils.validation import issubclass_safe


class DataModel(BaseModel):
    """Input or Output data model for Components"""

    pass


class ComponentMeta(type):
    def __new__(
        meta, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
        # Skip validation for the base Component class itself
        if name == "Component":
            return type.__new__(meta, name, bases, attrs)
            
        # extract required inputs and outputs from the run method signature
        run_method = attrs.get("run")
        run_context_method = attrs.get("run_with_context")
        run = run_context_method if run_context_method is not None else run_method
        if run is None:
            raise RuntimeError(
                f"You must implement either `run` or `run_with_context` in Component '{name}'"
            )
        sig = inspect.signature(run)
        attrs["component_inputs"] = {
            param.name: {
                "has_default": param.default != inspect.Parameter.empty,
                "annotation": param.annotation,
            }
            for param in sig.parameters.values()
            if param.name not in ("self", "kwargs", "context_")
        }
        # extract returned fields from the run method return type hint
        return_type = get_type_hints(run).get("return")
        if return_type is None:
            raise PipelineDefinitionError(
                f"The run method return type must be annotated in {name}"
            )
        
        # Must be AsyncGenerator[DataModel, None]
        if not (hasattr(return_type, '__origin__') and 
                (return_type.__origin__ is AsyncGenerator or return_type.__origin__ is AbcAsyncGenerator)):
            raise PipelineDefinitionError(
                f"The run method must return AsyncGenerator[DataModel, None] in {name}"
            )
        
        # Extract the yielded type from AsyncGenerator[DataModel, None]
        args = getattr(return_type, '__args__', ())
        if len(args) < 1:
            raise PipelineDefinitionError(
                f"AsyncGenerator return type must specify yielded type: AsyncGenerator[DataModel, None] in {name}"
            )
        
        return_model = args[0]  # First arg is the yielded type
        
        # the yielded type must be a subclass of DataModel
        if not issubclass_safe(return_model, DataModel):
            raise PipelineDefinitionError(
                f"The run method must yield a subclass of DataModel in {name}"
            )
        
        attrs["component_outputs"] = {
            f: {
                "has_default": field.is_required(),
                "annotation": field.annotation,
            }
            for f, field in return_model.model_fields.items()
        }
        return type.__new__(meta, name, bases, attrs)


class Component(metaclass=ComponentMeta):
    """Interface that needs to be implemented
    by all components.
    
    Components must yield results through AsyncGenerator.
    Each yielded result will create a new branch in the pipeline execution.
    """

    # these variables are filled by the metaclass
    # added here for the type checker
    # DO NOT CHANGE
    component_inputs: dict[str, dict[str, str | bool]]
    component_outputs: dict[str, dict[str, str | bool | type]]

    async def run(self, *args: Any, **kwargs: Any) -> AsyncGenerator[DataModel, None]:
        """Run the component and yield its results.
        
        Components must yield one or more results, with each result creating a new branch
        in the pipeline execution.

        Note: if `run_with_context` is implemented, this method will not be used.
        
        Yields:
            DataModel: Each yielded result creates a new execution branch
        """
        raise NotImplementedError(
            "You must implement the `run` or `run_with_context` method. "
        )
        # This is unreachable but needed for type checking
        yield  # type: ignore

    async def run_with_context(
        self, context_: RunContext, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[DataModel, None]:
        """This method is called by the pipeline orchestrator.
        The `context_` parameter contains information about
        the pipeline run: the `run_id` and a `notify` function
        that can be used to send events from the component to
        the pipeline callback.

        Components must yield one or more results, with each result creating a new branch
        in the pipeline execution.

        It defaults to calling the `run` method to prevent any breaking change.
        
        Yields:
            DataModel: Each yielded result creates a new execution branch
        """
        # default behavior to prevent a breaking change
        async for result in self.run(*args, **kwargs):
            yield result


