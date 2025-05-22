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
from typing import Any, get_type_hints, Dict

from pydantic import BaseModel

from neo4j_graphrag.experimental.pipeline.types.context import RunContext
from neo4j_graphrag.experimental.pipeline.exceptions import PipelineDefinitionError
from neo4j_graphrag.utils.validation import issubclass_safe
from neo4j_graphrag.experimental.pipeline.serializable import Serializable


class DataModel(BaseModel):
    """Input or Output data model for Components"""

    pass


class ComponentMeta(type):
    def __new__(
        meta, name: str, bases: tuple[type, ...], attrs: dict[str, Any]
    ) -> type:
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
        return_model = get_type_hints(run).get("return")
        if return_model is None:
            raise PipelineDefinitionError(
                f"The run method return type must be annotated in {name}"
            )
        # the type hint must be a subclass of DataModel
        if not issubclass_safe(return_model, DataModel):
            raise PipelineDefinitionError(
                f"The run method must return a subclass of DataModel in {name}"
            )
        attrs["component_outputs"] = {
            f: {
                "has_default": field.is_required(),
                "annotation": field.annotation,
            }
            for f, field in return_model.model_fields.items()
        }
        return type.__new__(meta, name, bases, attrs)


class Component(Serializable, metaclass=ComponentMeta):
    """Interface that needs to be implemented
    by all components.
    """

    # these variables are filled by the metaclass
    # added here for the type checker
    # DO NOT CHANGE
    component_inputs: dict[str, dict[str, str | bool]]
    component_outputs: dict[str, dict[str, str | bool | type]]

    async def run(self, *args: Any, **kwargs: Any) -> DataModel:
        """Run the component and return its result.

        Note: if `run_with_context` is implemented, this method will not be used.
        """
        raise NotImplementedError(
            "You must implement the `run` or `run_with_context` method. "
        )

    async def run_with_context(
        self, context_: RunContext, *args: Any, **kwargs: Any
    ) -> DataModel:
        """This method is called by the pipeline orchestrator.
        The `context_` parameter contains information about
        the pipeline run: the `run_id` and a `notify` function
        that can be used to send events from the component to
        the pipeline callback.

        This feature will be moved to the `run` method in a future
        release.

        It defaults to calling the `run` method to prevent any breaking change.
        """
        # default behavior to prevent a breaking change
        return await self.run(*args, **kwargs)
        
    def _serialize(self) -> Dict[str, Any]:
        """Serialize the component to a dictionary.
        
        Returns:
            Dictionary with component data
        """
        # By default, serialize all public attributes that aren't methods
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value):
                result[key] = value
        return result
    
    @classmethod
    def _deserialize(cls, data: Dict[str, Any]) -> Component:
        """Deserialize a component from a dictionary.
        
        Args:
            data: Dictionary with serialized component data
            
        Returns:
            Component instance
        """
        # Remove the type key
        component_data = {k: v for k, v in data.items() if k != "__type__"}
        
        # Create a new instance
        instance = cls.__new__(cls)
        
        # Initialize the instance with the data
        for key, value in component_data.items():
            setattr(instance, key, value)
            
        return instance
