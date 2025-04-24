from abc import ABC
from enum import Enum
from typing import Any, Dict, List, Callable, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict, AliasGenerator
from pydantic.alias_generators import to_camel, to_snake


class ParameterType(str, Enum):
    """Enum for parameter types supported in tool parameters."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class ToolParameter(BaseModel):
    """Base class for all tool parameters using Pydantic."""

    description: str
    type: ParameterType


class StringParameter(ToolParameter):
    """String parameter for tools."""

    type: Literal[ParameterType.STRING] = ParameterType.STRING
    enum: Optional[List[str]] = None


class IntegerParameter(ToolParameter):
    """Integer parameter for tools."""

    type: Literal[ParameterType.INTEGER] = ParameterType.INTEGER
    minimum: Optional[int] = None
    maximum: Optional[int] = None


class NumberParameter(ToolParameter):
    """Number parameter for tools."""

    type: Literal[ParameterType.NUMBER] = ParameterType.NUMBER
    minimum: Optional[float] = None
    maximum: Optional[float] = None


class BooleanParameter(ToolParameter):
    """Boolean parameter for tools."""

    type: Literal[ParameterType.BOOLEAN] = ParameterType.BOOLEAN


class ArrayParameter(ToolParameter):
    """Array parameter for tools."""

    type: Literal[ParameterType.ARRAY] = ParameterType.ARRAY
    items: "AnyToolParameter"
    min_items: Optional[int] = None
    max_items: Optional[int] = None

    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=to_snake,
            serialization_alias=to_camel,
        )
    )


class ObjectParameter(ToolParameter):
    """Object parameter for tools."""

    type: Literal[ParameterType.OBJECT] = ParameterType.OBJECT
    properties: Dict[str, "AnyToolParameter"]
    required: List[str] = []
    additional_properties: bool = True

    model_config = ConfigDict(
        alias_generator=AliasGenerator(
            validation_alias=to_snake,
            serialization_alias=to_camel,
        )
    )


AnyToolParameter = Annotated[
    Union[
        StringParameter,
        IntegerParameter,
        NumberParameter,
        BooleanParameter,
        ObjectParameter,
        ArrayParameter,
    ],
    Field(discriminator="type"),
]


class Tool(ABC):
    """Abstract base class defining the interface for all tools in the neo4j-graphrag library."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Union[ObjectParameter, Dict[str, Any]],
        execute_func: Callable[..., Any],
    ):
        self._name = name
        self._description = description

        # Allow parameters to be provided as a dictionary
        self._parameters = ObjectParameter.model_validate(parameters)
        self._execute_func = execute_func

    def get_name(self) -> str:
        """Get the name of the tool.

        Returns:
            str: Name of the tool.
        """
        return self._name

    def get_description(self) -> str:
        """Get a detailed description of what the tool does.

        Returns:
            str: Description of the tool.
        """
        return self._description

    def get_parameters(self, exclude: Optional[list[str]] = None) -> Dict[str, Any]:
        """Get the parameters the tool accepts in a dictionary format suitable for LLM providers.

        Returns:
            Dict[str, Any]: Dictionary containing parameter schema information.
        """
        return self._parameters.model_dump(
            by_alias=True,  # camelCase
            exclude_none=True,  # exclude None values
            exclude=exclude,  # exclude any specific field
        )

    def execute(self, query: str, **kwargs: Any) -> Any:
        """Execute the tool with the given query and additional parameters.

        Args:
            query (str): The query or input for the tool to process.
            **kwargs (Any): Additional parameters for the tool.

        Returns:
            Any: The result of the tool execution.
        """
        return self._execute_func(query, **kwargs)
