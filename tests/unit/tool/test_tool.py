import pytest
from typing import Any

from pydantic import TypeAdapter

from neo4j_graphrag.tool import (
    StringParameter,
    IntegerParameter,
    NumberParameter,
    BooleanParameter,
    ArrayParameter,
    ObjectParameter,
    Tool,
    AnyToolParameter,
    ParameterType,
)


def test_string_parameter() -> None:
    param = StringParameter(description="A string", enum=["a", "b"])
    assert param.description == "A string"
    assert param.enum == ["a", "b"]
    d = param.model_dump()
    assert d["type"] == ParameterType.STRING
    assert d["enum"] == ["a", "b"]


def test_integer_parameter() -> None:
    param = IntegerParameter(description="An int", minimum=0, maximum=10)
    d = param.model_dump()
    assert d["type"] == ParameterType.INTEGER
    assert d["minimum"] == 0
    assert d["maximum"] == 10


def test_number_parameter() -> None:
    param = NumberParameter(description="A number", minimum=1.5, maximum=3.5)
    d = param.model_dump()
    assert d["type"] == ParameterType.NUMBER
    assert d["minimum"] == 1.5
    assert d["maximum"] == 3.5


def test_boolean_parameter() -> None:
    param = BooleanParameter(description="A bool")
    d = param.model_dump()
    assert d["type"] == ParameterType.BOOLEAN
    assert d["description"] == "A bool"


def test_array_parameter_and_validation() -> None:
    arr_param = ArrayParameter(
        description="An array",
        items=StringParameter(description="str"),
        min_items=1,
        max_items=5,
    )
    d = arr_param.model_dump(by_alias=True)
    assert d["type"] == ParameterType.ARRAY
    assert d["items"]["type"] == ParameterType.STRING
    assert d["minItems"] == 1
    assert d["maxItems"] == 5

    # Test items as dict
    arr_param2 = ArrayParameter(
        description="Arr with dict",
        items={"type": "string", "description": "str"},  # type: ignore
    )
    assert isinstance(arr_param2.items, StringParameter)

    # Test error on invalid items
    with pytest.raises(ValueError):
        # Use type: ignore to bypass type checking for this intentional error case
        ArrayParameter(description="bad", items=123).validate_items()  # type: ignore


def test_object_parameter_and_validation() -> None:
    obj_param = ObjectParameter(
        description="Obj",
        properties={
            "foo": StringParameter(description="foo"),
            "bar": IntegerParameter(description="bar"),
        },
        required=["foo"],
        additional_properties=False,
    )
    d = obj_param.model_dump(by_alias=True)
    assert d["type"] == ParameterType.OBJECT
    assert d["properties"]["foo"]["type"] == ParameterType.STRING
    assert d["required"] == ["foo"]
    assert d["additionalProperties"] is False

    # Test properties as dicts
    obj_param2 = ObjectParameter(
        description="Obj2",
        properties={
            "foo": {"type": "string", "description": "foo"},  # type: ignore
        },
    )
    assert isinstance(obj_param2.properties["foo"], StringParameter)

    # Test error on invalid property
    with pytest.raises(ValueError):
        # Use type: ignore to bypass type checking for this intentional error case
        ObjectParameter(
            description="bad",
            properties={"foo": 123},  # type: ignore
        ).validate_properties()


def test_any_tool_parameter_union() -> None:
    adapter = TypeAdapter(AnyToolParameter)
    d = {"type": ParameterType.STRING, "description": "desc"}
    param = adapter.validate_python(d)
    assert isinstance(param, StringParameter)
    assert param.description == "desc"

    obj_dict = {
        "type": "object",
        "description": "obj",
        "properties": {"foo": {"type": "string", "description": "foo"}},
    }
    obj_param = adapter.validate_python(obj_dict)
    assert isinstance(obj_param, ObjectParameter)
    assert isinstance(obj_param.properties["foo"], StringParameter)

    arr_dict = {
        "type": "array",
        "description": "arr",
        "items": {"type": "integer", "description": "int"},
    }
    arr_param = adapter.validate_python(arr_dict)
    assert isinstance(arr_param, ArrayParameter)
    assert isinstance(arr_param.items, IntegerParameter)

    # Test unknown type
    with pytest.raises(ValueError):
        adapter.validate_python({"type": "unknown", "description": "bad"})

    # Test missing type
    with pytest.raises(ValueError):
        adapter.validate_python({"description": "no type"})


def test_required_parameter() -> None:
    object_param = ObjectParameter(
        description="Required object",
        properties={"prop": StringParameter(description="property")},
    )
    assert object_param.model_dump()["required"] == []

    object_param = ObjectParameter(
        description="Required object",
        properties={"prop": StringParameter(description="property")},
        required=["prop"],
    )
    assert object_param.model_dump()["required"] == ["prop"]


def test_tool_class() -> None:
    def dummy_func(query: str, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    params = ObjectParameter(
        description="params",
        properties={"a": StringParameter(description="a")},
    )
    tool = Tool(
        name="mytool",
        description="desc",
        parameters=params,
        execute_func=dummy_func,
    )
    assert tool.get_name() == "mytool"
    assert tool.get_description() == "desc"
    assert tool.get_parameters()["type"] == ParameterType.OBJECT
    assert tool.execute("query", a="b") == {"a": "b"}

    # Test parameters as dict
    params_dict = {
        "type": "object",
        "description": "params",
        "properties": {"a": {"type": "string", "description": "a"}},
    }
    tool2 = Tool(
        name="mytool2",
        description="desc2",
        parameters=params_dict,
        execute_func=dummy_func,
    )
    assert tool2.get_parameters()["type"] == ParameterType.OBJECT
    assert tool2.execute("query", a="b") == {"a": "b"}
