import asyncio
from typing import Any

from neo4j_graphrag.experimental.pipeline import Pipeline, Component, DataModel


class ComponentResult(DataModel):
    number: int


class MyComponent(Component):
    def __init__(self, f: int =  2):
        self.f = f

    async def run(self, value: int) -> ComponentResult:
        return ComponentResult(number=self.f * value)


def sub_pipeline(f):
    pipeline = Pipeline()
    pipeline.add_component(MyComponent(f=f), "internal_multiply")
    return pipeline


def main_pipeline():
    pipeline = Pipeline()
    pipeline.add_component(MyComponent(f=3), "multiply_by_3")
    pipeline_component = sub_pipeline(f=2)
    pipeline.add_component(pipeline_component, "pipeline_as_component")
    pipeline.add_component(MyComponent(f=10), "multiply_by_10")

    pipeline.connect("multiply_by_3", "pipeline_as_component", {
        "data": {
            "internal_multiply": "multiply_by_3.number"
        }
    })  # TODO: nested param parsing in validate_parameter_mapping_for_task
    pipeline.connect("pipeline_as_component", "multiply_by_10", {
        "number": "pipeline_as_component.result"
    })
    return pipeline


async def main() -> None:
    pipeline = main_pipeline()
    res = await pipeline.run({"pipeline_as_component": {"data": {}}})
    print(res)


if __name__ == "__main__":
    asyncio.run(main())
