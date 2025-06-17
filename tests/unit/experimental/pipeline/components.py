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
import asyncio
from typing import AsyncGenerator

from neo4j_graphrag.experimental.pipeline import Component, DataModel
from neo4j_graphrag.experimental.pipeline.types.context import RunContext


class StringResultModel(DataModel):
    result: str


class IntResultModel(DataModel):
    result: int


class ComponentNoParam(Component):
    async def run(self) -> AsyncGenerator[StringResultModel, None]:
        yield StringResultModel(result="")


class ComponentPassThrough(Component):
    async def run(self, value: str) -> AsyncGenerator[StringResultModel, None]:
        yield StringResultModel(result=f"value is: {value}")


class ComponentAdd(Component):
    async def run(self, number1: int, number2: int) -> AsyncGenerator[IntResultModel, None]:
        yield IntResultModel(result=number1 + number2)


class ComponentMultiply(Component):
    async def run(self, number1: int, number2: int = 2) -> AsyncGenerator[IntResultModel, None]:
        yield IntResultModel(result=number1 * number2)


class ComponentMultiplyWithContext(Component):
    async def run_with_context(
        self, context_: RunContext, number1: int, number2: int = 2
    ) -> AsyncGenerator[IntResultModel, None]:
        await context_.notify(
            message="my message", data={"number1": number1, "number2": number2}
        )
        yield IntResultModel(result=number1 * number2)


class SlowComponentMultiply(Component):
    def __init__(self, sleep: float = 1.0) -> None:
        self.sleep = sleep

    async def run(self, number1: int, number2: int = 2) -> AsyncGenerator[IntResultModel, None]:
        await asyncio.sleep(self.sleep)
        yield IntResultModel(result=number1 * number2)


class ComponentMultipleResults(Component):
    """Component that yields multiple results to test branching."""
    
    async def run(self, input_value: str) -> AsyncGenerator[StringResultModel, None]:
        """Yield multiple results with different values."""
        # Yield first result
        yield StringResultModel(result=f"{input_value}_branch_1")
        # Yield second result  
        yield StringResultModel(result=f"{input_value}_branch_2")
        # Yield third result
        yield StringResultModel(result=f"{input_value}_branch_3")


class ComponentBranchTracker(Component):
    """Component that tracks which branch it's processing."""
    
    component_inputs = {
        "input_value": str,
    }
    component_outputs = {
        "result": str,
        "processed_input": str,
    }
    
    async def run(self, input_value: str) -> AsyncGenerator[StringResultModel, None]:
        """Process input and yield result with the input value embedded."""
        yield StringResultModel(result=f"processed_{input_value}")
