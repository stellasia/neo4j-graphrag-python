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
import uuid
import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, AsyncGenerator

from neo4j_graphrag.experimental.pipeline.component import DataModel
from neo4j_graphrag.experimental.pipeline.types.context import RunContext
from neo4j_graphrag.experimental.pipeline.exceptions import (
    PipelineDefinitionError,
    PipelineMissingDependencyError,
    PipelineStatusUpdateError,
)
from neo4j_graphrag.experimental.pipeline.notification import EventNotifier
from neo4j_graphrag.experimental.pipeline.types.orchestration import (
    RunResult,
    RunStatus,
)
from neo4j_graphrag.experimental.pipeline.run_graph import RunGraph

if TYPE_CHECKING:
    from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline, TaskPipelineNode

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrate a pipeline.

    The orchestrator is responsible for:
    - finding the next tasks to execute
    - building the inputs for each task
    - calling the run method on each task

    Once a TaskNode is done, it calls the `on_task_complete` callback
    that will save the results, find the next tasks to be executed
    (checking that all dependencies are met), and run them.
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.event_notifier = EventNotifier(pipeline.callbacks)
        # Create RunGraph as intermediate layer between orchestrator and store
        self.run_graph = RunGraph(store=pipeline.store)
        self.run_id = self.run_graph.run_id

    async def run_task(self, task: TaskPipelineNode, data: dict[str, Any], branch_id: str | None = None) -> None:
        """Get inputs and run a specific task. Once the task is done,
        calls the on_task_complete method for each result.

        Args:
            task (TaskPipelineNode): The task to be run
            data (dict[str, Any]): The pipeline input data
            branch_id (str | None): The branch ID to execute this task in. If None, uses root branch.

        Returns:
            None
        """
        # Use provided branch_id or default to root branch
        current_branch_id = branch_id or self.run_graph.run_id
        
        param_mapping = self.get_input_config_for_task(task)
        inputs = await self.get_component_inputs(task.name, param_mapping, data, current_branch_id)
        try:
            await self.set_task_status(task.name, RunStatus.RUNNING, current_branch_id)
        except PipelineStatusUpdateError:
            logger.debug(
                f"ORCHESTRATOR: TASK ABORTED: {task.name} is already running or done, aborting"
            )
            return None
        await self.event_notifier.notify_task_started(self.run_id, task.name, inputs)
        # create the notifier function for the component, with fixed
        # run_id, task_name and event type:
        notifier = partial(
            self.event_notifier.notify_task_progress,
            run_id=self.run_id,
            task_name=task.name,
        )
        context = RunContext(run_id=self.run_id, task_name=task.name, notifier=notifier)
        
        # Components now return AsyncGenerator, so we iterate over results
        # Each result creates a new branch in the pipeline execution
        result_count = 0
        created_branches = []
        
        async for run_result in task.run(context, inputs):
            result_count += 1
            
            if result_count == 1:
                # First result: store in current branch
                result_branch_id = current_branch_id
                # Mark task as DONE after processing the first result
                # This allows downstream components to start
                await self.set_task_status(task.name, RunStatus.DONE, current_branch_id)
            else:
                # Subsequent results: create new branch
                result_branch_id = self.run_graph.create_branch(parent_branch_id=current_branch_id)
                created_branches.append(result_branch_id)
                # Also mark task as RUNNING then DONE in the new branch
                await self.set_task_status(task.name, RunStatus.RUNNING, result_branch_id)
                await self.set_task_status(task.name, RunStatus.DONE, result_branch_id)
            
            await self.event_notifier.notify_task_finished(self.run_id, task.name, run_result)
            # Handle each result in its own branch - this triggers downstream components
            await self.on_task_complete(data=data, task=task, result=run_result, branch_id=result_branch_id)
        
        if result_count == 0:
            logger.warning(f"Component {task.name} yielded no results")
            # Still mark as done even if no results were yielded
            await self.set_task_status(task.name, RunStatus.DONE, current_branch_id)
            await self.event_notifier.notify_task_finished(self.run_id, task.name, None)

    async def set_task_status(self, task_name: str, status: RunStatus, branch_id: str) -> None:
        """Set a new status for a task in a specific branch

        Args:
            task_name (str): Name of the component
            status (RunStatus): New status
            branch_id (str): The branch ID for this status

        Raises:
            PipelineStatusUpdateError if the new status is not
                compatible with the current one.
        """
        # prevent the method from being called by two concurrent async calls
        async with asyncio.Lock():
            current_status = await self.get_status_for_component(task_name, branch_id)
            if status == current_status:
                raise PipelineStatusUpdateError(f"Status is already {status} for {task_name} in branch {branch_id}")
            if status not in current_status.possible_next_status():
                raise PipelineStatusUpdateError(
                    f"Can't go from {current_status} to {status} for {task_name} in branch {branch_id}"
                )
            # Store status with branch-specific key
            status_key = f"{self.run_id}:{task_name}:{branch_id}:status"
            return await self.pipeline.store.add(status_key, status.value, overwrite=True)

    async def on_task_complete(
        self, data: dict[str, Any], task: TaskPipelineNode, result: RunResult, branch_id: str
    ) -> None:
        """When a given task is complete, it will call this method
        to find the next tasks to run.
        """
        # first save this component results
        res_to_save = None
        if result.result:
            res_to_save = result.result
        
        await self.add_result_for_component(
            task.name, res_to_save, branch_id, is_final=task.is_leaf()
        )
        # then get the next tasks to be executed
        # and run them in the specific branch
        next_tasks = []
        async for n in self.next(task, branch_id):
            next_tasks.append(n)
        
        await asyncio.gather(*[self.run_task(n, data, branch_id) for n in next_tasks])

    async def add_result_to_final_store(self, component_name: str, result: Any) -> None:
        """Add result to the final results store for pipeline output."""
        # The pipeline only returns the results of the leaf nodes
        # TODO: make this configurable in the future.
        result_data = result.model_dump() if result else None
        existing_results = await self.pipeline.final_results.get(self.run_id) or {}
        existing_results[component_name] = result_data
        await self.pipeline.final_results.add(
            self.run_id, existing_results, overwrite=True
        )

    async def check_dependencies_complete(self, task: TaskPipelineNode, branch_id: str) -> None:
        """Check that all parent tasks are complete in the given branch.

        Raises:
            MissingDependencyError if a parent task's status is not DONE.
        """
        dependencies = self.pipeline.previous_edges(task.name)
        for d in dependencies:
            d_status = await self.get_status_for_component(d.start, branch_id)
            if d_status != RunStatus.DONE:
                logger.debug(
                    f"ORCHESTRATOR {self.run_id}: TASK DELAYED: Missing dependency {d.start} for {task.name} "
                    f"(status: {d_status}) in branch {branch_id}. "
                    "Will try again when dependency is complete."
                )
                raise PipelineMissingDependencyError()

    async def next(
        self, task: TaskPipelineNode, branch_id: str
    ) -> AsyncGenerator[TaskPipelineNode, None]:
        """Find the next tasks to be executed after `task` is complete in the given branch.

        1. Find the task children
        2. Check each child's status:
            - if it's already running or done, do not need to run it again
            - otherwise, check that all its dependencies are met, if yes
                add this task to the list of next tasks to be executed
        """
        possible_next = self.pipeline.next_edges(task.name)
        
        for next_edge in possible_next:
            next_node = self.pipeline.get_node_by_name(next_edge.end)
            # check status in the specific branch
            next_node_status = await self.get_status_for_component(next_node.name, branch_id)
            
            if next_node_status in [RunStatus.RUNNING, RunStatus.DONE]:
                # already running
                continue
            # check deps
            try:
                await self.check_dependencies_complete(next_node, branch_id)
            except PipelineMissingDependencyError:
                continue
            logger.debug(
                f"ORCHESTRATOR {self.run_id}: enqueuing next task: {next_node.name} in branch {branch_id}"
            )
            yield next_node
        return

    def get_input_config_for_task(
        self, task: TaskPipelineNode
    ) -> dict[str, dict[str, str]]:
        """Build input definition for a given task.,
        The method needs to access the input defs defined in the edges
        between this task and its parents.

        Args:
            task (TaskPipelineNode): the task to get the input config for

        Returns:
            dict: a dict of
                {input_parameter: {source_component_name: "", param_name: ""}}
        """
        if not self.pipeline.is_validated:
            raise PipelineDefinitionError(
                "You must validate the pipeline input config first. Call `pipeline.validate_parameter_mapping()`"
            )
        return self.pipeline.param_mapping.get(task.name) or {}

    async def get_component_inputs(
        self,
        component_name: str,
        param_mapping: dict[str, dict[str, str]],
        input_data: dict[str, Any],
        branch_id: str,
    ) -> dict[str, Any]:
        """Find the component inputs from:
        - input_config: the mapping between components results and inputs
            (results are retrieved via RunGraph for the specific branch)
        - input_data: the user input data

        Args:
            component_name (str): the component/task name
            param_mapping (dict[str, dict[str, str]]): the input config
            input_data (dict[str, Any]): the pipeline input data (user input)
            branch_id (str): the branch ID to get results from
        """
        component_inputs: dict[str, Any] = input_data.get(component_name, {})
        if param_mapping:
            for parameter, mapping in param_mapping.items():
                component = mapping["component"]
                output_param = mapping.get("param")
                # Get results from the specific branch
                component_result = await self.get_results_for_component(component, branch_id)
                if component_result is not None:
                    if output_param is not None:
                        value = component_result.get(output_param)
                    else:
                        value = component_result
                    
                    if parameter in component_inputs:
                        m = f"{component}.{parameter}" if parameter else component
                        warnings.warn(
                            f"In component '{component_name}', parameter '{parameter}' from user input will be ignored and replaced by '{m}'"
                        )
                    component_inputs[parameter] = value
        return component_inputs

    async def add_result_for_component(
        self, name: str, result: DataModel | None, branch_id: str, is_final: bool = False
    ) -> None:
        """Add result for a component in a specific branch."""
        await self.run_graph.add_result_for_branch(
            branch_id, name, result
        )
        if is_final:
            await self.add_result_to_final_store(name, result)

    async def get_results_for_component(self, name: str, branch_id: str | None = None) -> Any:
        """Get results for a component using RunGraph from a specific branch."""
        branch_to_use = branch_id or self.run_graph.run_id
        return await self.run_graph.get_result_for_branch(
            branch_to_use, name
        )

    async def get_status_for_component(self, name: str, branch_id: str) -> RunStatus:
        """Get status for a component in a specific branch."""
        status_key = f"{self.run_id}:{name}:{branch_id}:status"
        status = await self.pipeline.store.get(status_key)
        if status is None:
            return RunStatus.UNKNOWN
        return RunStatus(status)
    
    async def run(self, data: dict[str, Any]) -> None:
        """Run the pipline, starting from the root nodes
        (node without any parent). Then the callback on_task_complete
        will handle the task dependencies.
        """
        await self.event_notifier.notify_pipeline_started(self.run_id, data)
        tasks = [self.run_task(root, data, self.run_graph.run_id) for root in self.pipeline.roots()]
        await asyncio.gather(*tasks)
        await self.event_notifier.notify_pipeline_finished(
            self.run_id, await self.pipeline.get_final_results(self.run_id)
        )

