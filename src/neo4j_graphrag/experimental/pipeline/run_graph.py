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

from typing import Optional, Dict, Any
from neo4j_graphrag.experimental.pipeline.component import DataModel
from neo4j_graphrag.experimental.pipeline.pipeline_graph import PipelineGraph, PipelineNode, PipelineEdge
from neo4j_graphrag.experimental.pipeline.stores import ResultStore
import uuid


class RunBranch(PipelineNode):
    """
    Represents a branch in the pipeline execution.
    Each branch has exactly one result per component.
    Parent/child relationships are managed by the graph structure.
    Results are stored in the persistent store, not in memory.
    """
    def __init__(self, branch_id: str):
        super().__init__(name=branch_id, data={})
        self.branch_id = branch_id


class RunEdge(PipelineEdge):
    """
    Represents a branching event from one branch to another.
    Contains information about which component and result index caused the branching.
    """
    def __init__(
        self,
        start: str,
        end: str,
        parent_component: Optional[str] = None,
        parent_result_idx: Optional[int] = None,
    ):
        data = {
            "parent_component": parent_component,
            "parent_result_idx": parent_result_idx,
        }
        super().__init__(start=start, end=end, data=data)
    
    @property
    def parent_component(self) -> Optional[str]:
        return self.data.get("parent_component") if self.data else None
    
    @property
    def parent_result_idx(self) -> Optional[int]:
        return self.data.get("parent_result_idx") if self.data else None


class RunGraph(PipelineGraph[RunBranch, RunEdge]):
    """
    Tracks all branches and their results during pipeline execution.
    Each branch has exactly one result per component.
    Uses graph structure to manage branch relationships and stores results in a persistent store.
    """
    def __init__(self, store: ResultStore):
        super().__init__()
        self.run_id = self._generate_branch_id()
        self.store = store
        # Create the root branch by default
        root_branch = RunBranch(branch_id=self.run_id)
        self.add_node(root_branch)

    def _generate_branch_id(self) -> str:
        return str(uuid.uuid4())

    def _get_store_key(self, component_name: str, branch_id: str) -> str:
        """Generate a unique store key for a component result in a branch."""
        return f"{self.run_id}:{component_name}:{branch_id}"

    def add_branch(
        self,
        parent_id: Optional[str] = None,
        parent_component: Optional[str] = None,
        parent_result_idx: Optional[int] = None,
    ) -> str:
        """
        Create a new branch, optionally as a child of an existing branch.
        Returns the new branch's ID.
        """
        branch_id = self._generate_branch_id()
        branch = RunBranch(branch_id=branch_id)
        self.add_node(branch)
        
        # If parent is specified, create an edge
        if parent_id is not None:
            edge = RunEdge(
                start=parent_id,
                end=branch_id,
                parent_component=parent_component,
                parent_result_idx=parent_result_idx,
            )
            self.add_edge(edge)
        
        return branch_id

    def get_branch(self, branch_id: str) -> Optional[RunBranch]:
        """
        Retrieve a branch by its ID. Returns None if not found.
        """
        try:
            return self.get_node_by_name(branch_id)
        except KeyError:
            return None

    async def add_result_for_branch(
        self, 
        branch_id: str, 
        component_name: str, 
        result: DataModel
    ) -> None:
        """
        Add the result for a specific component in a specific branch.
        Each branch can have only one result per component.
        
        Args:
            branch_id: The ID of the branch
            component_name: The name of the component that produced the result
            result: The result data (must inherit from DataModel)
            
        Raises:
            KeyError: If the branch doesn't exist
            ValueError: If a result already exists for this component in this branch
        """
        # Verify branch exists
        self.get_node_by_name(branch_id)  # This will raise KeyError if not found
        
        # Check if result already exists
        store_key = self._get_store_key(component_name, branch_id)
        result_data = result.model_dump()
        try:
            await self.store.add(store_key, result_data, overwrite=False)
        except KeyError:
            raise ValueError(f"Result already exists for component '{component_name}' in branch '{branch_id}'")

    async def get_result_for_branch(
        self, 
        branch_id: str, 
        component_name: str
    ) -> Optional[DataModel]:
        """
        Get the result for a specific component in a specific branch.
        Returns None if no result exists.
        
        Args:
            branch_id: The ID of the branch
            component_name: The name of the component
            
        Returns:
            The result for the component in that branch, or None if not found
            
        Raises:
            KeyError: If the branch doesn't exist
        """
        # Verify branch exists
        self.get_node_by_name(branch_id)  # This will raise KeyError if not found
        
        store_key = self._get_store_key(component_name, branch_id)
        return await self.store.get(store_key)

    async def get_result_for_component(
        self, 
        branch_id: str, 
        component_name: str,
        aggregate: bool = False
    ) -> Optional[DataModel] | list[DataModel]:
        """
        Get result for a component, searching up the branch hierarchy or aggregating across branches.
        
        Args:
            branch_id: The ID of the branch to start searching from
            component_name: The name of the component
            aggregate: If True, collects results from all branches that executed this component.
                      If False, searches up the ancestry of the given branch and returns the first result found.
            
        Returns:
            Single result (if aggregate=False) or list of results (if aggregate=True)
            
        Raises:
            KeyError: If the branch doesn't exist
        """
        if aggregate:
            return await self.get_all_results_for_component(component_name)
        
        current_branch_id = branch_id
        
        while current_branch_id is not None:
            # Check current branch for result
            result = await self.get_result_for_branch(current_branch_id, component_name)
            if result is not None:
                return result
            
            # Move to parent branch
            parent_edges = self.previous_edges(current_branch_id)
            if parent_edges:
                # Take the first parent (should only be one in a tree structure)
                current_branch_id = parent_edges[0].start
            else:
                # No more parents, reached root
                current_branch_id = None
        
        # Component not found in any ancestor branch
        return None

    async def get_all_results_for_component(self, component_name: str) -> list[DataModel]:
        """
        Get all results for a component across all branches in the run graph.
        This is useful for aggregating results when a component has been executed
        in multiple branches (e.g., collecting word counts from all text chunks).
        
        Args:
            component_name: The name of the component
            
        Returns:
            List of all results for the component across all branches
        """
        all_results = []
        
        # Iterate through all branches
        for branch_id in self._nodes:
            result = await self.get_result_for_branch(branch_id, component_name)
            if result is not None:
                all_results.append(result)
        
        return all_results

    def create_branch(
        self,
        parent_branch_id: str,
        parent_component: Optional[str] = None,
        parent_result_idx: Optional[int] = None,
    ) -> str:
        """
        Create a new branch as a child of an existing branch.
        This is a convenience method for the orchestrator.
        
        Args:
            parent_branch_id: The ID of the parent branch
            parent_component: The component that created this branch (optional)
            parent_result_idx: The result index that created this branch (optional)
            
        Returns:
            The new branch's ID
        """
        return self.add_branch(
            parent_id=parent_branch_id,
            parent_component=parent_component,
            parent_result_idx=parent_result_idx,
        )
