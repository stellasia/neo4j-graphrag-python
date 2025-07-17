"""
Executors for Pipeline tasks.

This module introduces a thin abstraction layer that lets the Orchestrator
submit a task to *some* execution backend (local asyncio, Ray, Celery, …).
For a first spike we provide:

• LocalExecutor – current behaviour (runs the task coroutine in-process).
• RayExecutor     – submits the task to a Ray cluster if the `ray` package is
                    available; otherwise it transparently falls back to local
                    execution so as not to introduce a hard dependency.

The interface purposefully mirrors the place in Orchestrator where we formerly
called `await task.run(...)`.  No other change is required to the DAG engine.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

from neo4j_graphrag.experimental.pipeline.types.context import RunContext
from neo4j_graphrag.experimental.pipeline.types.orchestration import RunResult

# Avoid circular import at runtime.  Only import for type-checking.
if TYPE_CHECKING:  # pragma: no cover
    from neo4j_graphrag.experimental.pipeline.pipeline import TaskPipelineNode


@runtime_checkable
class ExecutorProtocol(Protocol):
    """Minimal interface an executor must expose."""

    async def submit(
        self,
        task: TaskPipelineNode,
        context: RunContext,
        inputs: dict[str, Any],
    ) -> RunResult | None:  # noqa: D401 (simple type annotation)
        ...


class LocalExecutor:
    """Run the component coroutine in the current event-loop (default)."""

    async def submit(
        self,
        task: TaskPipelineNode,
        context: RunContext,
        inputs: dict[str, Any],
    ) -> RunResult | None:
        return await task.run(context, inputs)


# --- RayExecutor ------------------------------------------------------------
# We define *one* RayExecutor class.  If the ray package is absent we fall back
# to LocalExecutor semantics but keep the API so that importers need not branch.

class RayExecutor(LocalExecutor):
    """Executor that tries to off-load the work to a Ray cluster.

    When *ray* is not installed or cannot be initialised we gracefully degrade
    to the behaviour of :class:`LocalExecutor`.
    """

    def __init__(self, address: str | None = "auto") -> None:  # noqa: D401
        # Attempt to import ray lazily; set flag for later use.
        try:
            import ray

            # if not ray.is_initialized():
            #     ray.init(address=address, namespace="graphrag")

            # Define the remote runner lazily inside the initialisation so that
            # it exists only when ray is available.

            @ray.remote  # type: ignore[misc]
            def _run_component_remote(task_bytes: bytes, inputs: dict[str, Any]) -> Any:  # noqa: WPS430
                """Execute the component in the Ray worker."""

                import cloudpickle
                import asyncio
                from neo4j_graphrag.experimental.pipeline.types.context import RunContext

                task = cloudpickle.loads(task_bytes)

                dummy_ctx = RunContext(
                    run_id="remote-" + task.name,
                    task_name=task.name,
                    notifier=None,
                )

                async def _run() -> Any:  # noqa: WPS430
                    return await task.component.run_with_context(dummy_ctx, **inputs)

                return asyncio.run(_run())

            self._ray = ray
            self._remote_runner = _run_component_remote
            self._ray_available = True

        except ModuleNotFoundError:  # pragma: no cover
            # Ray not installed – fall back to local execution
            self._ray_available = False

    async def submit(
        self,
        task: "TaskPipelineNode",
        context: RunContext,
        inputs: dict[str, Any],
    ) -> RunResult | None:
        if not self._ray_available:
            return await super().submit(task, context, inputs)

        import cloudpickle

        try:
            task_bytes = cloudpickle.dumps(task)
        except Exception as exc:  # noqa: BLE001
            # Component (e.g., holding Neo4j driver) is not picklable – run locally.
            # This keeps the pipeline working even if some tasks cannot be off-loaded.
            import logging

            logging.getLogger(__name__).warning(
                "RayExecutor fallback to LocalExecutor for %s: %s", task.name, exc
            )
            return await super().submit(task, context, inputs)

        obj_ref = self._remote_runner.remote(task_bytes, inputs)
        res = self._ray.get(obj_ref)
        if res is None:
            return None
        return RunResult(result=res)

# Public re-exports
__all__ = [
    "ExecutorProtocol",
    "LocalExecutor",
    "RayExecutor",
] 