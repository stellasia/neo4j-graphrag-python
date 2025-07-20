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
import logging

from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
from pydantic import BaseModel
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
    """Executor that off-loads components to Ray *Actors*.

    Every component class gets its own long-lived actor which instantiates the
    heavy resources (Neo4j driver, GPU model, …) inside the worker process, so
    nothing un-picklable travels over the network.
    """

    def __init__(self, address: str | None = "auto") -> None:  # noqa: D401
        try:
            import ray

            # if not ray.is_initialized():
            #     ray.init(address=address, namespace="graphrag")

            self._ray = ray

            # Remote Actor definition
            @ray.remote  # type: ignore[misc]
            class _ComponentActor:  # noqa: WPS430
                def __init__(self, module: str | None, cls_name: str | None, init_kwargs: dict[str, Any]):
                    import importlib
                    from pydantic import BaseModel
                    import cloudpickle

                    if module is None and "pickled" in init_kwargs:
                        self._component = cloudpickle.loads(init_kwargs["pickled"])
                    else:
                        CompCls = getattr(importlib.import_module(module), cls_name)  # type: ignore[arg-type]
                        self._component = CompCls(**init_kwargs)

                async def run(self, ctx: dict[str, Any], inputs: dict[str, Any]):  # type: ignore[require-type-hints]  # noqa: E501
                    from neo4j_graphrag.experimental.pipeline.types.context import RunContext

                    context = RunContext(**ctx)
                    res = await self._component.run_with_context(context, **inputs)
                    return res.model_dump() if isinstance(res, BaseModel) else res

            self._ComponentActor = _ComponentActor
            self._actors: dict[str, Any] = {}
            self._ray_available = True
        except ModuleNotFoundError:  # pragma: no cover
            self._ray_available = False

    async def _get_actor(self, task: "TaskPipelineNode") -> Any:  # noqa: ANN401
        key = task.component.__class__.__qualname__
        if key not in self._actors:
            CompCls = task.component.__class__
            init_kwargs: dict[str, Any] | None = None
            if hasattr(task.component, "model_dump"):
                try:
                    init_kwargs = task.component.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    init_kwargs = None

            if init_kwargs is not None:
                self._actors[key] = self._ComponentActor.remote(
                    CompCls.__module__, CompCls.__name__, init_kwargs
                )
            else:
                # Fallback: send pickled instance
                import cloudpickle

                comp_bytes = cloudpickle.dumps(task.component)
                self._actors[key] = self._ComponentActor.remote(
                    None, None, {"pickled": comp_bytes}
                )
        return self._actors[key]

    async def submit(
        self,
        task: "TaskPipelineNode",
        context: RunContext,
        inputs: dict[str, Any],
    ) -> RunResult | None:
        if not self._ray_available:
            return await super().submit(task, context, inputs)

        ctx_dict = {"run_id": context.run_id, "task_name": context.task_name, "notifier": None}

        actor = await self._get_actor(task)

        try:
            obj_ref = actor.run.remote(ctx_dict, inputs)
            res_dict = self._ray.get(obj_ref)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "Remote execution failed for %s, fallback to LocalExecutor: %s", task.name, exc
            )
            return await super().submit(task, context, inputs)

        return RunResult(result=res_dict) if res_dict is not None else None

# Public re-exports
__all__ = [
    "ExecutorProtocol",
    "LocalExecutor",
    "RayExecutor",
]
