"""Benchmark real KG pipeline with LocalExecutor vs RayExecutor.

Usage examples:

    # Local benchmark on 5 identical texts
    python examples/benchmark_real_pipeline.py --executor local --docs 5 \
        --config examples/build_graph/from_config_files/simple_kg_pipeline_config.yaml

    # Ray benchmark (requires ray cluster running)
    python examples/benchmark_real_pipeline.py --executor ray --docs 5 \
        --config examples/build_graph/from_config_files/simple_kg_pipeline_config.yaml

Environment variables required by the sample config:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY

The script feeds the same text to the pipeline N times (or reads one text file
if provided) and reports total wall-clock time.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import List

from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner
from neo4j_graphrag.experimental.pipeline.executors import LocalExecutor, RayExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_TEXT = (
    "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of "
    "House Atreides, an aristocratic family that rules the planet Caladan."
)


async def build_runner(config_path: Path, executor_name: str) -> PipelineRunner:
    runner = PipelineRunner.from_config_file(config_path)
    # Override executor
    if executor_name == "ray":
        runner.pipeline.executor = RayExecutor(address="auto")
    else:
        runner.pipeline.executor = LocalExecutor()
    return runner


async def run_docs(runner: PipelineRunner, texts: List[str] | None, pdfs: List[Path] | None) -> None:
    if texts:
        for text in texts:
            await runner.run({"text": text})
    elif pdfs:
        for p in pdfs:
            await runner.run({"file_path": str(p)})


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark real KG pipeline")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML/JSON pipeline config file")
    parser.add_argument("--executor", choices=["local", "ray"], default="local")
    parser.add_argument("--docs", type=int, default=3, help="Number of documents to ingest")
    parser.add_argument("--text-file", type=Path, help="Optional text file to use as document content")
    parser.add_argument("--pdf-dir", type=Path, help="Directory containing PDF files to ingest")

    args = parser.parse_args()

    if args.pdf_dir:
        pdf_paths = sorted(list(args.pdf_dir.glob("*.pdf")))[: args.docs]
        if not pdf_paths:
            raise SystemExit("No PDF files found in directory")
        texts = None
    elif args.text_file:
        text_content = args.text_file.read_text()
        texts = [text_content for _ in range(args.docs)]
        pdf_paths = None
    else:
        text_content = DEFAULT_TEXT
        texts = [text_content for _ in range(args.docs)]
        pdf_paths = None

    runner = await build_runner(args.config, args.executor)

    start = time.perf_counter()
    await run_docs(runner, texts, pdf_paths)
    duration = time.perf_counter() - start

    print(
        f"Processed {args.docs} docs using {args.executor} executor in {duration:.2f}s "
        f"(avg {duration/args.docs:.3f}s/doc)"
    )

    # Graceful shutdown (close drivers etc.)
    await runner.close()


if __name__ == "__main__":
    asyncio.run(main()) 