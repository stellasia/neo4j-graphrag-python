from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner

if __name__ == "__main__":

    import asyncio
    import os

    os.environ["NEO4J_URI"] = "neo4j+s://demo.neo4jlabs.com"
    os.environ["NEO4J_USER"] = "recommendations"
    os.environ["NEO4J_PASSWORD"] = "recommendations"

    runner = PipelineRunner.from_config_file(
        "examples/question_answering/simple_rag_pipeline_config.json"
    )
    print(
        asyncio.run(
            runner.run(dict(
                query_text="show me a movie about cats",
                retriever_config={
                    "top_k": 2,
                },
                return_context=True,
            ))
        )
    )
