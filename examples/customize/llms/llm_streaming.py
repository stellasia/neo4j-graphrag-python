import asyncio

from neo4j_graphrag.llm import OpenAILLM, VertexAILLM

# set api key here on in the OPENAI_API_KEY env var
api_key = None

async def main():

    # llm = OpenAILLM(model_name="gpt-4o", api_key=api_key)
    llm = VertexAILLM(model_name="gemini-2.0-flash-001")

    async for res in llm.astream("tell me a futurist story"):
        print(res.content, end="\n\n\n --- \n\n\n")
    # for res in llm.stream("say something"):
    #     print(res.content, end="")
    print()


if __name__ == '__main__':
    asyncio.run(main())
