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
from unittest.mock import MagicMock, patch

import neo4j
import pytest
from neo4j.exceptions import CypherSyntaxError, Neo4jError
from neo4j_graphrag.exceptions import (
    RetrieverInitializationError,
    SchemaFetchError,
    SearchValidationError,
    Text2CypherRetrievalError,
)
from neo4j_graphrag.generation.prompts import Text2CypherTemplate
from neo4j_graphrag.llm import LLMResponse
from neo4j_graphrag.retrievers.text2cypher import Text2CypherRetriever, extract_cypher
from neo4j_graphrag.types import RetrieverResult, RetrieverResultItem


def test_t2c_retriever_initialization(driver: MagicMock, llm: MagicMock) -> None:
    with patch("neo4j_graphrag.retrievers.base.get_version") as mock_get_version:
        mock_get_version.return_value = ((5, 23, 0), False, False)
        Text2CypherRetriever(driver, llm, neo4j_schema="dummy-text")
        mock_get_version.assert_called_once()


@patch("neo4j_graphrag.retrievers.base.get_version")
@patch("neo4j_graphrag.retrievers.text2cypher.get_schema")
def test_t2c_retriever_schema_retrieval(
    get_schema_mock: MagicMock,
    mock_get_version: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    Text2CypherRetriever(driver, llm)
    get_schema_mock.assert_called_once()


@patch("neo4j_graphrag.retrievers.base.get_version")
@patch("neo4j_graphrag.retrievers.text2cypher.get_schema")
def test_t2c_retriever_schema_retrieval_failure(
    get_schema_mock: MagicMock,
    mock_get_version: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    get_schema_mock.side_effect = Neo4jError
    with pytest.raises(SchemaFetchError):
        Text2CypherRetriever(driver, llm)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_invalid_neo4j_schema(
    mock_get_version: MagicMock, driver: MagicMock, llm: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    with pytest.raises(RetrieverInitializationError) as exc_info:
        Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema=42,  # type: ignore[arg-type, unused-ignore]
        )

    assert "neo4j_schema" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_invalid_search_query(
    mock_get_version: MagicMock, driver: MagicMock, llm: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    with pytest.raises(SearchValidationError) as exc_info:
        retriever = Text2CypherRetriever(
            driver=driver, llm=llm, neo4j_schema="dummy-text"
        )
        retriever.search(query_text=42)

    assert "query_text" in str(exc_info.value)
    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_invalid_search_examples(
    mock_get_version: MagicMock, driver: MagicMock, llm: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    with pytest.raises(RetrieverInitializationError) as exc_info:
        Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema="dummy-text",
            examples=42,  # type: ignore[arg-type, unused-ignore]
        )

    assert "examples" in str(exc_info.value)
    assert "Initialization failed" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_happy_path(
    mock_get_version: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    t2c_query = "MATCH (n) RETURN n;"
    query_text = "may thy knife chip and shatter"
    neo4j_schema = "dummy-schema"
    examples = ["example-1", "example-2"]
    neo4j_database = "mydb"
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=neo4j_schema,
        examples=examples,
        neo4j_database=neo4j_database,
    )
    llm.invoke.return_value = LLMResponse(content=t2c_query)
    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    template = Text2CypherTemplate()
    prompt = template.format(
        schema=neo4j_schema,
        examples="\n".join(examples),
        query_text=query_text,
    )
    retriever.search(query_text=query_text)
    llm.invoke.assert_called_once_with(prompt)
    driver.execute_query.assert_called_once_with(
        query_=t2c_query,
        database_=neo4j_database,
        routing_=neo4j.RoutingControl.READ,
    )


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_cypher_error(
    mock_get_version: MagicMock, driver: MagicMock, llm: MagicMock
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    t2c_query = "this is not a cypher query"
    neo4j_schema = "dummy-schema"
    examples = ["example-1", "example-2"]
    retriever = Text2CypherRetriever(
        driver=driver, llm=llm, neo4j_schema=neo4j_schema, examples=examples
    )
    retriever.llm.invoke.return_value = LLMResponse(content=t2c_query)
    query_text = "may thy knife chip and shatter"
    driver.execute_query.side_effect = CypherSyntaxError
    with pytest.raises(Text2CypherRetrievalError) as e:
        retriever.search(query_text=query_text)
    assert "Failed to get search result" in str(e)


@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_with_result_format_function(
    mock_get_version: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
    result_formatter: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    retriever = Text2CypherRetriever(
        driver=driver, llm=llm, result_formatter=result_formatter
    )
    t2c_query = "MATCH (n) RETURN n;"
    retriever.llm.invoke.return_value = LLMResponse(content=t2c_query)
    query_text = "may thy knife chip and shatter"
    driver.execute_query.return_value = [
        [neo4j_record],
        None,
        None,
    ]

    records = retriever.search(query_text=query_text)

    assert records == RetrieverResult(
        items=[
            RetrieverResultItem(
                content="dummy-node", metadata={"score": 1.0, "node_id": 123}
            ),
        ],
        metadata={"cypher": t2c_query, "__retriever": "Text2CypherRetriever"},
    )


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_initialization_with_custom_prompt(
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    prompt = "This is a custom prompt. {query_text}"
    retriever = Text2CypherRetriever(driver=driver, llm=llm, custom_prompt=prompt)
    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    retriever.search(query_text="test")

    llm.invoke.assert_called_once_with("This is a custom prompt. test")


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_initialization_with_custom_prompt_and_schema_and_examples(
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    prompt = "This is a custom prompt. {query_text}"
    neo4j_schema = "dummy-schema"
    examples = ["example-1", "example-2"]

    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        custom_prompt=prompt,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    retriever.search(query_text="test")

    llm.invoke.assert_called_once_with("This is a custom prompt. test")


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_initialization_with_custom_prompt_and_schema_and_examples_for_prompt_params(
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    prompt = "This is a custom prompt. {query_text} {schema} {examples}"
    neo4j_schema = "dummy-schema"
    examples = ["example-1", "example-2"]

    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        custom_prompt=prompt,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    retriever.search(query_text="test")

    llm.invoke.assert_called_once_with(
        "This is a custom prompt. test dummy-schema example-1\nexample-2"
    )


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_initialization_with_custom_prompt_and_unused_schema_and_examples(
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    prompt = "This is a custom prompt. {query_text} {schema} {examples}"
    neo4j_schema = "dummy-schema"
    examples = ["example-1", "example-2"]

    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        custom_prompt=prompt,
        neo4j_schema=neo4j_schema,
        examples=examples,
    )

    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    retriever.search(
        query_text="test",
        prompt_params={"schema": "another-dummy-schema", "examples": "another-example"},
    )

    llm.invoke.assert_called_once_with(
        "This is a custom prompt. test another-dummy-schema another-example"
    )


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_invalid_custom_prompt_type(
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    with pytest.raises(RetrieverInitializationError) as exc_info:
        Text2CypherRetriever(
            driver=driver,
            llm=llm,
            custom_prompt=42,  # type: ignore[arg-type, unused-ignore]
        )

    assert "Input should be a valid string" in str(exc_info.value)


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_with_custom_prompt_prompt_params(
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    prompt = "This is a custom prompt. {query_text} {examples_custom}"
    query = "test"
    examples = ["example A", "example B"]

    retriever = Text2CypherRetriever(driver=driver, llm=llm, custom_prompt=prompt)
    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    retriever.search(query_text=query, prompt_params={"examples_custom": examples})

    llm.invoke.assert_called_once_with(
        """This is a custom prompt. test ['example A', 'example B']"""
    )


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
def test_t2c_retriever_with_custom_prompt_bad_prompt_params(
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    prompt = "This is a custom prompt. {query_text} {examples}"
    query = "test"
    examples = ["example A", "example B"]

    retriever = Text2CypherRetriever(driver=driver, llm=llm, custom_prompt=prompt)
    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )
    retriever.search(
        query_text=query,
        prompt_params={
            "examples": examples,
            "bad_param": "this should not be present in template.",
        },
    )

    llm.invoke.assert_called_once_with(
        """This is a custom prompt. test ['example A', 'example B']"""
    )


@patch("neo4j_graphrag.retrievers.text2cypher.extract_cypher")
@patch("neo4j_graphrag.retrievers.base.get_version")
@patch("neo4j_graphrag.retrievers.text2cypher.get_schema")
def test_t2c_retriever_with_custom_prompt_and_schema(
    get_schema_mock: MagicMock,
    mock_get_version: MagicMock,
    mock_extract_cypher: MagicMock,
    driver: MagicMock,
    llm: MagicMock,
    neo4j_record: MagicMock,
) -> None:
    mock_get_version.return_value = ((5, 23, 0), False, False)
    prompt = "This is a custom prompt. {query_text} {schema}"
    query = "test"

    driver.execute_query.return_value = (
        [neo4j_record],
        None,
        None,
    )

    retriever = Text2CypherRetriever(driver=driver, llm=llm, custom_prompt=prompt)
    retriever.search(
        query_text=query,
        prompt_params={},
    )

    get_schema_mock.assert_not_called()
    llm.invoke.assert_called_once_with("""This is a custom prompt. test """)


@pytest.mark.parametrize(
    "description, cypher_query, expected_output",
    [
        ("No changes", "MATCH (n) RETURN n;", "MATCH (n) RETURN n;"),
        (
            "Surrounded by backticks",
            "Cypher query: ```MATCH (n) RETURN n;```",
            "MATCH (n) RETURN n;",
        ),
        (
            "Spaces in label",
            "Cypher query: ```MATCH (n: Label With Spaces ) RETURN n;```",
            "MATCH (n:`Label With Spaces`) RETURN n;",
        ),
        (
            "No spaces in label",
            "Cypher query: ```MATCH (n: LabelWithNoSpaces ) RETURN n;```",
            "MATCH (n: LabelWithNoSpaces ) RETURN n;",
        ),
        (
            "Backticks in label",
            "Cypher query: ```MATCH (n: `LabelWithBackticks` ) RETURN n;```",
            "MATCH (n: `LabelWithBackticks` ) RETURN n;",
        ),
        (
            "Spaces in property key",
            "Cypher query: ```MATCH (n: { prop 1: 1, prop 2: 2 }) RETURN n;```",
            "MATCH (n: { `prop 1`: 1, `prop 2`: 2 }) RETURN n;",
        ),
        (
            "No spaces in property key",
            "Cypher query: ```MATCH (n: { prop1: 1, prop2: 2 }) RETURN n;```",
            "MATCH (n: { prop1: 1, prop2: 2 }) RETURN n;",
        ),
        (
            "Backticks in property key",
            "Cypher query: ```MATCH (n: { `prop 1`: 1, `prop 2`: 2 }) RETURN n;```",
            "MATCH (n: { `prop 1`: 1, `prop 2`: 2 }) RETURN n;",
        ),
        (
            "Spaces in relationship type",
            "Cypher query: ```MATCH (n)-[: Relationship With Spaces ]->(m) RETURN n, m;```",
            "MATCH (n)-[:`Relationship With Spaces`]->(m) RETURN n, m;",
        ),
        (
            "No spaces in relationship type",
            "Cypher query: ```MATCH (n)-[ : RelationshipWithNoSpaces ]->(m) RETURN n, m;```",
            "MATCH (n)-[ : RelationshipWithNoSpaces ]->(m) RETURN n, m;",
        ),
        (
            "Backticks in relationship type",
            "Cypher query: ```MATCH (n)-[ : `RelationshipWithBackticks` ]->(m) RETURN n, m;```",
            "MATCH (n)-[ : `RelationshipWithBackticks` ]->(m) RETURN n, m;",
        ),
    ],
)
def test_extract_cypher(
    description: str, cypher_query: str, expected_output: str
) -> None:
    assert (
        extract_cypher(cypher_query) == expected_output
    ), f"Failed test case: {description}"
