import asyncio
import logging
from typing import Any, Type

import json
import jsonref
from pydantic import BaseModel
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    Part,
)

logging.basicConfig()
logging.getLogger("neo4j_graphrag").setLevel(logging.INFO)
logging.getLogger("examples").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


graph_schema = {
  "node_types": [
    {
      "label": "Person",
      "description": "Represents a character or individual.",
      "properties": [
        {
          "name": "name",
          "type": "STRING",
          "description": "The full name of the person."
        },
        {
          "name": "title",
          "type": "STRING",
          "description": "The title held by the person, e.g., 'Duke', 'Lady'."
        }
      ]
    },
    {
      "label": "House",
      "description": "Represents an aristocratic or ruling family/organization.",
      "properties": [
        {
          "name": "name",
          "type": "STRING",
          "description": "The name of the house."
        }
      ]
    },
    {
      "label": "Planet",
      "description": "Represents a celestial body.",
      "properties": [
        {
          "name": "name",
          "type": "STRING",
          "description": "The name of the planet."
        },
        {
          "name": "climate",
          "type": "STRING",
          "description": "A description of the planet's climate."
        }
      ]
    }
  ],
  "relationship_types": [
    {
      "label": "IS_SON_OF",
      "description": "Indicates a filial relationship where one person is the son of another.",
      "properties": []
    },
    {
      "label": "IS_HEIR_OF",
      "description": "Indicates a person is the designated successor to a house or entity.",
      "properties": []
    },
    {
      "label": "RULES",
      "description": "Indicates a house or entity exerts control over a planet or territory.",
      "properties": [
        {
          "name": "since_year",
          "type": "INTEGER",
          "description": "The year from which the rule has been established."
        }
      ]
    },
    {
      "label": "MEMBER_OF",
      "description": "Indicates a person belongs to a specific house.",
      "properties": []
    }
  ],
  "patterns": [
    [
      "Person",
      "IS_SON_OF",
      "Person"
    ],
    [
      "Person",
      "IS_HEIR_OF",
      "House"
    ],
    [
      "House",
      "RULES",
      "Planet"
    ],
    [
      "Person",
      "MEMBER_OF",
      "House"
    ]
  ]
}


def schema_from_pydantic(model: Type[BaseModel]) -> dict[str, Any]:
    """This function does not work out of the box for VertexAI,
    see the next one."""
    return jsonref.replace_refs(model.model_json_schema())


def get_response_schema_for_vertexai(fn):
    """Reading a manually (to gain time on testing) edited
    file generated from the previous function. Basically, remove "nul" and
    the type of items in the 'patterns' tuple.
    """
    with open(fn) as f:
        response_schema = json.load(f)
    return response_schema


def get_vertexai_response(prompt, schema_model_or_file, model_name="gemini-2.5-flash-preview-05-20"):
    # response_schema = schema_from_pydantic(schema_model_or_file)  # GraphSchema
    # use saved files instead, which have slight modifications compared to
    # the output of the above function to make it work with VertexAI
    response_schema = get_response_schema_for_vertexai(schema_model_or_file)
    generation_config = GenerationConfig(
        # temperature=0.0,
        # max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    model = GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        # system_instruction=system_message,
    )
    result = model.generate_content([
        Content(role="user", parts=[Part.from_text(prompt)])
    ])
    return result.text


TEXT = """The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House Atreides,
an aristocratic family that rules the planet Caladan, the rainy planet, since 10191."""


async def main_schema_extraction() -> None:
    """
    """
    prompt_template="""Build a schema including node and relationship types
    in order to build a clean and easily navigable labeled property graph
    to represent the following text.

    Text:
    {text}
    """
    prompt = prompt_template.format(text=TEXT)
    response = get_vertexai_response(prompt, schema_model_or_file="examples/json_schema_of_graph_schema_for_vertexai.json")
    print("=" * 50)
    print("SCHEMA EXTRACTION FROM TEXT")
    print("TEXT=", TEXT)
    print("SCHEMA:")
    print(response)
    print("=" * 50)


async def main_entity_extraction() -> None:
    prompt_template = """Build a labeled property graph
        to represent the following text.

        Use the following graph schema:
        {schema}

        Text:
        {text}
    """
    prompt = prompt_template.format(
        text=TEXT,
        schema=graph_schema,
    )
    response = get_vertexai_response(prompt, schema_model_or_file="examples/json_schema_of_entities_relations_for_vertexai.json")
    print("=" * 50)
    print("ENTITY EXTRACTION FROM TEXT")
    print("TEXT=", TEXT)
    print("GRAPH:")
    print(response)
    print("=" * 50)



if __name__ == "__main__":
    # asyncio.run(main_schema_extraction())
    asyncio.run(main_entity_extraction())
