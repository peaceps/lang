from pathlib import Path
from langchain_classic.chains.openai_functions.openapi import openapi_spec_to_openai_fn
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_community.utilities.openapi import OpenAPISpec
from chain.core.llm_chain import ToolChain


def load_petstore_api() -> None:
    spec = OpenAPISpec.from_file(Path(__file__).parent.parent / "resources" / "swagger_petstore_min.json")
    pet_openai_functions = openapi_spec_to_openai_fn(spec)[0]
    return list(map(lambda f: convert_to_openai_tool(f), pet_openai_functions))


def run() -> None:
    model = ToolChain(tools=load_petstore_api())
    model.invoke("what are three pets names?")
    model.invoke("Tell me about the pet with id 15?")
