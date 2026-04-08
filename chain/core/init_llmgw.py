"""
Minimal RAG smoke test. Uses embeddings through LLMGW when llm_gateway.use_llmgw is true.

Requires the gateway to expose an OpenAI-compatible POST /v1/embeddings (same base as chat).
Optional JSON field llm_gateway.embedding_model — defaults to text-embedding-3-small.
"""


import json
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever


def _load_llm_gateway_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "llmgw_config.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)["llm_gateway"]


def load_env() -> dict:
    config = _load_llm_gateway_config()
    return {
        "embedding_model": config.get("embedding_model", "text-embedding-3-small"),
        "model": config.get("llmgw_model", "gpt-4o-mini"),
        "base": config.get("llmgw_api_base").strip().rstrip("/"),
        "api_key": config.get("llmgw_api_key"),
        "headers": {
            "api-key": config.get("llmgw_api_key"),
            "workspacename": config.get("llmgw_workspace", None),
        },
        "timeout": config.get("timeout", 300),
        "temperature": float(config.get("temperature", 0.1)),
        "max_tokens": int(config.get("max_tokens", 8000))
    }


gw = load_env()


def get_rag_retriever(docs: list[str]) -> VectorStoreRetriever:
    embedding = OpenAIEmbeddings(
        model=gw["embedding_model"],
        api_key=gw["api_key"],
        base_url=gw["base"].rstrip("/"),
        default_headers=gw["headers"],
    )
    return InMemoryVectorStore.from_texts(docs, embedding=embedding).as_retriever()


def get_openai_client(
    tools: list[dict] | None = None,
    *,
    tool_choice: str | dict[str, Any] | None = None,
) -> RunnableLambda:
    chat = ChatOpenAI(
        model=gw["model"],
        api_key=gw["api_key"],
        base_url=gw["base"].rstrip("/"),
        default_headers=gw["headers"],
        temperature=gw["temperature"],
        max_tokens=gw["max_tokens"],
        timeout=gw["timeout"]
    )
    if tools:
        chat = chat.bind_tools(tools=tools, tool_choice=None if tool_choice is None else {"name": tool_choice})
    return chat