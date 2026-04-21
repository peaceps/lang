"""
Minimal RAG smoke test. Uses embeddings through LLMGW when llm_gateway.use_llmgw is true.

Requires the gateway to expose an OpenAI-compatible POST /v1/embeddings (same base as chat).
Optional JSON field llm_gateway.embedding_model — defaults to text-embedding-3-small.
"""

import warnings

# 部分 OpenAI 兼容网关把 choice.logprobs 写成字符串 "null"，openai SDK 反序列化后触发 pydantic 序列化 UserWarning
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic.main",
)

import json
from pathlib import Path
from typing import Any

from tavily import TavilyClient
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain.chat_models import BaseChatModel, init_chat_model
from langmem import create_multi_prompt_optimizer
from langchain_tavily import TavilySearch


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "llmgw_config.json"


def _load_llm_gateway_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)["llm_gateway"]


def load_env() -> dict:
    config = _load_llm_gateway_config()
    headers = {"api-key": config.get("llmgw_api_key")}
    if config.get("llmgw_workspace"):
        headers["workspacename"] = config.get("llmgw_workspace")
    return {
        "embedding_model": config.get("embedding_model", "text-embedding-3-small"),
        "model": config.get("llmgw_model", "gpt-4o-mini"),
        "base": config.get("llmgw_api_base").strip().rstrip("/"),
        "api_key": config.get("llmgw_api_key"),
        "headers": headers,
        "timeout": config.get("timeout", 300),
        "temperature": float(config.get("temperature", 0.1)),
        "max_tokens": int(config.get("max_tokens", 8000)),
        "tavily_api_key": config.get("tavily_api_key")
    }


gw = load_env()


def get_embeddings() -> Embeddings:
    return OpenAIEmbeddings(
        model=gw["embedding_model"],
        api_key=gw["api_key"],
        base_url=gw["base"].rstrip("/"),
        default_headers=gw["headers"],
        # DashScope 等兼容接口只接受字符串 input；默认 True 会发 token id 列表导致 400
        check_embedding_ctx_length=False,
    )


def get_rag_retriever(docs: list[str]) -> VectorStoreRetriever:
    embedding = get_embeddings()
    return InMemoryVectorStore.from_texts(docs, embedding=embedding).as_retriever()


def get_openai_chat_model(
    tools: list[dict] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> ChatOpenAI:
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
        chat = chat.bind_tools(
            tools=tools,
            tool_choice=tool_choice,
        )
    return chat


def get_chat_model(provider: str = "openai") -> BaseChatModel:
    return init_chat_model(
        model_provider=provider,
        model=gw["model"],
        api_key=gw["api_key"],
        base_url=gw["base"].rstrip("/"),
        default_headers=gw["headers"],
        temperature=gw["temperature"],
        max_tokens=gw["max_tokens"],
        timeout=gw["timeout"]
    )


def get_multi_prompt_optimizer() -> Runnable:
    return create_multi_prompt_optimizer(
        get_chat_model(),
        kind="prompt_memory",
    )


def get_tavily_search_model(max_results: int = 3) -> TavilySearch:
    return TavilySearch(max_results=max_results, tavily_api_key=gw["tavily_api_key"])

def get_tavily_client() -> TavilyClient:
    return TavilyClient(api_key=gw["tavily_api_key"])