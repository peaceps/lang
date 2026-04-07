"""
Minimal RAG smoke test. Uses embeddings through LLMGW when llm_gateway.use_llmgw is true.

Requires the gateway to expose an OpenAI-compatible POST /v1/embeddings (same base as chat).
Optional JSON field llm_gateway.embedding_model — defaults to text-embedding-3-small.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

import httpx

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda
from openai import OpenAI

# Same SSL behavior as llm/src/llm_gateway/llm_client.py (corporate proxies)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _normalize_tool_choice(value: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    """OpenAI ``tool_choice``; expand legacy shorthand ``{\"name\": \"MyTool\"}``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and set(value.keys()) == {"name"}:
        return {"type": "function", "function": {"name": value["name"]}}
    return value


def _load_llm_gateway_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "llmgw_config.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)["llm_gateway"]


def load_env() -> dict:
    config = _load_llm_gateway_config()
    api_key = (config.get("llmgw_api_key") or os.environ.get("LLMGW_API_KEY") or "").strip()
    workspace = (config.get("llmgw_workspace") or os.environ.get("LLMGW_WORKSPACE") or "").strip()
    base = (config.get("llmgw_api_base") or os.environ.get("LLMGW_API_BASE") or "").strip().rstrip("/")
    if not api_key or not workspace or not base:
        raise ValueError(
            "LLMGW: set llmgw_api_key, llmgw_workspace, llmgw_api_base in "
            "llm/config/llmgw_config.json (or LLMGW_* env vars)."
        )
    embedding_model = (
        config.get("embedding_model")
        or os.environ.get("LLMGW_EMBEDDING_MODEL")
        or "text-embedding-3-small"
    )
    chat_model = (
        config.get("llmgw_model")
        or config.get("model")
        or os.environ.get("LLMGW_CHAT_MODEL")
        or "gpt-4o-mini"
    )
    headers = {
        "api-key": api_key,
        "workspacename": workspace,
    }
    temperature = float(config.get("temperature", 0.3))
    max_tokens = int(config.get("max_tokens", 8000))
    
    return {
        "model": embedding_model,
        "chat_model": chat_model,
        "base": base,
        "headers": headers,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


gw = load_env()


class GatewayOpenAIEmbeddings(Embeddings):
    """Same wire format as OpenAI embeddings; uses custom httpx client (api-key / workspacename headers)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        http_client: httpx.Client,
    ) -> None:
        self._model = model
        self._client = OpenAI(
            api_key="NONE",
            base_url=base_url.rstrip("/"),
            http_client=http_client,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
            encoding_format="float",
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding() -> Embeddings:
    # Avoid langchain_openai.OpenAIEmbeddings: it pulls chat_models → transformers → torch (c10.dll on your machine).
    return GatewayOpenAIEmbeddings(
        model=gw["model"],
        base_url=gw["base"],
        http_client=httpx.Client(verify=False, headers=gw["headers"])
    )


class OpenAIClient:
    """OpenAI SDK + LLMGW headers. Use ``.as_runnable()`` for ``prompt | llm`` LCEL."""

    def __init__(self) -> None:
        self.openai_client = OpenAI(
            api_key="NONE",
            base_url=gw["base"].rstrip("/"),
            http_client=httpx.Client(verify=False, headers=gw["headers"]),
        )
        self.tools: list[dict] = []
        self.tool_choice: str | dict[str, Any] | None = None

    def bind_tools(
        self,
        tools: list[dict] | None = None,
        *,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        if tools is not None:
            self.tools = tools
        if tool_choice is not None:
            self.tool_choice = tool_choice
        return self

    def invoke(self, messages: list[dict]) -> dict:
        kwargs: dict[str, Any] = {
            "model": gw["chat_model"],
            "messages": messages,
            "temperature": gw["temperature"],
            "max_tokens": gw["max_tokens"],
        }
        if self.tools:
            kwargs["tools"] = self.tools
            # Avoid sending parallel_tool_calls if the gateway's upstream rejects it.
            kwargs["parallel_tool_calls"] = False
        tc = _normalize_tool_choice(self.tool_choice)
        if tc is not None:
            kwargs["tool_choice"] = tc
        msg = self.openai_client.chat.completions.create(**kwargs).choices[0].message
        res: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_calls:
            call = msg.tool_calls[0]
            res["tool_calls"] = {
                "type": call.type,
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments),
            }
        return res

    @staticmethod
    def _raw_response_to_ai_message(raw: dict) -> AIMessage:
        content = raw.get("content") or ""
        if "tool_calls" not in raw:
            return AIMessage(content=content)
        tc = raw["tool_calls"]
        return AIMessage(
            content=content,
            tool_calls=[
                {
                    "name": tc["name"],
                    "args": tc["arguments"],
                    "id": "llmgw_tool_0",
                    "type": "tool_call",
                }
            ],
        )

    def _langchain_input_to_openai_messages(self, input_val: Any) -> list[dict]:
        if isinstance(input_val, ChatPromptValue):
            lc_messages = input_val.to_messages()
        elif isinstance(input_val, list) and (
            not input_val or hasattr(input_val[0], "type")
        ):
            lc_messages = input_val
        else:
            raise TypeError(
                "Expected ChatPromptValue or list of LangChain messages; got "
                f"{type(input_val).__name__}. Use .invoke(openai_dict_messages) for raw API dicts."
            )
        return convert_to_openai_messages(lc_messages)

    def as_runnable(self) -> RunnableLambda:
        """Return a Runnable so you can write ``ChatPromptTemplate | client.as_runnable()``."""

        def _step(input_val: Any) -> AIMessage:
            oai_messages = self._langchain_input_to_openai_messages(input_val)
            raw = self.invoke(oai_messages)
            return self._raw_response_to_ai_message(raw)

        return RunnableLambda(_step)


def get_openai_client(
    tools: list[dict] | None = None,
    *,
    tool_choice: str | dict[str, Any] | None = None,
) -> RunnableLambda:
    """LLMGW-compatible client (no langchain_openai / torch).

    ``tool_choice``: ``None`` (omit, let server default), ``\"auto\"``, ``\"none\"``,
    ``\"required\"``, full dict ``{\"type\":\"function\",\"function\":{\"name\":...}}``,
    or shorthand ``{\"name\": \"ToolName\"}``.
    """
    client = OpenAIClient()
    return client.bind_tools(
        tools=tools, tool_choice=tool_choice
    ).as_runnable()