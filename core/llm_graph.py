from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from core.init_llmgw import get_chat_model, get_embeddings


def get_graph_agent(
    tools: list[Any],
    system_prompt: str | None = None,
    *,
    middleware: Sequence[AgentMiddleware[Any, Any]] = (),
    state_schema: type[AgentState[Any]] | None = None,
) -> CompiledStateGraph:
    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware,
        state_schema=state_schema,
        store=InMemoryStore(index={"embed": get_embeddings()}),
    )
