from core.init_llmgw import get_chat_model, get_embeddings
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore


def get_graph_agent(tools: list[dict], system_prompt: str) -> CompiledStateGraph:
    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=system_prompt,
        store=InMemoryStore(index={"embed": get_embeddings()})
    )