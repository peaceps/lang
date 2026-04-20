import asyncio
import operator
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

import aiosqlite
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from core.init_llmgw import get_tavily_search_model
from core.init_llmgw import get_openai_chat_model


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class ReactGraphAgent:
    def __init__(
        self,
        system_prompt: str,
        tools: dict = []
    ):
        tools = [get_tavily_search_model(max_results=4)] + tools
        self.llm_model = get_openai_chat_model(tools)
        self.system_prompt = system_prompt
        self.tool_map = {t.name: t for t in tools}
        self._checkpoint_path = Path(__file__).resolve().parent / ".checkpoints" / "react_graph.sqlite"

    def _init_sync_graph(self):
        self._ensure_checkpoint_parent()
        conn = sqlite3.connect(
            str(self._checkpoint_path),
            check_same_thread=False,
        )
        checkpointer = SqliteSaver(conn)
        self.graph = self._init_graph(checkpointer)

    async def _init_async_graph(self):
        self._ensure_checkpoint_parent()
        conn = await aiosqlite.connect(str(self._checkpoint_path))
        checkpointer = AsyncSqliteSaver(conn)
        self.async_graph = self._init_graph(checkpointer)

    def _ensure_checkpoint_parent(self) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_graph(self, checkpointer):
        return StateGraph(AgentState)\
            .add_node("llm", self._call_llm)\
            .add_node("action", self._take_action)\
            .add_conditional_edges("llm", self._check_tool_calls, {True: "action", False: END})\
            .add_edge("action", "llm")\
            .set_entry_point("llm")\
            .compile(checkpointer=checkpointer)

    def _call_llm(self, state: AgentState) -> AgentState:
        messages = [SystemMessage(self.system_prompt)] + state["messages"]
        res = self.llm_model.invoke(messages)
        return {"messages": [res]}

    def _take_action(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            result = self.tool_map[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

    def _check_tool_calls(self, state: AgentState) -> bool:
        tool_calls = state["messages"][-1].tool_calls
        return len(tool_calls) > 0

    def invoke_sync(self, input: str | list[str], config) -> None:
        self._init_sync_graph()
        input = input if isinstance(input, list) else [input]
        for i in input:
            res = self.graph.invoke({"messages": [HumanMessage(i)]}, config)
            print(res['messages'][-1].content)
        self.graph.checkpointer.conn.close()

    def invoke_steps(self, input: str | list[str], config) -> None:
        self._init_sync_graph()
        input = input if isinstance(input, list) else [input]
        for i in input:
            stream_res = self.graph.stream({"messages": [HumanMessage(i)]}, config)
            res = ""
            for chunk in stream_res:
                messages = chunk['llm' if 'llm' in chunk else 'action']['messages']
                content = messages[-1].content
                print(content)
                res += messages[-1].content
            print("\n")
            # print(res)
        self.graph.checkpointer.conn.close()

    def invoke_stream(self, input: str | list[str], config) -> None:
        input = input if isinstance(input, list) else [input]
        asyncio.run(self._run_async_llm(input, config))

    def shutdown(self) -> None:
        self._checkpoint_path.unlink(missing_ok=True)
        self._checkpoint_path.parent.rmdir()

    async def _run_async_llm(self, input: list[str], config) -> None:
        await self._init_async_graph()
        for i in range(len(input)):
            stream_res = self.async_graph.astream_events({"messages": [HumanMessage(input[i])]}, config)
            async for chunk in stream_res:
                if chunk["event"] == "on_chat_model_stream":
                    content = chunk["data"]["chunk"].content
                    if content:
                        # Empty content in the context of OpenAI means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        print(content, end="")
            await stream_res.aclose()
            print("\n")
        await self.async_graph.checkpointer.conn.close()