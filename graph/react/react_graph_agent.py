import operator
import asyncio
import sqlite3
import aiosqlite
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from core.init_llmgw import get_openai_chat_model


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class ReactGraphAgent:
    def __init__(self, system_prompt: str, tools: dict):
        self.llm_model = get_openai_chat_model(tools)
        self.system_prompt = system_prompt
        self.tool_map = {t.name: t for t in tools}

    def _init_sync_graph(self):
        if not hasattr(self, 'graph'):
            conn = sqlite3.connect(":memory:")
            checkpointer = SqliteSaver(conn)
            self.graph = self._init_graph(checkpointer)

    async def _init_async_graph(self):
        if not hasattr(self, 'async_graph'):
            conn = await aiosqlite.connect(":memory:")
            checkpointer = AsyncSqliteSaver(conn)
            self.async_graph = self._init_graph(checkpointer)

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

    def invoke_stream(self, input: str | list[str], config) -> None:
        self._init_sync_graph()
        input = input if isinstance(input, list) else [input]
        for i in input:
            stream_res = self.graph.stream({"messages": [HumanMessage(i)]}, config)
            res = ""
            for chunk in stream_res:
                messages = chunk['llm' if 'llm' in chunk else 'action']['messages']
                print(messages)
                res += messages[-1].content
            print("\n")
            print(res)

    def invoke_async(self, input: str | list[str], config) -> None:
        input = input if isinstance(input, list) else [input]
        asyncio.run(self.run_async_llm(input, config))

    async def run_async_llm(self, input: list[str], config) -> None:
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