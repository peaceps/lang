import asyncio
from uuid import uuid4
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


"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


class ReactGraphAgent:
    def __init__(
        self,
        system_prompt: str,
        tools: dict = [],
        require_confirmation_on_tool_call: bool = False,
    ):
        tools = [get_tavily_search_model(max_results=4)] + tools
        self.llm_model = get_openai_chat_model(tools)
        self.system_prompt = system_prompt
        self.tool_map = {t.name: t for t in tools}
        self._checkpoint_path = Path(__file__).resolve().parent / ".checkpoints" / "react_graph.sqlite"
        self.sync_msg_index = {}
        self.require_confirmation_on_tool_call = require_confirmation_on_tool_call
        self._clear_cache()

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
            .compile(
                checkpointer=checkpointer,
                interrupt_before=["action"] if self.require_confirmation_on_tool_call else None
            )

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

    def _invoke_sync_graph(self, text: str | None, user, action) -> None:
        self._init_sync_graph()
        for i in self._format_input(text):
            getattr(self, action)(i, user)
        self.graph.checkpointer.conn.close()

    def invoke_sync(self, text: AgentState|None, user) -> None:
        if user['configurable']['thread_id'] not in self.sync_msg_index:
            self.sync_msg_index[user['configurable']['thread_id']] = 0
        self._invoke_sync_graph(text, user, '_run_by_sync')

    def _run_by_sync(self, i: AgentState|None, user) -> None:
        res = self.graph.invoke(i, user)
        for msg in res['messages'][self.sync_msg_index[user['configurable']['thread_id']]:]:
            print(self._extract_content(msg))
        self.sync_msg_index[user['configurable']['thread_id']] = len(res['messages'])
        self._loop_actions(user, '_run_by_sync')

    def invoke_steps(self, text: AgentState|None, user) -> None:
        self._invoke_sync_graph(text, user, '_run_by_step')

    def _run_by_step(self, i: AgentState|None, user) -> None:
        stream_res = self.graph.stream(i, user)
        for chunk in stream_res:
            if 'llm' in chunk or 'action' in chunk:
                messages = chunk['llm' if 'llm' in chunk else 'action']['messages']
                print(self._extract_content(messages[-1]))
        self._loop_actions(user, '_run_by_step')

    def _loop_actions(self, user, action) -> None:
        if self.graph.get_state(user).next:
            _input = input("proceed?\n>>>")
            if _input.lower() != "y":
                print("aborting")
            else:
                getattr(self, action)(None, user)

    def invoke_stream(self, text: AgentState|None, user) -> None:
        asyncio.run(self._run_async_llm(self._format_input(text), user))

    def invoke_chat(self) -> None:
        user = {"configurable": {"thread_id": str(uuid4())}}
        asyncio.run(self._run_chat(user))
        self.shutdown()

    async def _run_chat(self, user) -> None:
        await self._init_async_graph()
        while True:
            text = input("输入你想问的内容，或者输入'exit'退出\n>>>")
            if text.lower() == "exit":
                break
            text = self._format_input(text)[0]
            await self._run_by_stream(text, user)
        await self.async_graph.checkpointer.conn.close()

    async def _run_async_llm(self, text: list[str], user) -> None:
        await self._init_async_graph()
        for i in range(len(text)):
            await self._run_by_stream(text[i], user)
        await self.async_graph.checkpointer.conn.close()

    async def _run_by_stream(self, i: AgentState|None, user) -> None:
        stream_res = self.async_graph.astream_events(i, user)
        async for chunk in stream_res:
            if chunk["event"] == "on_chat_model_stream":
                content = chunk["data"]["chunk"].content
                if content:
                    print(content, end="")
        await stream_res.aclose()
        print("\n")
        snapshot = await self.async_graph.aget_state(user)
        if snapshot.next and snapshot.next[0] == 'action' and len(snapshot.values['messages'][-1].tool_calls) > 0:
            print(self._extract_content(snapshot.values['messages'][-1]))
            _input = input("proceed?\n>>>")
            if _input.lower() != "y":
                print("aborting")
            else:
                # user = await self._small_trick(snapshot, user)
                await self._run_by_stream(None, user)

    async def _small_trick(self, snapshot, user):
        tool_call = snapshot.values['messages'][-1].tool_calls[0]
        if '杭州' in tool_call['args']['query'] and '天气' in tool_call['args']['query']:
            state_update = {
                "messages": [
                    ToolMessage(
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                        content="54 degree celcius",
                    )
                ]
            }
            user = await self.async_graph.aupdate_state(
                snapshot.config, 
                state_update,
                as_node="action"
            )
        return user

    def _format_input(self, text: str | list[str] | None) -> list[AgentState|None]:
        if not isinstance(text, list):
            text = [text]
        return [(None if i is None else {"messages": [HumanMessage(i)]}) for i in text]

    def _extract_content(self, message: AnyMessage) -> str:
        if isinstance(message, ToolMessage):
            return ""
        if message.content:
            return message.content
        if hasattr(message, 'tool_calls'):
            tool_call = message.tool_calls[0]
            return f"Will call: {tool_call['name']} with args: {tool_call['args']}"
        return ""

    def shutdown(self) -> None:
        self._clear_cache()

    def _clear_cache(self) -> None:
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
        if self._checkpoint_path.parent.exists():
            self._checkpoint_path.parent.rmdir()