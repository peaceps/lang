from typing import Any
from uuid import uuid4

from abc import ABC, abstractmethod

import aiosqlite
import asyncio
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.init_llmgw import get_openai_chat_model

from graph.graph_ui import display
from tools.tools import extract_content


class GraphAgent(ABC):
    def __init__(self, newMemory: bool = True, tools: list[Any]=[]):
        self._init_tools(tools)
        self.llm_model = get_openai_chat_model(self.tools)
        self._checkpoint_path = Path(__file__).resolve().parent / ".checkpoints" / "react_graph.sqlite"
        if newMemory:
            self._clear_cache()

    def _init_tools(self, tools: list[Any]):
        self.tools = tools

    async def _init_graph(self):
        self._ensure_checkpoint_parent()
        conn = await aiosqlite.connect(str(self._checkpoint_path))
        checkpointer = AsyncSqliteSaver(conn)
        self.graph = self._create_graph(checkpointer)

    async def _close_graph(self):
        await self.graph.checkpointer.conn.close()

    def _ensure_checkpoint_parent(self) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _clear_cache(self) -> None:
        try:
            self._checkpoint_path.unlink(missing_ok=True)
            for suf in ("-wal", "-shm"):
                Path(f"{self._checkpoint_path}{suf}").unlink(missing_ok=True)
        except OSError:
            pass

    def display_graph(self):
        asyncio.run(self._display_graph_async())

    async def _display_graph_async(self):
        await self._init_graph()
        display(self.graph)
        await self._close_graph()

    @abstractmethod
    def _create_graph(self, checkpointer=None):
        pass

    @abstractmethod
    def _format_input(self, text: str) -> list[dict[str, Any]]:
        """子类实现"""
        ...

    def invoke(self, text: str, user: RunnableConfig | dict[str, Any]=None, *args) -> None:
        if user is None:
            user = {"configurable": {"thread_id": str(uuid4())}}
        self._invoke(text, user, *args)

    def _invoke(self, text: str, user: RunnableConfig | dict[str, Any]):
        asyncio.run(self._run_async_llm(self._format_input(text), user))

    async def _run_async_llm(
        self, parts: list[dict[str, Any] | None], user: RunnableConfig | dict[str, Any]
    ) -> None:
        await self._init_graph()
        for p in parts:
            await self._run_by_stream(p, user)
        await self._close_graph()

    async def _run_by_stream(
        self, p: dict[str, Any] | None, user: RunnableConfig | dict[str, Any]
    ) -> None:
        stream_res = self.graph.astream_events(p, user)
        async for chunk in stream_res:
            if chunk["event"] == "on_chat_model_stream":
                content = chunk["data"]["chunk"].content
                if content:
                    print(content, end="")
        await stream_res.aclose()
        print("\n")
        snapshot = await self.graph.aget_state(user)
        if snapshot.next and snapshot.next[0] == 'action' and len(snapshot.values['messages'][-1].tool_calls) > 0:
            print(extract_content(snapshot.values['messages'][-1]))
            _input = input("proceed?\n>>>")
            if _input.lower() != "y":
                print("aborting")
            else:
                await self._run_by_stream(None, user)

    def shutdown(self) -> None:
        self._clear_cache()