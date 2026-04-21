import asyncio
from typing import Any
from langchain_core.runnables import RunnableConfig
from typing import override

from graph.react.react_messages_agent import ReactMessagesAgent


class ReactChatAgent(ReactMessagesAgent):
    def __init__(
        self,
        system_prompt: str,
        newMemory: bool = True,
        tools: dict = [],
    ):
        super().__init__(system_prompt, newMemory, tools)

    def invoke(self, user: RunnableConfig | dict[str, Any]=None) -> None:
        super().invoke('', user)

    @override
    def _invoke(self, text: str, user: RunnableConfig | dict[str, Any]) -> None:
        asyncio.run(self._run_chat(user))

    async def _run_chat(self, user) -> None:
        await self._init_graph()
        print("输入你想问的内容，或者输入'exit'退出")
        while True:
            text = input(">>>")
            if text.lower() == "exit":
                break
            text = self._format_input(text)[0]
            await self._run_by_stream(text, user)
        await self.graph.checkpointer.conn.close()
