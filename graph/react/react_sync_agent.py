import sqlite3
from uuid import uuid4
from typing import override

from langgraph.checkpoint.sqlite import SqliteSaver
from graph.react.react_messages_agent import ReactMessagesAgent, MessagesAgentState
from tools.tools import extract_content


class ReactSyncAgent(ReactMessagesAgent):
    def __init__(
        self,
        system_prompt: str,
        newMemory: bool = True,
        tools: dict = [],
        require_confirmation_on_action: bool = False,
    ):
        super().__init__(system_prompt, newMemory, tools)

    @override
    def _init_graph(self, interrupt_before: list[str] = None):
        self._ensure_checkpoint_parent()
        conn = sqlite3.connect(
            str(self._checkpoint_path),
            check_same_thread=False,
        )
        checkpointer = SqliteSaver(conn)
        self.graph = self._create_graph(checkpointer, interrupt_before)

    def _invoke_graph(self, text: str | None, user, action, interrupt_before: list[str] = None) -> None:
        if user is None:
            user = {"configurable": {"thread_id": str(uuid4())}}
        self._init_graph(interrupt_before)
        for p in self._format_input(text):
            getattr(self, action)(p, user)
        self.graph.checkpointer.conn.close()

    def invoke_sync(self, text: MessagesAgentState|None, user=None) -> None:
        self._invoke_graph(text, user, '_run_by_sync')

    def _run_by_sync(self, p: MessagesAgentState|None, user) -> None:
        res = self.graph.invoke(p, user)
        for msg in res['messages']:
            print(extract_content(msg))

    def invoke_steps(self, text: MessagesAgentState|None, user=None) -> None:
        self._invoke_graph(text, user, '_run_by_step', interrupt_before=['action'])

    def _run_by_step(self, p: MessagesAgentState|None, user) -> None:
        stream_res = self.graph.stream(p, user)
        for chunk in stream_res:
            if 'llm' in chunk or 'action' in chunk:
                messages = chunk['llm' if 'llm' in chunk else 'action']['messages']
                print(extract_content(messages[-1]))
        self._loop_step_actions(user)

    def _loop_step_actions(self, user) -> None:
        if self.graph.get_state(user).next:
            _input = input("允许调用工具吗？(y/n)\n>>>")
            if _input.lower() != "y":
                print("退出工具调用流程")
            else:
                self._run_by_step(None, user)
