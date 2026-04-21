from uuid import uuid4
from typing import Annotated, TypedDict, override

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from graph.graph_agent import GraphAgent


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


class MessagesAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


class ReactMessagesAgent(GraphAgent):

    def __init__(self, system_prompt: str, newMemory: bool = True, tools: dict = []):
        super().__init__(newMemory, tools)
        self.system_prompt = system_prompt
    
    @override
    def _create_graph(self, checkpointer, interrupt_before: list[str] = None):
        return StateGraph(MessagesAgentState)\
            .add_node("llm", self._call_llm)\
            .add_node("action", self._take_action)\
            .add_conditional_edges("llm", self._check_tool_calls, {True: "action", False: END})\
            .add_edge("action", "llm")\
            .set_entry_point("llm")\
            .compile(
                checkpointer=checkpointer,
                interrupt_before=interrupt_before
            )

    def _call_llm(self, state: MessagesAgentState) -> MessagesAgentState:
        messages = [SystemMessage(self.system_prompt)] + state["messages"]
        res = self.llm_model.invoke(messages)
        return {"messages": [res]}

    def _take_action(self, state: MessagesAgentState) -> MessagesAgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            result = self.tool_map[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

    def _check_tool_calls(self, state: MessagesAgentState) -> bool:
        tool_calls = state["messages"][-1].tool_calls
        return len(tool_calls) > 0

    @override
    def _format_input(self, text: str | list[str] | None) -> list[MessagesAgentState|None]:
        return ReactMessagesAgent.format_list_input(text)

    @staticmethod
    def format_list_input(text: str | list[str] | None) -> list[MessagesAgentState|None]:
        if not isinstance(text, list):
            text = [text]
        return [(None if i is None else {"messages": [HumanMessage(i)]}) for i in text]