from langgraph.graph import StateGraph, END, START
from typing import Any, Literal, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from core.init_llmgw import get_openai_chat_model, get_tavily_search_model


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class ReactGraphAgent:
    def __init__(self, system_prompt: str, tools: dict):
        self.llm_model = get_openai_chat_model(tools)
        self.system_prompt = system_prompt
        self.graph = StateGraph(AgentState)\
            .add_node("llm", self._call_llm)\
            .add_node("action", self._take_action)\
            .add_conditional_edges("llm", self._check_tool_calls, {True: "action", False: END})\
            .add_edge("action", "llm")\
            .set_entry_point("llm")\
            .compile()
        self.tool_map = {t.name: t for t in tools}

    def _call_llm(self, state: AgentState) -> AgentState:
        messages = [SystemMessage(self.system_prompt)] + state["messages"]
        res = self.llm_model.invoke(messages)
        return {"messages": [res]}

    def _take_action(self, state: AgentState) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling tool {t}")
            result = self.tool_map[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        return {'messages': results}

    def _check_tool_calls(self, state: AgentState) -> bool:
        tool_calls = state["messages"][-1].tool_calls
        return len(tool_calls) > 0

    def invoke(self, input: str) -> None:
        res = self.graph.invoke({"messages": [HumanMessage(input)]})
        print(res['messages'][-1].content)