from typing import Any, TypedDict, override

from langmem.prompts.types import AnyMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

from core.init_llmgw import get_tavily_client
from graph.graph_agent import GraphAgent


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: list[str]
    revision_number: int
    max_revisions: int


class Queries(BaseModel):
    queries: list[str] = Field(..., description="检索查询列表")


class EssayAgent(GraphAgent):
    def __init__(self, prompts: dict, newMemory: bool = True, tools: list[Any]=[]):
        super().__init__(newMemory, tools)
        self.prompts = prompts
        self.tavily = get_tavily_client()

    @override
    def _create_graph(self, checkpointer=None):
        return StateGraph(AgentState) \
            .add_node("planner", self._plan_node) \
            .add_node("generate", self._generation_node) \
            .add_node("reflect", self._reflection_node) \
            .add_node("research_plan", self._research_plan_node) \
            .add_node("research_critique", self._research_critique_node) \
            .set_entry_point("planner") \
            .add_conditional_edges("generate", self._should_continue, {
                "reflect": "reflect",
                END: END
            }) \
            .add_edge("planner", "research_plan") \
            .add_edge("research_plan", "generate") \
            .add_edge("reflect", "research_critique") \
            .add_edge("research_critique", "generate") \
            .compile(checkpointer=checkpointer)
    
    def _plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.prompts['plan']), 
            HumanMessage(content=state['task'])
        ]
        response = self.llm_model.invoke(messages)
        return {"plan": response.content}

    def _research_plan_node(self, state: AgentState):
        queries = self._queries_llm_model([
            SystemMessage(content=self.prompts['research_plan']),
            HumanMessage(content=state['task']),
        ])
        content = state['content'] or []
        for q in queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def _generation_node(self, state: AgentState):
        content = "\n\n".join(state['content'] or [])
        user_message = HumanMessage(
            content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
        )
        messages = [
            SystemMessage(
                content=self.prompts['writer'].format(content=content)
            ),
            user_message
        ]
        response = self.llm_model.invoke(messages)
        return {
            "draft": response.content, 
            "revision_number": state.get("revision_number", 1) + 1
        }

    def _reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.prompts['reflection']), 
            HumanMessage(content=state['draft'])
        ]
        response = self.llm_model.invoke(messages)
        return {"critique": response.content}

    def _research_critique_node(self, state: AgentState):
        queries = self._queries_llm_model([
            SystemMessage(content=self.prompts['research_critique']),
            HumanMessage(content=state['critique']),
        ])
        content = state['content'] or []
        for q in queries:
            response = self.tavily.search(query=q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return {"content": content}

    def _should_continue(self, state: AgentState):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "reflect"

    @override
    def _format_input(self, text: str) -> list[AgentState]:
        return [{
            'task': text,
            "content": [],
            "plan": "",
            "draft": "",
            "critique": "",
            "max_revisions": 2,
            "revision_number": 1,
        }]

    def _queries_llm_model(self, messages: list[AnyMessage]):
        """function_calling：规避部分网关上 json_schema 流式 + logprobs 的兼容问题。llm 勿再绑会与结构化 tool 竞争的其它工具。"""
        res = self.llm_model.invoke(messages)
        if res.content:
            return [q.strip() for q in res.content.split("\n") if q.strip()]
        return [str(messages[-1].content or "")[:200]]
