import difflib

from typing import Any, Literal
from langchain.agents.middleware import dynamic_prompt
from langchain.agents.middleware.types import ModelRequest
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.config import get_config
from langmem.utils import AnyMessage

from core.llm_graph import get_graph_agent
from core.init_llmgw import get_multi_prompt_optimizer
from graph.graph_ui import display
from graph.mail.definitions import ResponseAgentState, State
from graph.mail.mail_triage_agent import MailTriageAgent
from tools.tools import check_calendar_availability, schedule_meeting, write_email
from tools.store_utils import get_prompt_from_store, update_prompt_in_store, get_messages_store_namespace, get_user_store_namespace


agent_system_prompt = f"""
< Role >
You are {{full_name}}'s executive assistant. You are a top-notch executive assistant who cares about {{name}} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {{name}}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{{instructions}}
</ Instructions >
"""


def _response_agent_dynamic_prompt(default_system_prompt: str):
    """每次调用模型前根据 state['email_input'] 拼完整 system（create_agent 的 system_prompt 会被覆盖）。

    在同一次模型调用里如需 BaseStore / RunnableConfig，不要用 __init__ 闭包捕获，应使用：
    - ``request.runtime.store`` — LangGraph 注入的 store（未 ``compile(store=...)`` 时可能为 ``None``）
    - ``get_config()`` — 当前 runnable 的 ``RunnableConfig``（含 ``configurable``、thread 等）
    """

    @dynamic_prompt
    def _mail_response_system(request: ModelRequest) -> str:
        # 需要 RunnableConfig 时: from langgraph.config import get_config 然后 get_config()
        # 需要 BaseStore 时: request.runtime.store（可能为 None）
        user_namespace = get_user_store_namespace(get_config())
        prompt = get_prompt_from_store(request.runtime.store, user_namespace, "agent_instructions", default_system_prompt)
        return prompt

    return _mail_response_system


class MailAgent():

    def __init__(self, profile: dict, prompt_instructions: dict):
        self.prompt_instructions = prompt_instructions
        self.triage_agent = MailTriageAgent(profile, prompt_instructions)
        static_system = agent_system_prompt.format(
            instructions=prompt_instructions["agent_instructions"], **profile
        )
        self.llm_model = get_graph_agent(
            tools=[
                write_email,
                schedule_meeting,
                check_calendar_availability,
                create_manage_memory_tool(namespace=get_messages_store_namespace()),
                create_search_memory_tool(namespace=get_messages_store_namespace()),
            ],
            system_prompt=None,
            middleware=[_response_agent_dynamic_prompt(static_system)],
            state_schema=ResponseAgentState,
        )
        self.graph = StateGraph(State)\
            .add_node(self.triage_router)\
            .add_node("response_agent", self.llm_model)\
            .add_edge(START, "triage_router")\
            .compile(store=self.llm_model.store)
        self.optimizer = get_multi_prompt_optimizer()
    
    def triage_router(
        self,
        state: State,
        *,
        config: RunnableConfig,
        store: BaseStore,
    ) -> Command[Literal["response_agent", "__end__"]]:
        result = self.triage_agent.invoke(state, config=config, store=store)

        if result.classification == "respond":
            print("📧 Classification: RESPOND - This email requires a response")
            goto = "response_agent"
            update = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Respond to the email {state['email_input']}",
                    }
                ]
            }
        elif result.classification == "ignore":
            print("🚫 Classification: IGNORE - This email can be safely ignored")
            update = None
            goto = END
        elif result.classification == "notify":
            # If real life, this would do something else
            print("🔔 Classification: NOTIFY - This email contains important information")
            update = None
            goto = END
        else:
            raise ValueError(f"Invalid classification: {result.classification}")
        return Command(goto=goto, update=update)

    def draw_graph(self) -> None:
        display(self.graph)

    def invoke(self, input: dict, config: RunnableConfig | dict[str, Any]) -> dict:
        res = self.graph.invoke(input, config=config)
        for m in res["messages"]:
            m.pretty_print()
        return res

    def feedback_trace(self, messages: list[AnyMessage], feedback: str, config: dict[str, Any]):
        # trajectories: 每项为 (messages, feedback)；勿用 (messages,) 单元素元组
        if len(messages) == 0:
            return
        prompts_info = self._get_prompts_info(config)
        updated_prompts = self.optimizer.invoke(
            {
                "trajectories": [(messages, feedback)],
                "prompts": prompts_info,
            }
        )
        update_prompt_in_store(self.graph.store, get_user_store_namespace(config), updated_prompts)


    def _get_prompts_info(self, config: dict) -> list[dict]:
        store = self.graph.store
        user_namespace = get_user_store_namespace(config)
        return [
            {
                "name": "agent_instructions",
                "prompt": get_prompt_from_store(store, user_namespace, "agent_instructions", self.prompt_instructions["agent_instructions"]),
                "update_instructions": "keep the instructions short and to the point",
                "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
                
            },
            {
                "name": "triage_ignore", 
                "prompt": get_prompt_from_store(store, user_namespace, "triage_ignore", self.prompt_instructions["triage_rules"]["ignore"]),
                "update_instructions": "keep the instructions short and to the point",
                "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

            },
            {
                "name": "triage_notify", 
                "prompt": get_prompt_from_store(store, user_namespace, "triage_notify", self.prompt_instructions["triage_rules"]["notify"]),
                "update_instructions": "keep the instructions short and to the point",
                "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

            },
            {
                "name": "triage_respond", 
                "prompt": get_prompt_from_store(store, user_namespace, "triage_respond", self.prompt_instructions["triage_rules"]["respond"]),
                "update_instructions": "keep the instructions short and to the point",
                "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

            },
        ]