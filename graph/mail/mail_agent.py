from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command
from langmem import create_manage_memory_tool, create_search_memory_tool

from core.llm_graph import get_graph_agent
from graph.graph_ui import display
from graph.mail.definitions import State
from graph.mail.mail_triage_agent import MailTriageAgent


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


@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"


store_namespace = ["email_assistant", "langgraph_user_id"]
manage_memory_tool = create_manage_memory_tool(
    namespace=(store_namespace[0], f"{{{store_namespace[1]}}}", "collection")
)
search_memory_tool = create_search_memory_tool(
    namespace=(store_namespace[0], f"{{{store_namespace[1]}}}", "collection")
)


class MailAgent():

    def __init__(self, profile: dict, prompt_instructions: dict):
        self.triage_agent = MailTriageAgent(profile, prompt_instructions, store_namespace)
        self.llm_model = get_graph_agent(
            tools=[write_email, schedule_meeting, check_calendar_availability, manage_memory_tool, search_memory_tool],
            system_prompt=agent_system_prompt.format(instructions=prompt_instructions["agent_instructions"], **profile)
        )
        self.graph = StateGraph(State)\
            .add_node(self.triage_router)\
            .add_node("response_agent", self.llm_model)\
            .add_edge(START, "triage_router")\
            .compile(store=self.llm_model.store)
    
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
        return self.graph.invoke(input, config=config)