from typing import Any

from langchain.agents.middleware.types import AgentState
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, NotRequired, TypedDict
from langgraph.graph import add_messages


class ResponseAgentState(AgentState[Any]):
    """create_agent 子图状态：保留 messages/jump_to 等，并带上 email_input 供 dynamic_prompt 读取。"""

    email_input: NotRequired[dict]


class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]


class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )