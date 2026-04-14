from chain.tools.functions import search_wikipedia, get_temperature
from core.llm_chain import Agent

def run() -> None:
    tools = [search_wikipedia, get_temperature]
    agent_chain = Agent(tools=tools)
    agent_chain.invoke("hi, I'm bob")
    agent_chain.invoke("hi, what's my name?")
    agent_chain.invoke("what is the temperature in Hangzhou and Shenzhen?")
