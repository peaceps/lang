from core.llm_chain import FunctionCallingToolChain
from tools.tools import search_wikipedia, get_temperature

def run() -> None:
    tools = [search_wikipedia, get_temperature]
    model = FunctionCallingToolChain(tools=tools)
    model.invoke("what is the temperature in Beijing?")
    model.invoke("what is the summary of the page 'Hangzhou'?")
    model.invoke("hi!")
