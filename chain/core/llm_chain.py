from typing import Any, List, override
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.agents import AgentFinish
from langchain_classic.agents.output_parsers import ToolsAgentOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chain.core.init_llmgw import get_openai_client
from pydantic import BaseModel


class ToolChain:

    def __init__(self, tools: List[Any], tool_choice: str=None, system_prompt: str = "Think carefully."):
        self.tools = tools
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"{system_prompt}"),
            ("user", "{input}")
        ])
        self.llm = get_openai_client(
            tools=tools,
            tool_choice=tool_choice,
        )
        self.llm_chain = self.prompt_template | self.llm
        self.parser = self._get_parser()

    def _get_parser(self) -> Any:
        return JsonOutputToolsParser()

    def _get_reducer(self, split: bool = False) -> Any:
        return ToolChainUtils.unwrap

    def _call_model(self, input: str, split: bool = False) -> Any:
        llm_chain = self.llm_chain | self.parser
        chain = ToolChainUtils.splitter | llm_chain.map() | self._get_reducer(split)
        return chain.invoke(input if split else ([ToolChainUtils.wrap_input(input)]))

    def invoke(self, input: str, split: bool = False) -> None:
        res = self._call_model(input, split)
        print(res)


class PydanticToolChain(ToolChain):

    def __init__(self, tool_class: type[BaseModel],
     system_prompt="Think carefully. And finally call a tool, must call a tool."):
        self.tool_class = tool_class
        openai_tool = convert_to_openai_tool(tool_class)
        super().__init__([openai_tool], "required", system_prompt)

    @override
    def _get_parser(self) -> Any:
        return PydanticToolsParser(
            tools=[self.tool_class], first_tool_only=True
        )
        
    @override
    def _get_reducer(self, split: bool = False) -> Any:
        if not split:
            return ToolChainUtils.unwrap
        return PydanticToolChain._merge_pydantic_list

    @override
    def _call_model(self, input: str, split: bool = False) -> Any:
        res = super()._call_model(input, split)
        if res is None:
            raise ValueError(
                "Model output had no matching tool call (parser returned None). "
                "Check tool_choice matches tools[].function.name."
            )
        return self.tool_class.model_validate(res)
        
    @staticmethod
    def _merge_pydantic_list(items: list[type[BaseModel]]) -> dict: 
        pydantic_dists = map(lambda i: dict(i), filter(lambda j: j is not None, items))
        return ToolChainUtils.merge_dict_list(pydantic_dists)


class AgentToolChain(ToolChain):

    def __init__(self, tools: List[Any]):        
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        super().__init__(openai_tools)
        self.tools_dict = {t.name: t for t in tools}

    def _get_parser(self) -> Any:
        return ToolsAgentOutputParser()

    def _get_reducer(self, split: bool = False) -> Any:
        return self._reduce_result

    def _reduce_result(self, items: list) -> Any:
        agent_result = ToolChainUtils.unwrap(items)
        if isinstance(agent_result, AgentFinish):
            return agent_result.return_values['output']
        else:
            action = agent_result[0]
            return self.tools_dict[action.tool].invoke(action.tool_input)
        
        
class ToolChainUtils:

    splitter = RunnableLambda(lambda x: x if isinstance(x, list) else [ToolChainUtils.wrap_input(doc) for doc in RecursiveCharacterTextSplitter().split_text(x)])

    @staticmethod
    def merge_dict_list(items: map) -> dict:
        merged: dict = {}
        for d in items:
            for k, v in d.items():
                if isinstance(v, list):
                    merged.setdefault(k, []).extend(v)
                else:
                    merged[k] = v
        return merged

    @staticmethod
    def wrap_input(input: str) -> list:
        return {"input": input}

    @staticmethod
    def unwrap(x: map) -> Any:
        for i in x:
            return i