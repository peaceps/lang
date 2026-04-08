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
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"{system_prompt}"),
            ("user", "{input}")
        ])
        llm = get_openai_client(
            tools=tools,
            tool_choice=tool_choice,
        )
        parser = self._get_parser()
        self.llm_chain = prompt_template | llm | parser

    def _get_parser(self) -> Any:
        return JsonOutputToolsParser()

    def _call_model(self, input: str) -> Any:
        return self.llm_chain.invoke(input)

    def invoke(self, input: str) -> None:
        res = self._call_model(input)
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
    def _call_model(self, input: str, split: bool = False) -> Any:
        if not split:
            return super()._call_model(input)
        chain = ToolChainUtils.splitter | self.llm_chain.map() | PydanticToolChain._merge_pydantic_list
        res = chain.invoke(input)
        if res is None:
            raise ValueError(
                "Model output had no matching tool call (parser returned None). "
                "Check tool_choice matches tools[].function.name."
            )
        return self.tool_class.model_validate(res)

    @override
    def invoke(self, input: str, split: bool = False) -> None:
        res = self._call_model(input, split)
        print(res)
        
    @staticmethod
    def _merge_pydantic_list(items: list[type[BaseModel]]) -> dict: 
        pydantic_dists = map(lambda i: dict(i), filter(lambda j: j is not None, items))
        return ToolChainUtils.merge_dict_list(pydantic_dists)


class AgentToolChain(ToolChain):

    def __init__(self, tools: List[Any]):        
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        super().__init__(openai_tools)
        self.tools_dict = {t.name: t for t in tools}
        self.llm_chain = self.llm_chain | self._reduce_result

    @override
    def _get_parser(self) -> Any:
        return ToolsAgentOutputParser()

    def _reduce_result(self, agent_result: Any) -> Any:
        if isinstance(agent_result, AgentFinish):
            return agent_result.return_values['output']
        else:
            action = agent_result[0]
            return self.tools_dict[action.tool].invoke(action.tool_input)


class ToolChainUtils:

    splitter = RunnableLambda(lambda x: [ToolChainUtils.wrap_input(doc) for doc in RecursiveCharacterTextSplitter().split_text(x)])

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
