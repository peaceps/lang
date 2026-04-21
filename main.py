from chain.tools.pydantic_tools import run as pydantic_run
from chain.tools.openapi_tools import run as openapi_run
from chain.tools.function_calling_tool import run as function_calling_run
from chain.tools.agent import run as agent_run
from chain.rag.retriever import run as retriever_run
from graph.mail.mail import run as mail_run
from graph.react.react import run as react_run
from graph.essay.essay import run as essay_run


def main() -> None:
    # retriever_run()
    # openapi_run()
    # pydantic_run()
    # function_calling_run()
    # agent_run()
    # mail_run()
    react_run()
    # essay_run()

if __name__ == "__main__":
    main()