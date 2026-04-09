from chain.tools.pydantic_tools import run as prun
from chain.tools.openapi_tools import run as orun
from chain.tools.function_calling_tool import run as frun
from chain.tools.agent import run as arun
from chain.rag.retriever import run as erun


def main() -> None:
    erun()
    orun()
    prun()
    frun()
    arun()


if __name__ == "__main__":
    main()