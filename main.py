from chain.tool_calls.pydantic_tools import run as prun
from chain.tool_calls.openapi_tools import run as orun
from chain.tool_calls.agent_tools import run as arun
from chain.embedding import run as erun


def main() -> None:
    erun()


if __name__ == "__main__":
    main()