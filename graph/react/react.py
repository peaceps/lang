from graph.react.react_simple_agent import ReactSimpleAgent
from graph.react.react_chat_agent import ReactChatAgent
from graph.react.react_sync_agent import ReactSyncAgent


dog_system_prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()


searcher_system_prompt = """你是一个智能的研究助手。使用搜索引擎来查找信息。\
你可以进行多次调用（可以同时进行，也可以按顺序进行）。\
只有在你明确知道自己想查什么时才进行搜索。\
如果在提出后续问题之前需要先查找一些信息，你也可以这样做！

重要：对**同一个用户问题**，在已经收到至少一条工具返回的观测结果后，你必须用**一条最终回复**直接回答用户，\
整合工具结果即可；不要为「核实、纠错、反复确认」而**再次调用搜索工具**，除非用户明确提出了**新的、独立的事实问题**。\
若工具返回与常识不符，可在回答中简要说明不确定性，但仍须结束本轮工具循环并给出结论。
""".strip()


def calculate(what):
    return eval(what)


def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")


known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}


def run() -> None:
    user1 = {"configurable": {"thread_id": "123"}}
    user2 = {"configurable": {"thread_id": "321"}}
    user3 = {"configurable": {"thread_id": "111"}}
    # simple_agent = ReactSimpleAgent(dog_system_prompt, known_actions)
    # sync_agent = ReactSyncAgent(searcher_system_prompt, False)
    # sync_agent.invoke("巴黎今天的天气怎样？", user1)
    # sync_agent.invoke("柏林呢？", user1, True)
    # sync_agent.invoke("杭州呢？", user1)
    # sync_agent.invoke("气候怎样？", user1, True)
    chat_agent = ReactChatAgent(searcher_system_prompt, newMemory=False)
    chat_agent.invoke(user=user1)