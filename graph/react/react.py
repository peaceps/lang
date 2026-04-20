from graph.react.react_simple_agent import ReactSimpleAgent
from graph.react.react_graph_agent import ReactGraphAgent


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


weather_system_prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
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
    # react_agent = ReactSimpleAgent(dog_system_prompt, known_actions)
    react_agent = ReactGraphAgent(weather_system_prompt)
    # react_agent.invoke("2025年欧冠的冠军是哪个队伍？它所在的城市当年的GDP是多少？")
    user1 = {"configurable": {"thread_id": "123"}}
    user2 = {"configurable": {"thread_id": "321"}}
    user3 = {"configurable": {"thread_id": "444"}}
    react_agent.invoke_stream(["杭州天气如何？"], user1)
    react_agent.invoke_sync(["巴黎今年的GPD是多少？"], user2)
    react_agent.invoke_steps(["伦敦的经纬度是多少？"], user3)
    react_agent.invoke_sync(["北京呢？"], user3)
    react_agent.invoke_steps(["北京呢？"], user1)
    react_agent.invoke_stream(["北京呢？"], user2)
    react_agent.shutdown()