import re
from core.init_llmgw import get_openai_chat_model

action_re = re.compile(r"^Action: (\w+): (.*)$")

class ReactSimpleAgent:
    def __init__(self, system_prompt: str, known_actions: dict, max_iterations: int = 10):
        self.llm_model = get_openai_chat_model()
        self.known_actions = known_actions
        self.system_prompt = system_prompt
        self.messages = [("system", self.system_prompt)]
    
    def call_model(self, input: str) -> None:
        self.messages.append(("user", input))
        res = self.llm_model.invoke(self.messages).content.strip()
        self.messages.append(("assistant", res))
        print(res)
        return res

    def invoke(self, input: str) -> None:
        i = 0
        next_prompt = input
        while i < max_iterations:
            i += 1
            res = self.call_model(next_prompt)
            actions = [
                action_re.match(a) 
                for a in res.split('\n') 
                if action_re.match(a)
            ]
            if actions:
                # There is an action to run
                action, action_input = actions[0].groups()
                if action not in self.known_actions:
                    raise Exception("Unknown action: {}: {}".format(action, action_input))
                print(" -- running {} {}".format(action, action_input))
                observation = self.known_actions[action](action_input)
                print("Observation:", observation)
                next_prompt = "Observation: {}".format(observation)
            else:
                break
        return res
