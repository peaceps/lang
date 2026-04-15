from langgraph.store.base import BaseStore


store_config = {
    "agent_key": "email_assistant",
    "user_key": "langgraph_user_id",
}


def set_store_config(agent_key: str, user_key: str):
    store_config["agent_key"] = agent_key
    store_config["user_key"] = user_key


def get_messages_store_namespace(config: dict = None):
    return (store_config["agent_key"], store_config["user_key"] if config is None else config["configurable"][store_config["user_key"]], "messages")


def get_examples_store_namespace(config: dict = None):
    return (store_config["agent_key"], store_config["user_key"] if config is None else config["configurable"][store_config["user_key"]], "examples")


def get_user_store_namespace(config: dict = None):
    return (config["configurable"][store_config["user_key"]],)


def get_config_from_user(user_id: str):
    return { "configurable": { store_config["user_key"]: user_id } }


def get_prompt_from_store(store: BaseStore, namespace: tuple[str, ...], key: str, default_prompt: str):
    result = store.get(namespace, key)
    if result is None:
        store.put(namespace, key, {"prompt": default_prompt})
        prompt = default_prompt
    else:
        prompt = result.value["prompt"]
    return prompt


def update_prompt_in_store(store: BaseStore, namespace: tuple[str, ...], updated: list[dict]):
    for updated_prompt in updated:
        name = updated_prompt['name']
        old_prompt = get_prompt_from_store(store, namespace, name, updated_prompt['prompt'])
        if updated_prompt['prompt'] != old_prompt:
            print(f"updated {name}")
            store.put(namespace, name, {"prompt": updated_prompt['prompt']})