from graph.mail.definitions import State, Router
from core.init_llmgw import get_chat_model


triage_system_prompt = f"""
< Role >
You are {{full_name}}'s executive assistant. You are a top-notch executive assistant who cares about {{name}} performing as well as possible.
</ Role >

< Background >
{{user_profile_background}}. 
</ Background >

< Instructions >

{{name}} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {{name}} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {{name}}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{{triage_no}}

There are also other things that {{name}} should know about, but don't require an email response. For these, you should notify {{name}} (using the `notify` response). Examples of this include:
{{triage_notify}}

Emails that are worth responding to:
{{triage_email}}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above

{{examples}}
</ Few shot examples >
"""

triage_user_prompt = f"""
Please determine how to handle the below email thread:

From: {{author}}
To: {{to}}
Subject: {{subject}}
{{email_thread}}
"""


searched_email_template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""


class MailTriageAgent():

    def __init__(self, profile: dict, prompt_instructions: dict, store_namespace: list[str]):
        self.llm_model = get_chat_model().with_structured_output(Router)
        self.profile = profile
        self.prompt_instructions = prompt_instructions
        self.store_namespace = store_namespace

    def invoke(self, state: State, config: dict, store) -> Router:
        namespace = (
            self.store_namespace[0],
            config["configurable"][self.store_namespace[1]],
            "examples",
        )
        examples = store.search(
            namespace,
            query=str({"email": state['email_input']})
        )
        system_prompt = triage_system_prompt.format(
            full_name=self.profile["full_name"],
            name=self.profile["name"],
            user_profile_background=self.profile["user_profile_background"],
            triage_no=self.prompt_instructions["triage_rules"]["ignore"],
            triage_notify=self.prompt_instructions["triage_rules"]["notify"],
            triage_email=self.prompt_instructions["triage_rules"]["respond"],
            examples=MailTriageAgent.format_few_shot_examples(examples)
        )
        user_prompt = triage_user_prompt.format(
            author=state['email_input']['author'], 
            to=state['email_input']['to'], 
            subject=state['email_input']['subject'], 
            email_thread=state['email_input']['email_thread']
        )
        return self.llm_model.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

    @staticmethod
    def format_few_shot_examples(examples):
        strs = ["Here are some previous examples:"]
        for eg in examples:
            strs.append(
                searched_email_template.format(
                    subject=eg.value["email"]["subject"],
                    to_email=eg.value["email"]["to"],
                    from_email=eg.value["email"]["author"],
                    content=eg.value["email"]["email_thread"][:400],
                    result=eg.value["label"],
                )
            )
        return "\n\n------------\n\n".join(strs)