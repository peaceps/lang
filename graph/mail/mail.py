from graph.mail.mail_agent import MailAgent


profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently. Remember to search memory before answering any questions."
}

email = [
{
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
},
{
    "author": "Good luck <good.luck@awards.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "$50000 award won!",
    "email_thread": """
Hi John,

You get the great award $50000! Reply for the award details!!!

Thanks!
GL""",
},
{
    "author": "Nobel Prize <nobel.prize@nobel.org>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Title of Nobel Prize Physics Award",
    "email_thread": """
Hi John,

Congratulations on winning the Nobel Prize in Physics! You are the first person to win the Nobel Prize in Physics for the second time.
The committee has decided to award you the prize for your work on the theory of relativity.
The prize amount is $1000000.

The celebration will be held on October 10th, 2026 at the Stockholm Concert Hall.

Thanks!
Nobel prize committee""",
},
{
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Follow up",
    "email_thread": """
Any update on my previous ask?""",
},
]

config = { "configurable": { "langgraph_user_id": "john_doe" } }

def run() -> None:
    email_agent = MailAgent(profile, prompt_instructions)
    res = email_agent.invoke({"email_input": email[0]}, config=config)
    for m in res["messages"]:
        m.pretty_print()
    res = email_agent.invoke({"email_input": email[3]}, config=config)
    for m in res["messages"]:
        m.pretty_print()
