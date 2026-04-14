import uuid
from graph.mail.mail_agent import MailAgent, store_namespace


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

emails = [
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
{
    "author": "Sarah Chen <sarah.chen@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Update: Backend API Changes Deployed to Staging",
    "email_thread": """Hi John,

Just wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:

- Implemented JWT refresh token rotation
- Added rate limiting for login attempts
- Updated API documentation with new endpoints

All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*

No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.

Best regards,
Sarah
"""
},
{
    "author": "Tom Jones <tome.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John - want to buy documentation?""",
}
]


examples = [
    {"email": emails[0], "label": "respond"},
    {"email": emails[1], "label": "ignore"},
    {"email": emails[2], "label": "notify"},
    {"email": emails[3], "label": "respond"},
    {"email": emails[4], "label": "ignore"},
    {"email": emails[5], "label": "respond"},
]


config = { "configurable": { store_namespace[1]: "smith" } }


def run() -> None:
    email_agent = MailAgent(profile, prompt_instructions)
    store = email_agent.graph.store
    store.put(
        (store_namespace[0], config["configurable"][store_namespace[1]], "examples"),
        str(uuid.uuid4()),
        examples[5]
    )
    test_mail = {
        "author": "Joe Kite <joe.kite@bar.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Quick question about API documentation",
        "email_thread": """Hi John - want to buy documentation?""",
    }
    res = email_agent.invoke({"email_input": test_mail}, config=config)
    for m in res["messages"]:
        m.pretty_print()
