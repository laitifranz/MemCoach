SYSTEM_PROMPT = "You are an observer."

USER_PROMPT = {
    "feedback_elicitation_prompt": """
Your task is to determine the actions required to transform Image A into Image B. 
Strictly avoid explicit or implicit references to the images when suggesting action items, and make each action item self-contained. 

Produce a structured JSON object that must include: 
- actions: a list of precise and well-informative semantic actions.

Respond with a valid JSON object and no explanation. 
""".strip()
}
