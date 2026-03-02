SYSTEM_PROMPT = "You are an observer."

USER_PROMPT = {
    "training_prompt": """
Your task is to determine the actions required to improve the memorability of the input image. 
Output a set of precise and well-informative semantic actions separated by commas. No explanation.
""".strip(),
    "inference_prompt": """
Your task is to determine the actions required to improve the memorability of the input image. 
Output a set of precise and well-informative semantic actions separated by commas. No explanation.
""".strip(),
    "inference_constrained_prompt": """
Your task is to determine the actions required to to improve the memorability of the input image.

Produce a structured JSON object that must include: 
- actions: a list of precise and well-informative semantic actions.

Respond with a valid JSON object and no explanation. 
""".strip(),
}
