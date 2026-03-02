SYSTEM_PROMPT = "You are a camera coach for a photographer."

USER_PROMPT = {
    "zero_shot_memorability_prompt": """
Your task is to determine the actions required to to improve the memorability of the input image.

Produce a structured JSON object that must include: 
- actions: a list of precise and well-informative semantic actions.

Respond with a valid JSON object and no explanation. 
""".strip()
}
