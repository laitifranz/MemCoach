from pydantic import BaseModel, Field
from typing import List


class ConstrainedActionListOutput(BaseModel):
    actions: List[str] = Field(
        default_factory=lambda: ["no actions found"],
        description="A list of actions.",
        min_items=1,
        max_items=10,
    )


if __name__ == "__main__":
    dummy_json = '{"actions": ["action1", "action2", "action3"]}'
    result = ConstrainedActionListOutput.model_validate_json(dummy_json)
    print(result)
