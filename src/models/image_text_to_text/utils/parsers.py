import logging
from pydantic import BaseModel, ValidationError
from json_repair import repair_json


def parse_output(json_string: str, output_schema: BaseModel | None) -> str | dict:
    if output_schema is None:
        return json_string
    else:
        tmp_json_string = repair_json(
            json_string
        )  # if the string was super broken this will return an empty string
        try:
            return output_schema.model_validate_json(
                tmp_json_string
            ).model_dump()  # if not valid, return default output_schema instance
        except ValidationError:  # just in case the syntax is not valid, return default output_schema instance
            logging.warning(
                "Error parsing constrained output, returning default output_schema instance."
            )
            return output_schema().model_dump()
