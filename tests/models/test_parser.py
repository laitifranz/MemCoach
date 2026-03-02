from pydantic import BaseModel

from src.models.image_text_to_text.utils.parsers import parse_output


class DummySchema(BaseModel):
    text: str = "default"
    count: int = 0


def test_parse_output_without_schema_returns_raw_string():
    raw = '{"foo": "bar"}'
    assert parse_output(raw, None) == raw


def test_parse_output_with_schema_and_repair():
    broken_json = '{"text": "hello", "count": 3,}'
    result = parse_output(broken_json, DummySchema)
    assert result == {"text": "hello", "count": 3}


def test_parse_output_returns_default_on_validation_error(caplog):
    caplog.set_level("WARNING")
    result = parse_output("not valid json", DummySchema)
    assert result == {"text": "default", "count": 0}
    assert "Error parsing constrained output" in caplog.text
