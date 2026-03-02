import pytest
import os
from pydantic import BaseModel
from outlines import inputs
from PIL import Image as PILImage

from src.models.image_text_to_text._litellm import LiteLLMModel
from src.utils._runtime_paths import resolve_project_root
from dotenv import load_dotenv

load_dotenv()


def _encode_image_for_outlines(image_path: str) -> PILImage.Image:
    image = PILImage.open(image_path)
    image_extension = os.path.basename(image_path).split(".")[-1].lower()
    if image_extension == "jpg":
        image_extension = "jpeg"
    image.format = image_extension
    return image


class Classification(BaseModel):
    object: str
    confidence: float


@pytest.fixture
def image_path():
    return os.path.join(resolve_project_root(), "tests/models/assets/plane.jpg")


@pytest.fixture
def output_schema():
    return Classification


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_litellm_model(image_path, output_schema):
    config = {
        "model": {
            "name": "openrouter/openai/gpt-5-nano",
        }
    }
    model = LiteLLMModel(config)
    assert model is not None
    prompt = inputs.Chat(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Tell me the name of the object in the image.",
                    },
                    {
                        "type": "image",
                        "image": inputs.Image(_encode_image_for_outlines(image_path)),
                    },
                ],
            }
        ]
    )
    result = model.generate(prompt, output_schema)
    assert result is not None
    assert result["object"] is not None
    assert result["confidence"] is not None
