import pytest
from pydantic import BaseModel
from PIL import Image as PILImage
import io
from outlines.inputs import Chat, Image
from src.models.image_text_to_text._hf import HfModel


@pytest.fixture
def image():
    width, height = 256, 256
    blue_background = (0, 0, 255)
    image = PILImage.new("RGB", (width, height), blue_background)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image = PILImage.open(buffer)

    return image


@pytest.fixture
def model():
    config = {
        "model": {
            "name": "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration",
            "max_new_tokens": 20,
            "temperature": 0.001,
        }
    }
    model = HfModel(config)
    return model


def test_hf_model(model, image):
    class Foo(BaseModel):
        name: str

    # single message
    result = model.generate(
        Chat(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one sentence:",
                        },
                        {"type": "image", "image": Image(image)},
                    ],
                },
            ]
        ),
        Foo,
    )
    assert "name" in result

    # batch messages
    result = model.generate(
        [
            Chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in one sentence:",
                            },
                            {"type": "image", "image": Image(image)},
                        ],
                    },
                ]
            ),
            Chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in one sentence:",
                            },
                            {"type": "image", "image": Image(image)},
                        ],
                    },
                ]
            ),
        ]
    )

    assert isinstance(result, list)
    assert len(result) == 2


def test_hf_loading(model):
    assert model is not None
    assert model.config is not None
    assert (
        model.config["model"]["name"]
        == "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
    )
