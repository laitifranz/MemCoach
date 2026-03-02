import pytest

from src.models.image_text_to_text.utils import prompt_builder


@pytest.fixture(autouse=True)
def stub_outlines(monkeypatch):
    """Make PromptBuilder return raw chat payloads without depending on outlines."""
    monkeypatch.setattr(prompt_builder.inputs, "Chat", lambda chat: chat)
    monkeypatch.setattr(
        prompt_builder.inputs, "Image", lambda image: {"encoded": image}
    )


def test_get_prompt_with_system_prompt():
    builder = prompt_builder.PromptBuilder(
        {
            "user_prompt": "Describe the scene.",
            "system_prompt": "You are a helpful assistant.",
        }
    )

    result = builder.get_prompt()

    assert result[0]["role"] == "system"
    assert result[0]["content"][0]["text"] == "You are a helpful assistant."
    assert result[1]["role"] == "user"
    assert result[1]["content"][0]["text"] == "Describe the scene."


def test_get_prompt_with_images_and_assistant(monkeypatch):
    builder = prompt_builder.PromptBuilder(
        {"user_prompt": "Describe the initial photo."}
    )
    monkeypatch.setattr(
        builder, "_encode_image_for_outlines", lambda path: f"encoded:{path}"
    )
    pairs = [
        ("First observation.", "/tmp/img1.png"),
        ("Second observation.", "/tmp/img2.png"),
    ]

    result = builder.get_prompt(
        text_image_pairs=pairs, assistant_prompt="Acknowledged."
    )

    user_entry = result[0]
    assert user_entry["role"] == "user"
    # Expect original user prompt plus interleaved text/image items
    texts = [item["text"] for item in user_entry["content"] if item["type"] == "text"]
    assert texts == [
        "Describe the initial photo.",
        "First observation.",
        "Second observation.",
    ]
    images = [
        item["image"] for item in user_entry["content"] if item["type"] == "image"
    ]
    assert images == [
        {"encoded": "encoded:/tmp/img1.png"},
        {"encoded": "encoded:/tmp/img2.png"},
    ]
    assert result[-1]["role"] == "assistant"
    assert result[-1]["content"][0]["text"] == "Acknowledged."


def test_get_prompt_without_text_image_pairs():
    builder = prompt_builder.PromptBuilder({"user_prompt": "Hello there!"})

    result = builder.get_prompt()

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == [{"type": "text", "text": "Hello there!"}]
