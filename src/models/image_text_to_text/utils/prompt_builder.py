from outlines import inputs
from PIL import Image as PILImage
import os


class PromptBuilder:
    def __init__(self, config: dict):
        self._system_prompt = config.get("system_prompt", None)
        self._user_prompt = config["user_prompt"]

    def _encode_image_for_outlines(self, image_path: str) -> PILImage.Image:
        image = PILImage.open(image_path)
        image_extension = os.path.basename(image_path).split(".")[-1].lower()
        if image_extension == "jpg":
            image_extension = "jpeg"
        image.format = image_extension
        return image

    def get_prompt(
        self,
        text_image_pairs: list[tuple[str, str]] = None,
        assistant_prompt: str = None,
        image_before_text: bool = False,
    ) -> inputs.Chat:
        chat = []
        if self._system_prompt is not None:
            chat.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._system_prompt}],
                }
            )

        if text_image_pairs is not None:
            if image_before_text:
                tmp_user_content = {"role": "user", "content": []}
                for text, image in text_image_pairs:
                    tmp_user_content["content"].extend(
                        [
                            {
                                "type": "image",
                                "image": inputs.Image(
                                    self._encode_image_for_outlines(image)
                                ),
                            },
                            {"type": "text", "text": text},
                        ]
                    )
                tmp_user_content["content"].extend(
                    [{"type": "text", "text": self._user_prompt}]
                )

            else:
                tmp_user_content = {
                    "role": "user",
                    "content": [{"type": "text", "text": self._user_prompt}],
                }
                for text, image in text_image_pairs:
                    tmp_user_content["content"].extend(
                        [
                            {"type": "text", "text": text},
                            {
                                "type": "image",
                                "image": inputs.Image(
                                    self._encode_image_for_outlines(image)
                                ),
                            },
                        ]
                    )
            chat.append(tmp_user_content)
        else:
            chat.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self._user_prompt}],
                }
            )

        if assistant_prompt is not None:
            chat.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_prompt}],
                }
            )
        return inputs.Chat(chat)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def user_prompt(self) -> str:
        return self._user_prompt

    @staticmethod
    def get_message_variable(
        input_chat: inputs.Chat | list[inputs.Chat], unwrap_image: bool = False
    ):
        def _unwrap(chat: inputs.Chat):
            if not unwrap_image:
                return chat.messages
            unwrapped = []
            for message in chat.messages:
                content = message.get("content", [])
                if isinstance(content, list):
                    content = [
                        {**item, "image": item["image"].image}
                        if isinstance(item, dict)
                        and item.get("type") == "image"
                        and isinstance(item.get("image"), inputs.Image)
                        else item
                        for item in content
                    ]
                unwrapped.append({**message, "content": content})
            return unwrapped

        if isinstance(input_chat, inputs.Chat):
            return _unwrap(input_chat)
        return [_unwrap(chat) for chat in input_chat]
