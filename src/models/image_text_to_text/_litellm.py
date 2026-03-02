from src.models.image_text_to_text._base import ImageTextToTextModel
import litellm
from outlines import inputs
import logging
from src.utils._runtime_paths import get_model_name

logger = logging.getLogger(__name__)


class LiteLLMModel(ImageTextToTextModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._model = get_model_name(config)
        if config.get("enable_debug", False):
            litellm._turn_on_debug()
        logger.info(f"Loaded LiteLLM class with model {self._model}")

    def generate(
        self, prompt: inputs.Chat | list[inputs.Chat], output_schema: None, **kwargs
    ):
        if not isinstance(prompt, list):
            response = litellm.completion(
                model=self._model,
                messages=self._convert_image_to_image_url(prompt.messages),
                response_format=output_schema,
            )
            output = response.choices[0].message.content
        else:
            responses = litellm.batch_completion(
                model=self._model,
                messages=[self._convert_image_to_image_url(p.messages) for p in prompt],
                response_format=output_schema,
            )
            output = [response.choices[0].message.content for response in responses]
        return self.parse_generation(output, output_schema)

    def _convert_image_to_image_url(self, messages: list):
        for message in messages:
            if isinstance(message["content"], list):
                for i, content in enumerate(message["content"]):
                    if content.get("type", None) == "image":
                        new_image_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{content.get('image').image_format};base64,{content.get('image').image_str}"
                            },
                        }
                        message["content"][i] = new_image_content
        return messages
