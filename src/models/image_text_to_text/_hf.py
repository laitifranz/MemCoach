import outlines
from outlines import inputs
import torch
from transformers import set_seed

from src.models.image_text_to_text._base import ImageTextToTextModel
from src.utils._runtime_paths import get_model_name

import logging

logger = logging.getLogger(__name__)


class HfModel(ImageTextToTextModel):
    def __init__(self, config: dict):
        super().__init__(config)
        self._model_name = get_model_name(config)
        self._max_new_tokens = config.get("max_new_tokens", 2048)
        self._temperature = config.get("temperature", 0.001)
        set_seed(42)
        self._model = self._build_model()
        logger.info(f"Loaded HF class with model {self._model_name}")

    def generate(
        self, prompt: inputs.Chat | list[inputs.Chat], output_schema=None, **kwargs
    ):
        with torch.autocast(device_type=self._device), torch.no_grad():
            if isinstance(prompt, list):
                output = self._model.batch(
                    prompt,
                    output_type=output_schema,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=True,
                    temperature=self._temperature,
                    **kwargs,
                )
            else:
                output = self._model(
                    prompt,
                    output_type=output_schema,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=True,
                    temperature=self._temperature,
                    **kwargs,
                )
        return self.parse_generation(output, output_schema)

    @property
    def raw_model(self):
        return self._model.model

    @property
    def raw_processor(self):
        return self._model.processor

    @property
    def model_id(self):
        return self._model_name

    @property
    def num_text_hidden_layers(self):
        return self._model.model.config.text_config.num_hidden_layers

    @property
    def get_pad_token_id(self):
        return self._model.processor.tokenizer.pad_token_id

    def _get_model_and_processor(self):
        def _load_qwen2_5_vl():
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            return Qwen2_5_VLForConditionalGeneration, AutoProcessor

        def _load_llava():
            from transformers import LlavaForConditionalGeneration, AutoProcessor

            logger.warning(
                "To fix chat_template issue, copy the content of chat_template.json to the key chat_template in tokenizer_config.json"
            )
            return LlavaForConditionalGeneration, AutoProcessor

        def _load_llava_onevision():
            from transformers import (
                LlavaOnevisionForConditionalGeneration,
                AutoProcessor,
            )

            return LlavaOnevisionForConditionalGeneration, AutoProcessor

        def _load_idefics3():
            from transformers import AutoModelForImageTextToText, AutoProcessor

            logger.warning(
                "To fix chat_template issue, copy the content of chat_template.json to the key chat_template in tokenizer_config.json"
            )
            return AutoModelForImageTextToText, AutoProcessor

        def _load_default():
            from transformers import AutoModelForImageTextToText, AutoProcessor

            return AutoModelForImageTextToText, AutoProcessor

        # model -> loader, processor_kwargs, model_kwargs, avoid loading unneeded models and processors
        MODEL_MAPPING = {
            "Qwen/Qwen2.5-VL-7B-Instruct": {
                "loader": _load_qwen2_5_vl,
                "processor_kwargs": {
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
            },
            "llava-hf/llava-onevision-qwen2-7b-ov-hf": {
                "loader": _load_llava_onevision,
            },
            "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf": {
                "loader": _load_llava_onevision,
            },
            "HuggingFaceM4/Idefics3-8B-Llama3": {
                "loader": _load_idefics3,
            },
            "default": {
                "loader": _load_default,
            },
            # dummy models for testing
            "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration": {
                "loader": _load_qwen2_5_vl,
            },
            "trl-internal-testing/tiny-LlavaForConditionalGeneration": {
                "loader": _load_llava,
            },
        }

        model_name = self._model_name
        spec = MODEL_MAPPING.get(model_name, MODEL_MAPPING["default"])

        model_cls, processor_cls = spec["loader"]()
        processor_kwargs = spec.get("processor_kwargs", {})
        model_kwargs = spec.get("model_kwargs", {})

        return model_cls, processor_cls, model_kwargs, processor_kwargs

    def _build_model(self):
        model, processor, model_kwargs, processor_kwargs = (
            self._get_model_and_processor()
        )
        if not model_kwargs:  # default model kwargs
            model_kwargs = {
                "dtype": torch.bfloat16,
                "device_map": "auto",
            }
            if self._device == "cuda" and self.config.get(
                "use_flash_attention_2", False
            ):
                model_kwargs["attn_implementation"] = "flash_attention_2"

        model = outlines.from_transformers(
            model.from_pretrained(self._model_name, **model_kwargs),
            processor.from_pretrained(self._model_name, **processor_kwargs),
        )

        return model
