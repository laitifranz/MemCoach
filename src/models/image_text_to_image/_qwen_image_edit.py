import logging

import torch

from src.models.image_text_to_image._base import ImageTextToImageModel
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import load_image
from PIL import Image as PILImage
import math


class QwenImageEdit(ImageTextToImageModel):
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = config["model"]["name"]
        weight_name = config["model"]["weight_name"]

        # from https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        self._pipe = DiffusionPipeline.from_pretrained(
            model_name, scheduler=scheduler, torch_dtype=torch.bfloat16
        ).to(self._device)
        self._pipe.load_lora_weights(model_name, weight_name=weight_name)
        logging.info(
            f"Loaded QwenImageEdit model with model name {model_name} and weight name {weight_name}"
        )

    def generate(self, prompt: str, image: str) -> PILImage.Image:
        image = load_image(image)
        with torch.inference_mode():
            return self._pipe(
                image=image,
                prompt=prompt,
                negative_prompt=" ",
                width=1024,
                height=1024,
                num_inference_steps=8,
                true_cfg_scale=1.0,
                generator=torch.manual_seed(0),
            ).images[0]
