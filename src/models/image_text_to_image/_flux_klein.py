from src.models.image_text_to_image._base import ImageTextToImageModel
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from PIL import Image as PILImage

import logging


class Flux2Klein(ImageTextToImageModel):
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = config["name"]
        self._pipe = Flux2KleinPipeline.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self._pipe.to(self._device)
        logging.info(f"Loaded Flux2Klein model with model name {model_name}")

    def generate(self, prompt: str, image: str) -> PILImage.Image:
        image = load_image(image)
        return self._pipe(
            image=image,
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.manual_seed(0),
        ).images[0]
