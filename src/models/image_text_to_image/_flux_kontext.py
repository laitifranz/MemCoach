from src.models.image_text_to_image._base import ImageTextToImageModel
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from PIL import Image as PILImage

import logging


class FluxKontext(ImageTextToImageModel):
    def __init__(self, config: dict):
        super().__init__(config)
        model_name = config["name"]
        self._pipe = FluxKontextPipeline.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self._pipe.to(self._device)
        logging.info(f"Loaded FluxKontext model with model name {model_name}")

    def generate(self, prompt: str, image: str) -> PILImage.Image:
        image = load_image(image)
        return self._pipe(
            image=image,
            prompt=prompt,
            guidance_scale=2.5,
            num_inference_steps=28,
            generator=torch.manual_seed(0),
        ).images[0]
