from abc import ABC, abstractmethod
from PIL import Image as PILImage

import torch


class ImageTextToImageModel(ABC):
    """
    Abstract base class for a edit generation model.

    This class defines the interface that all edit generation models should implement.
    """

    @abstractmethod
    def __init__(self, config: dict):
        self._config = config
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available. Please check your GPU configuration."
            )
        self._device = "cuda"

    @abstractmethod
    def generate(self, prompt: str, image: str) -> PILImage.Image:
        """
        Generates an image from a prompt and an image.
        Args:
            prompt: The prompt to generate the image from.
            image: The image to generate the image from.
        Returns:
            PILImage.Image: The generated image.
        """
        pass

    @property
    def config(self) -> dict:
        """Return the model configuration."""
        return self._config

    @property
    def device(self) -> str:
        """Return the model device."""
        return self._device
