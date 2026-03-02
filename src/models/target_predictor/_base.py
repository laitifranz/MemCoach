from abc import ABC, abstractmethod

import torch
from PIL import Image as PILImage


class BaseTargetPredictor(ABC):
    def __init__(self, config: dict):
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def predict(self, image_path: str | list[str] | PILImage.Image) -> torch.Tensor:
        """
        Predicts the target score for the given image(s).
        Args:
            image_path: The path to the image(s) or the image(s) to predict the target score for.
        Returns:
            torch.Tensor: The predicted target score.
        """
        pass

    @property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> str:
        return self._device
