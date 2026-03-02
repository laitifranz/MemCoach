from abc import ABC, abstractmethod
from typing import Any, Optional

from outlines import inputs
import torch

from src.models.image_text_to_text.utils.parsers import parse_output
from pydantic import BaseModel


class ImageTextToTextModel(ABC):
    """
    Abstract base class for Multimodal Large Language Models using Outlines.
    Defines a standard interface for all MLLM implementations using Outlines.
    """

    def __init__(self, config: dict):
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def generate(
        self,
        prompt: inputs.Chat | list[inputs.Chat],
        output_schema: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """
        Generates a response from a prompt or list of prompts.
        Args:
            prompt (inputs.Chat | list[inputs.Chat]): The input prompt or list of prompts.
            output_schema (Optional[Any]): The schema for structured output or list of schemas for structured output.
            **kwargs: Additional keyword arguments for generation.
        Returns:
            Any: The generated output, which can be a string for text generation or a list of strings for text generation
                 or a list of Pydantic BaseModel instances for structured output.
        """
        pass

    def parse_generation(
        self, generation: str | list[str], output_schema: BaseModel | None
    ) -> str | dict:
        """
        Parses the generation from the model.
        Args:
            generation: The generation from the model.
            output_schema: The schema for structured output.
        Returns:
            str | dict: The parsed generation.
        """
        if isinstance(generation, list):
            return [parse_output(g, output_schema) for g in generation]
        else:
            return parse_output(generation, output_schema)

    @property
    def config(self) -> dict:
        """Return the model configuration."""
        return self._config

    @property
    def device(self) -> str:
        """Return the model device."""
        return self._device
