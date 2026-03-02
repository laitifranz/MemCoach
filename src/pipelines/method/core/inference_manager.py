from src.models.image_text_to_text.model_factory import get_model
from src.pipelines.data.feedback_pairs_dataset import FeedbackPairsDataset
from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder
from src.pipelines.method.input.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.pipelines.method.core.activation_steer import SteeringActivationsInjector
from src.utils._runtime_paths import resolve_project_relative_path

import torch
from torch.utils.data import DataLoader
import logging
from outlines import inputs

logger = logging.getLogger(__name__)


class InferenceManager:
    def __init__(self, config: dict):
        self._config = config
        self._steering_vector = self._load_config_and_steering_vector()
        self._model = get_model(self._config["model"])
        self._model_id = self._config["model"]["name"]
        self._module = self._config["activation_settings"]["module"]
        self._prompt_builder = PromptBuilder(
            {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_PROMPT[self._config["prompt"]["version"]],
            }
        )
        self._coeff = self._config["activation_settings"]["coeff"]
        self._target_layer = self._config["activation_settings"]["target_layer"]

    def _load_config_and_steering_vector(
        self, aggregation: str = "mean_prompt"
    ) -> torch.Tensor:
        _config_actvs = self._config["activation_settings"]
        if "steering_vector_file" in _config_actvs:
            path = str(
                resolve_project_relative_path(_config_actvs["steering_vector_file"])
            )
            logger.info(f"Loading pre-built steering vector from {path}")
            return torch.load(path, weights_only=True)
        assert "negative" in _config_actvs["negative_actvs_file"]
        assert "positive" in _config_actvs["positive_actvs_file"]

        neg_path = str(
            resolve_project_relative_path(_config_actvs["negative_actvs_file"])
        )
        pos_path = str(
            resolve_project_relative_path(_config_actvs["positive_actvs_file"])
        )
        neg: dict = torch.load(neg_path)
        pos: dict = torch.load(pos_path)

        assert (
            neg["metadata"]["model"]["provider"] == pos["metadata"]["model"]["provider"]
        )
        assert neg["metadata"]["model"]["name"] == pos["metadata"]["model"]["name"]
        assert (
            neg["metadata"]["activation_settings"]["module"]
            == pos["metadata"]["activation_settings"]["module"]
        )

        self._config["model"].update(
            {
                "provider": neg["metadata"]["model"]["provider"],
                "name": neg["metadata"]["model"]["name"],
            }
        )
        self._config["activation_settings"].update(
            {"module": neg["metadata"]["activation_settings"]["module"]}
        )

        steering_vector = (
            pos[aggregation] - neg[aggregation]
        )  # shape: (layers, samples, embedding_dim)
        steering_vector = steering_vector.mean(dim=1)  # shape: (layers, embedding_dim)

        return steering_vector

    def prepare_support_set_dataloader(self):
        feedback_pairs_dataset = FeedbackPairsDataset(
            self._config["data"], split="test"
        )
        grouped_dataset_with_criteria = (
            feedback_pairs_dataset.group_by_scene_id_with_criteria()
        )

        dataloader_support_set = DataLoader(
            grouped_dataset_with_criteria,
            batch_size=self._config["model"].get("batch_size", 8),
            shuffle=True,
            generator=torch.Generator(device="cpu"),
            collate_fn=lambda x: x,
        )
        return dataloader_support_set

    def generate(
        self, prompt: inputs.Chat | list[inputs.Chat], output_schema=None, **kwargs
    ):
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self._model.device,
                dtype=next(self._model.raw_model.parameters()).dtype,
            ),
        ):
            with SteeringActivationsInjector(
                self._model,
                self._steering_vector,
                self._target_layer,
                self._module,
                self._coeff,
            ):
                output = self._model.generate(prompt, output_schema, **kwargs)
        return output


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import fire
    import yaml

    with open("config/method_steering/inference/internvl3_5_8B.yaml", "r") as f:
        config = yaml.safe_load(f)
    fire.Fire(InferenceManager(config=config))
