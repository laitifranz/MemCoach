from src.models.image_text_to_text.model_factory import get_model
from src.pipelines.data.feedback_pairs_dataset import FeedbackPairsDataset
from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder
from src.pipelines.method.input.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.pipelines.method.core.activation_steer import SteeringActivationsExtractor
from src.utils._runtime_paths import resolve_dataset_image_path

import torch
from torch.utils.data import DataLoader
import os
import json
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrainingManager:
    def __init__(self, config: dict):
        self._config = config
        self._model = get_model(self._config["model"])
        self._model_id = self._config["model"]["name"]
        self._module = self._config["activation_settings"]["module"]
        self._prompt_builder = PromptBuilder(
            {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_PROMPT[self._config["prompt"]["version"]],
            }
        )
        self._num_hidden_layers = self._model.num_text_hidden_layers

    def prepare_support_set_dataloader(self):
        feedback_pairs_dataset = FeedbackPairsDataset(
            self._config["data"], split="train"
        )
        grouped_dataset_with_criteria = (
            feedback_pairs_dataset.group_by_scene_id_with_criteria()
        )

        dataloader_support_set = DataLoader(
            grouped_dataset_with_criteria,
            batch_size=self._config["model"].get("batch_size", 8),
            shuffle=True,
            generator=torch.Generator(device="cpu"),
            collate_fn=self._collate_fn_support_set,
        )
        return dataloader_support_set

    def _collate_fn_support_set(self, batch):
        prompts = []
        for feedback_pair in batch:
            prompts.append(
                self._prompt_builder.get_prompt(
                    [
                        (
                            "Image:",
                            resolve_dataset_image_path(
                                self._config["data"]["dataset_name"],
                                feedback_pair["source_image_path"],
                            ),
                        ),
                    ],
                    assistant_prompt=", ".join(feedback_pair["actions"]),
                    image_before_text=True,
                )
            )

        processed_chats = self._model.raw_processor.apply_chat_template(
            PromptBuilder.get_message_variable(prompts, unwrap_image=True),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        )

        prompt_lengths = []
        for inp in processed_chats["input_ids"]:
            # get length of prompt (without padding)
            n_pad_tokens = sum(inp == self._model.get_pad_token_id)
            prompt_len = len(inp) - n_pad_tokens
            prompt_lengths.append(prompt_len)

        processed_batch = {
            "embedded_prompts": processed_chats,
            "original_prompts": prompts,
            "prompt_lengths": prompt_lengths,
        }
        return processed_batch

    def _get_activations_from_batch(self, batch: dict):
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self._model.device,
                dtype=next(self._model.raw_model.parameters()).dtype,
            ),
        ):
            with SteeringActivationsExtractor(
                self._model, self._num_hidden_layers, self._module
            ) as full_set_actv:
                _ = self._model.raw_model(
                    **batch["embedded_prompts"].to(self._model.device)
                )
        return full_set_actv

    def extract_activations_from_support_set(
        self, support_set_dataloader: DataLoader
    ) -> dict[str, torch.Tensor]:
        actvs = {
            "mean_prompt": [[] for _ in range(self._num_hidden_layers)],
        }

        for batch in tqdm(support_set_dataloader, desc="Get activations"):
            full_set_actv = self._get_activations_from_batch(batch)

            for b_idx in range(batch["embedded_prompts"]["input_ids"].shape[0]):
                for layer in range(self._num_hidden_layers):
                    layer_acts = full_set_actv[layer][b_idx]
                    prompt_len = batch["prompt_lengths"][b_idx]
                    # Mean of all prompt tokens
                    mean_prm = layer_acts[:prompt_len, :].mean(dim=0).detach().cpu()
                    actvs["mean_prompt"][layer].append(mean_prm)

        # convert actvs to a torch tensor object (shape: [n_hidden_layers, embedding_dim])
        layer_tensors = [
            torch.stack(layer_actvs, dim=0) for layer_actvs in actvs["mean_prompt"]
        ]
        actvs_tensor = torch.stack(layer_tensors, dim=0)

        return {
            "mean_prompt": actvs_tensor,
        }

    def save_activations(self, actvs: dict[str, torch.Tensor], save_path: str):
        actvs.update({"metadata": self._config})
        os.makedirs(save_path, exist_ok=True)
        torch.save(actvs, os.path.join(save_path, "actvs_vector.pt"))
        with open(os.path.join(save_path, "metadata.json"), "w") as f:
            json.dump(self._config, f, indent=4)
        logger.info(f"Saved activations to {save_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import fire
    import yaml

    with open("config/method_steering/training/internvl3_5_8B_negative.yaml", "r") as f:
        config = yaml.safe_load(f)
    fire.Fire(TrainingManager(config=config))
