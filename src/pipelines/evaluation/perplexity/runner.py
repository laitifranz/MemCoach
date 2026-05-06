import os
import yaml
import json
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils._logging_utils import setup_logging
from src.utils._runtime_paths import (
    resolve_project_root,
    resolve_project_experiments_root,
    resolve_dataset_image_path,
)
from src.pipelines.data.feedback_pairs_dataset import FeedbackPairsDataset
from src.models.image_text_to_text.model_factory import get_model
from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder
from src.pipelines.method.input.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.pipelines.evaluation.perplexity.perplexity import compute_perplexity

setup_logging()
logger = logging.getLogger(__name__)


def main(
    config_path: str,
    overwrite: bool = False,
    include_datetime: bool = True,
):
    with open(config_path, "r") as f:
        logging.info(f"Loading config from {config_path}")
        config = yaml.safe_load(f)
        config["statistics"] = {}

    model_cfg = config["model"]
    if model_cfg.get("provider", "hf") != "hf":
        raise ValueError(
            "Perplexity evaluation supports only provider 'hf'. "
            "Got provider: %s" % model_cfg.get("provider")
        )

    task_name = config.get("task_name", "perplexity")
    path_parts = [
        str(resolve_project_root()),
        str(resolve_project_experiments_root()),
        task_name,
    ]
    if include_datetime:
        path_parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    path_parts.extend(
        [config["data"]["dataset_name"], model_cfg["name"].split("/")[-1]]
    )
    save_path = os.path.join(*path_parts)

    if os.path.exists(save_path) and not overwrite:
        logger.info("Dataset already exists, skipping")
        return

    feedback_pairs_dataset = FeedbackPairsDataset(config["data"], split="test")
    grouped_dataset_with_criteria = (
        feedback_pairs_dataset.group_by_scene_id_with_criteria()
    )
    dataloader = DataLoader(
        grouped_dataset_with_criteria,
        batch_size=model_cfg.get("batch_size", 8),
        shuffle=False,
        collate_fn=lambda x: x,
    )

    config["statistics"]["total_dataset_size"] = len(grouped_dataset_with_criteria)
    logger.info("Processing %s samples", len(grouped_dataset_with_criteria))

    prompt_version = config.get("prompt", {}).get("version", "inference_prompt")
    prompt_builder = PromptBuilder(
        {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT[prompt_version],
        }
    )

    use_steering = "activation_settings" in config
    if use_steering:
        from src.pipelines.method.core.inference_manager import InferenceManager

        inference_manager = InferenceManager(config)
        model = None
    else:
        inference_manager = None
        model = get_model(config["model"])

    processed_dataset = []
    for batch in tqdm(dataloader, desc="Perplexity"):
        for entry in batch:
            feedback = config.get("empty_prompt", entry["actions"])
            if isinstance(feedback, list):
                feedback = ", ".join(feedback)
            source_image_path = resolve_dataset_image_path(
                config["data"]["dataset_name"],
                entry["source_image_path"],
            )
            prompt = prompt_builder.get_prompt(
                text_image_pairs=[("Image:", source_image_path)],
                assistant_prompt=feedback,
                image_before_text=True,
            )
            if use_steering:
                ppl = inference_manager.compute_perplexity(prompt)
            else:
                ppl = compute_perplexity(model, prompt)
            processed_dataset.append(
                {
                    "entry_input": entry,
                    "perplexity": ppl,
                }
            )

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "dataset.jsonl"), "w") as f:
        for item in processed_dataset:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(config, f, indent=4)
    logger.info("Saved dataset to %s", save_path)


if __name__ == "__main__":
    from dotenv import load_dotenv
    import fire

    load_dotenv()
    fire.Fire(main)
