import os
import yaml
from datetime import datetime
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import json

from src.utils._logging_utils import setup_logging
from src.pipelines.data.feedback_pairs_dataset import FeedbackPairsDataset
from src.models.image_text_to_image.model_factory import get_model as get_editing_model
from src.models.target_predictor.model_factory import (
    get_model as get_target_predictor_model,
)
from src.utils._runtime_paths import (
    resolve_project_root,
    resolve_project_experiments_root,
    resolve_project_dataset_root,
)

setup_logging()
logger = logging.getLogger(__name__)


def main(
    config_path: str,
    overwrite: bool = False,
    dry_run: bool = False,
    include_datetime: bool = True,
):
    with open(config_path, "r") as f:
        logging.info(f"Loading config from {config_path}")
        config = yaml.safe_load(f)
        config["statistics"] = {}

    save_path = os.path.join(
        resolve_project_root(),
        resolve_project_experiments_root(),
        config["task_name"],
        datetime.now().strftime("%Y%m%d_%H%M%S") if include_datetime else "",
        config["data"]["dataset_name"],
        config["editing_model"]["model"]["name"].split("/")[-1],
        config["target_predictor"]["model"]["provider"],
    )

    if os.path.exists(save_path):
        if overwrite:
            logger.info("Dataset already exists, overwriting")
        else:
            logger.info("Dataset already exists, skipping generation")
            return

    feedback_pairs_dataset = FeedbackPairsDataset(config["data"], split="test")
    grouped_dataset_with_criteria = (
        feedback_pairs_dataset.group_by_scene_id_with_criteria()
    )
    dataloader_feedback_pairs = DataLoader(
        grouped_dataset_with_criteria,
        batch_size=config["editing_model"]["model"].get("batch_size", 8),
        shuffle=False,
        collate_fn=lambda x: x,
    )

    config["statistics"]["total_dataset_size"] = len(grouped_dataset_with_criteria)
    logging.info(f"Processing {len(grouped_dataset_with_criteria)} samples")

    editing_model = get_editing_model(config["editing_model"]["model"])
    target_predictor_model = get_target_predictor_model(config["target_predictor"])

    processed_dataset = []
    for batch in tqdm(dataloader_feedback_pairs, desc="Processing feedback pairs"):
        for entry in batch:
            feedback = config["editing_model"].get("empty_prompt", entry["actions"])
            if isinstance(feedback, list):
                feedback = ", ".join(feedback)
            source_image_path = os.path.join(
                resolve_project_root(),
                resolve_project_dataset_root(),
                config["data"]["dataset_name"],
                entry["source_image_path"],
            )
            # breakpoint()
            generated_image = editing_model.generate(feedback, source_image_path)
            target_score = target_predictor_model.predict(generated_image)
            base_name = os.path.splitext(entry["source_image_path"])[0]
            generated_image_path = os.path.join(
                save_path, "edit_imgs", f"{base_name}_edited.jpg"
            )
            os.makedirs(os.path.dirname(generated_image_path), exist_ok=True)
            generated_image.save(generated_image_path)
            os.path.exists(
                os.path.join(save_path, "edit_imgs", f"{base_name}_source.jpg")
            ) or os.symlink(
                source_image_path,
                os.path.join(save_path, "edit_imgs", f"{base_name}_source.jpg"),
            )
            processed_dataset.append(
                {
                    "entry_input": entry,
                    "generated_image_path": os.path.basename(generated_image_path),
                    "target_score": target_score.item(),
                }
            )

    with open(os.path.join(save_path, "dataset.jsonl"), "w") as f:
        for item in processed_dataset:
            f.write(json.dumps(item) + "\n")
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(config, f, indent=4)
    logging.info(f"Saved dataset to {save_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import fire

    fire.Fire(main)
