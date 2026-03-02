import json
import os
import logging
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


from src.pipelines.membench_gen.constr_data_gen.input.schemas import (
    ConstrainedActionListOutput,
)
from src.pipelines.membench_gen.constr_data_gen.input.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from src.utils._sharding_via_slurm import get_start_stop_index
from src.utils._logging_utils import setup_logging
from src.pipelines.data.scene_pairs_dataset import ScenePairsDataset
from src.models.image_text_to_text.model_factory import get_model
from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder
from src.utils._runtime_paths import (
    build_stage_save_path,
    load_yaml_config,
    resolve_dataset_image_path,
)

setup_logging()
logger = logging.getLogger(__name__)


def main(config_path: str, overwrite: bool = False, dry_run: bool = False):

    config = load_yaml_config(config_path)
    config["statistics"] = {}

    save_path = build_stage_save_path(config)
    if os.path.exists(save_path):
        if overwrite:
            logger.info("Dataset already exists, overwriting")
        else:
            logger.info("Dataset already exists, skipping generation")
            return

    scene_pairs_dataset = ScenePairsDataset(config["data"])
    scene_dataloader = DataLoader(
        scene_pairs_dataset,
        batch_size=config["model"].get("batch_size", 8),
        shuffle=False,
        collate_fn=lambda x: x,
        sampler=SubsetRandomSampler(
            indices=range(*get_start_stop_index(len(scene_pairs_dataset)))
        ),
    )

    config["statistics"]["total_dataset_size"] = len(scene_pairs_dataset)
    config["statistics"]["subset_processed_size"] = len(scene_dataloader) * config[
        "model"
    ].get("batch_size", 8)

    if not dry_run:
        model = get_model(config["model"])

    prompt_builder = PromptBuilder(
        {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT[config["prompt"]["version"]],
        }
    )
    output_schema = ConstrainedActionListOutput

    processed_dataset = []

    if not dry_run:
        for batch_scene_pairs in tqdm(scene_dataloader, desc="Processing scenes"):
            prompts = []
            for scene_pair in batch_scene_pairs:
                prompts.append(
                    prompt_builder.get_prompt(
                        [
                            (
                                "Image A:",
                                resolve_dataset_image_path(
                                    config["data"]["dataset_name"],
                                    scene_pair["source_image_path"],
                                ),
                            ),
                            (
                                "Image B:",
                                resolve_dataset_image_path(
                                    config["data"]["dataset_name"],
                                    scene_pair["target_image_path"],
                                ),
                            ),
                        ]
                    )
                )
            batch_results = model.generate(prompts, output_schema=output_schema)
            for scene_pair, result in zip(batch_scene_pairs, batch_results):
                scene_pair.update({"actions": result["actions"]})
                processed_dataset.append(scene_pair)

    if not dry_run:
        if os.getenv("SLURM_ARRAY_TASK_ID"):
            filename_suffix = f"_{os.getenv('SLURM_ARRAY_TASK_ID')}"
        else:
            filename_suffix = ""
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"dataset{filename_suffix}.jsonl"), "w") as f:
            for item in processed_dataset:
                f.write(json.dumps(item) + "\n")
        with open(os.path.join(save_path, f"metadata{filename_suffix}.json"), "w") as f:
            json.dump(config, f, indent=4)
    else:
        logger.info("DRY RUN: Skipping dataset generation")
        import pprint

        logger.info("Config:\n%s", pprint.pformat(config, indent=4, width=120))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    import fire

    fire.Fire(main)
