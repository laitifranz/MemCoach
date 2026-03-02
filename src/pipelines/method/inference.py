import os
import logging
from tqdm import tqdm
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from src.models.image_text_to_text.utils.prompt_builder import PromptBuilder
from src.pipelines.method.input.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.utils._logging_utils import setup_logging
from src.pipelines.method.core.inference_manager import InferenceManager
from src.pipelines.method.input.schemas import ConstrainedActionListOutput
from src.utils._runtime_paths import build_stage_save_path, resolve_dataset_image_path

setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=to_absolute_path("config/method_steering/inference"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, cfg.runtime.strict_config)  # enable creation of new keys

    config = OmegaConf.to_container(
        cfg, resolve=True
    )  # in case I specify a key ['preset']
    config["statistics"] = {}

    inference_manager = InferenceManager(config)
    support_set_dataloader = inference_manager.prepare_support_set_dataloader()

    prompt_builder = PromptBuilder(
        {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT[config["prompt"]["version"]],
        }
    )

    config["statistics"]["support_set_size"] = len(support_set_dataloader) * config[
        "model"
    ].get("batch_size", 8)

    save_path = build_stage_save_path(
        config, include_datetime=cfg.runtime.include_datetime
    )

    if os.path.exists(save_path):
        if cfg.runtime.overwrite:
            logger.info("Dataset already exists, overwriting")
        else:
            logger.info("Dataset already exists, skipping generation")
            return

    if not cfg.runtime.dry_run:
        processed_dataset = []

        for batch_scene_pairs in tqdm(
            support_set_dataloader, desc="Processing support set"
        ):
            prompts = []
            for scene_pair in batch_scene_pairs:
                prompts.append(
                    prompt_builder.get_prompt(
                        [
                            (
                                "Image:",
                                resolve_dataset_image_path(
                                    config["data"]["dataset_name"],
                                    scene_pair["source_image_path"],
                                ),
                            ),
                        ],
                        image_before_text=True,
                    )
                )

            batch_results = inference_manager.generate(
                prompts,
                output_schema=ConstrainedActionListOutput
                if config["prompt"]["schema"]
                else None,
            )

            for scene_pair, result in zip(batch_scene_pairs, batch_results):
                scene_pair.update({"actions": result})
                processed_dataset.append(scene_pair)

    if not cfg.runtime.dry_run:
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
        logging.info(f"Saved dataset to {save_path}")
    else:
        logger.info("DRY RUN: Skipping dataset generation")
        print(OmegaConf.to_yaml(config))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
